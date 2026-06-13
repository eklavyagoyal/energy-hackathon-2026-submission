"""PyPSAEurLoader: hand-map a PyPSA / PyPSA-Eur network onto pandapower (U12).

There is NO pandapower<-PyPSA converter (checked: pandapower.converter ships none), so this maps the
PyPSA tables by hand. Two intake formats, neither needs pandapower's optional deps:

  - a CSV folder (PyPSA's `network.export_to_csv_folder()` layout): read with pandas alone, so the
    loader is testable WITHOUT pypsa installed.
  - a netCDF `.nc` file: read via `import pypsa` (guarded; a clear, actionable error if pypsa is
    absent), then the same hand-map runs on the in-memory tables.

KTD3 (the portability cornerstone): pandapower 3.4 crashes on string bus indices
(auxiliary._preserve_dtypes), so we create INT-indexed buses and keep a str<->int bus-name map IN
THE LOADER (and attach it to the net as net["bus_name_map"]). The engine stays int-indexed; results
are translated back to PyPSA bus names at the boundary, never inside the solver.

KTD6 (the hand-map rules):
  - line thermal limit:  max_i_ka = s_nom / (sqrt(3) * v_nom)
  - transformer reactance: vk_percent = sqrt(r^2 + x^2) * 100, vkr_percent = r * 100  (r,x p.u.)
  - one ext_grid per connected sub-network: every island needs a slack or the power flow has no
    angle reference; we promote a slack-tagged generator (else the largest) per component.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
import pandapower as pp
import pandapower.topology as top

logger = logging.getLogger(__name__)

SQRT3 = math.sqrt(3.0)
_CSV_TABLES = ("buses", "lines", "transformers", "generators", "loads")


def _read_csv_folder(folder: Path) -> dict:
    """Read the PyPSA CSV-folder export. buses.csv is required; the rest are optional (a network
    may have no transformers, etc.). Returns {table_name: DataFrame indexed by the 'name' column}."""
    tables: dict = {}
    for name in _CSV_TABLES:
        path = folder / f"{name}.csv"
        if path.is_file():
            df = pd.read_csv(path)
            if "name" in df.columns:
                df = df.set_index("name")
            tables[name] = df
        else:
            tables[name] = pd.DataFrame()
    if tables["buses"].empty:
        raise ValueError(f"{folder}/buses.csv is missing or empty; not a PyPSA CSV export")
    return tables


def _tables_from_pypsa(path: Path) -> dict:
    """Load a netCDF via pypsa (optional dep) and expose its component DataFrames in the same shape
    as the CSV reader. Import is local so pypsa is only required for the .nc path."""
    try:
        import pypsa  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "reading a PyPSA .nc network needs the 'pypsa' package (pip install pypsa). "
            "Alternatively point GRID_DATA_PATH at a CSV-folder export, which needs no extra deps."
        ) from exc
    n = pypsa.Network(str(path))
    return {
        "buses": n.buses, "lines": n.lines, "transformers": n.transformers,
        "generators": n.generators, "loads": n.loads,
    }


def _f(row, col, default=0.0) -> float:
    """Read a float cell with a default for missing columns/NaN (PyPSA tables vary by network)."""
    if col not in row or pd.isna(row[col]):
        return float(default)
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return float(default)


def _s(row, col, default: str = "") -> str:
    """Read a string cell with a default for missing columns/NaN."""
    if col not in row or pd.isna(row[col]):
        return default
    return str(row[col])


def _carrier_cost(carrier: str, gen_type: str = "") -> float:
    """Deterministic fallback marginal costs for PyPSA exports that do not ship costs.

    The values are not market claims; they only make OPF paths structurally available for the demo
    while preserving the expected dispatch order: renewables cheap, flexible thermal expensive.
    """
    key = f"{carrier} {gen_type}".lower()
    if "solar" in key or "pv" in key or "wind" in key:
        return 1.0
    if "hydro" in key:
        return 8.0
    if "nuclear" in key:
        return 15.0
    if "lignite" in key or "coal" in key:
        return 35.0
    if "gas" in key or "ccgt" in key or "ocgt" in key:
        return 80.0
    return 50.0


class PyPSAEurLoader:
    """Load a PyPSA / PyPSA-Eur network as a pandapower net. See module docstring for the format and
    the KTD3/KTD6 mapping rules. The str<->int bus map lives on the instance after load()."""

    name = "pypsa_eur"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.bus_name_to_idx: dict[str, int] = {}
        self.idx_to_bus_name: dict[int, str] = {}

    # ----- public -----------------------------------------------------------
    def load(self) -> "pp.pandapowerNet":
        if self.path.is_dir():
            tables = _read_csv_folder(self.path)
        elif self.path.suffix.lower() == ".nc":
            tables = _tables_from_pypsa(self.path)
        else:
            raise ValueError(
                f"PyPSA path {self.path} is neither a CSV folder nor a .nc file; "
                "export with network.export_to_csv_folder() or pass the netCDF"
            )
        net = self._build(tables)
        self._ensure_slack_per_component(net, tables["generators"])
        net["bus_name_map"] = dict(self.idx_to_bus_name)
        logger.info(
            "loaded PyPSA-Eur %s: %d buses, %d lines, %d trafos, %d gens, %d ext_grids",
            self.path.name, len(net.bus), len(net.line), len(net.trafo),
            len(net.gen), len(net.ext_grid),
        )
        return net

    def native_bus(self, idx: int) -> str:
        """Translate a pandapower int bus index back to its PyPSA bus name (boundary use)."""
        return self.idx_to_bus_name.get(int(idx), str(idx))

    # ----- mapping -----------------------------------------------------------
    def _build(self, tables: dict) -> "pp.pandapowerNet":
        net = pp.create_empty_network(name="pypsa_eur")
        buses, lines = tables["buses"], tables["lines"]
        trafos, gens, loads = tables["transformers"], tables["generators"], tables["loads"]

        for bus_name, row in buses.iterrows():
            idx = pp.create_bus(net, vn_kv=_f(row, "v_nom", 380.0), name=str(bus_name))
            self.bus_name_to_idx[str(bus_name)] = int(idx)
            self.idx_to_bus_name[int(idx)] = str(bus_name)
            net.bus.at[idx, "x"] = _f(row, "x", float("nan"))
            net.bus.at[idx, "y"] = _f(row, "y", float("nan"))
            if "country" in row and not pd.isna(row["country"]):
                net.bus.at[idx, "country"] = str(row["country"])

        for line_name, row in lines.iterrows():
            b0, b1 = self._bus(row, "bus0"), self._bus(row, "bus1")
            if b0 is None or b1 is None:
                continue
            v_nom = _f(row, "v_nom", net.bus.at[b0, "vn_kv"])
            s_nom = _f(row, "s_nom", 0.0)
            # KTD6: thermal limit from apparent-power rating. Fall back to a high cap if s_nom absent
            # so the line is never the artificial binding constraint.
            max_i_ka = s_nom / (SQRT3 * v_nom) if s_nom > 0 and v_nom > 0 else 10.0
            b_total = _f(row, "b", 0.0)  # shunt susceptance [S]; c_nf = b / (2*pi*50) * 1e9
            c_nf = (b_total / (2.0 * math.pi * 50.0)) * 1e9 if b_total else 0.0
            idx = pp.create_line_from_parameters(
                net, b0, b1, length_km=1.0,
                r_ohm_per_km=_f(row, "r", 0.01), x_ohm_per_km=_f(row, "x", 0.1),
                c_nf_per_km=max(c_nf, 0.0), max_i_ka=max_i_ka, name=str(line_name),
            )
            net.line.at[idx, "pypsa_name"] = str(line_name)
            net.line.at[idx, "v_nom"] = v_nom
            net.line.at[idx, "s_nom"] = s_nom

        for tr_name, row in trafos.iterrows():
            b0, b1 = self._bus(row, "bus0"), self._bus(row, "bus1")
            if b0 is None or b1 is None:
                continue
            r, x = _f(row, "r", 0.0), _f(row, "x", 0.1)
            s_nom = _f(row, "s_nom", 0.0) or 100.0
            idx = pp.create_transformer_from_parameters(
                net, hv_bus=b0, lv_bus=b1, sn_mva=s_nom,
                vn_hv_kv=net.bus.at[b0, "vn_kv"], vn_lv_kv=net.bus.at[b1, "vn_kv"],
                vk_percent=max(math.sqrt(r * r + x * x) * 100.0, 0.1),  # KTD6
                vkr_percent=max(r * 100.0, 0.0),
                pfe_kw=0.0, i0_percent=0.0, name=str(tr_name),
            )
            net.trafo.at[idx, "pypsa_name"] = str(tr_name)
            net.trafo.at[idx, "max_loading_percent"] = 100.0

        slack_done = False
        for gen_name, row in gens.iterrows():
            bus = self._bus(row, "bus")
            if bus is None:
                continue
            control = str(row.get("control", "PV")).strip().lower()
            p_set = _f(row, "p_set", _f(row, "p_nom", 0.0))
            p_nom = _f(row, "p_nom", p_set)
            carrier = _s(row, "carrier")
            gen_type = _s(row, "type")
            if control == "slack" and not slack_done:
                idx = pp.create_ext_grid(net, bus, vm_pu=1.0, name=str(gen_name))
                cap = max(abs(p_nom), abs(p_set), 1.0)
                net.ext_grid.at[idx, "min_p_mw"] = -cap * 2.0
                net.ext_grid.at[idx, "max_p_mw"] = cap * 2.0
                net.ext_grid.at[idx, "carrier"] = carrier
                net.ext_grid.at[idx, "type"] = gen_type
                net.ext_grid.at[idx, "pypsa_name"] = str(gen_name)
                pp.create_poly_cost(net, idx, "ext_grid", cp1_eur_per_mw=_carrier_cost(carrier, gen_type))
                slack_done = True
            else:
                idx = pp.create_gen(net, bus, p_mw=p_set, vm_pu=1.0,
                                    max_p_mw=max(p_nom, p_set, 0.0), min_p_mw=0.0,
                                    controllable=True, name=str(gen_name))
                net.gen.at[idx, "carrier"] = carrier
                net.gen.at[idx, "type"] = gen_type
                net.gen.at[idx, "p_nom"] = p_nom
                net.gen.at[idx, "pypsa_name"] = str(gen_name)
                pp.create_poly_cost(net, idx, "gen", cp1_eur_per_mw=_carrier_cost(carrier, gen_type))

        for load_name, row in loads.iterrows():
            bus = self._bus(row, "bus")
            if bus is None:
                continue
            pp.create_load(net, bus, p_mw=_f(row, "p_set", 0.0),
                           q_mvar=_f(row, "q_set", 0.0), name=str(load_name))
        net["grid_dataset"] = "pypsa_eur"
        return net

    def _bus(self, row, col: str):
        """Map a PyPSA bus name reference to the pandapower int index, or None if unknown."""
        name = row.get(col)
        if name is None or pd.isna(name):
            return None
        return self.bus_name_to_idx.get(str(name))

    def _ensure_slack_per_component(self, net: "pp.pandapowerNet", gens: pd.DataFrame) -> None:
        """KTD6: every connected sub-network needs an angle reference. For each component with no
        ext_grid, promote the largest generator on it to a slack (or, failing that, its first bus)."""
        graph = top.create_nxgraph(net, respect_switches=False)
        import networkx as nx

        ext_buses = set(net.ext_grid.bus.tolist())
        for comp in nx.connected_components(graph):
            if ext_buses & comp:
                continue
            gen_here = net.gen[net.gen.bus.isin(comp)]
            if len(gen_here):
                pick = gen_here["max_p_mw"].idxmax() if "max_p_mw" in gen_here else gen_here.index[0]
                bus = int(net.gen.at[pick, "bus"])
                net.gen.drop(index=pick, inplace=True)  # becomes the slack instead of a PV gen
                pp.create_ext_grid(net, bus, vm_pu=1.0, name=f"slack_comp_{bus}")
            else:
                bus = sorted(comp)[0]
                pp.create_ext_grid(net, bus, vm_pu=1.0, name=f"slack_comp_{bus}")
            ext_buses.add(bus)
