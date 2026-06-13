"""GridLoader abstraction: the single dataset seam.

The active grid is selected by Settings.grid_dataset ("case118" or
"tso_real"). Every other module in the codebase takes a pandapowerNet
as input and never references a loader, so swapping case118 for a real
TSO dataset is a config change plus, at most, work INSIDE
TSORealLoader. Same isolation principle as the Phase 2 Screener slot
(docs/06-phase2-gridsfm.md section 1).

What may differ in the real dataset and where it is handled:
- bus count and non-sequential or string identifiers: all engine and
  battery code iterates net.bus.index and types references int | str
- multiple ext_grids: slack handling is set-based everywhere
  (src/engine/preflight.py, src/battery/verification.py)
- missing poly_cost: opf_available() guards every potential runopp
  path; the battery loop never needs OPF
- active switches: islanding checks pass respect_switches=True
- per-line thermal ratings via max_i_ka: pandapower derives
  res_line.loading_percent from max_i_ka natively, so loading-based
  logic keeps working; see the TODO below for rating normalization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandapower as pp
import pandapower.networks as pn

from src.config import Settings
from src.engine.network import apply_proportional_line_ratings, lift_voltage_profile

logger = logging.getLogger(__name__)


@runtime_checkable
class GridLoader(Protocol):
    name: str

    def load(self) -> "pp.pandapowerNet": ...


class Case118Loader:
    """Development default: the pandapower IEEE 118-bus case.

    The shipped case carries 9900 MVA placeholder thermal ratings
    (base-case loadings under 5 percent), under which no load scale can
    manufacture congestion or cascades, and a voltage profile that dips
    below 0.95 p.u. under load. The loader therefore assigns
    deterministic proportional line ratings (D59) and lifts the voltage
    profile into band (D60) once at load time, and strips geodata that
    analysis never reads to speed up the per-contingency deep copies.
    Real TSO datasets bring real ratings and a real profile and skip all
    of this.

    Rating target 30 percent base loading gives roughly 3x headroom, so
    the unstressed base and most single contingencies stay secure;
    localized stress (src/engine/network.inject_local_stress) is what
    then turns specific corridors into cascades.
    """

    name = "case118"
    rating_target_pct = 30.0

    def load(self) -> "pp.pandapowerNet":
        net = pn.case118()
        for tbl in ("bus_geodata", "line_geodata"):
            if tbl in net and len(net[tbl]):
                net[tbl] = net[tbl].iloc[0:0]
        apply_proportional_line_ratings(net, base_loading_pct=self.rating_target_pct)
        lift_voltage_profile(net)
        net["grid_dataset"] = self.name
        logger.info(
            "loaded case118: %d buses, %d lines, %d trafos, %d ext_grids "
            "(ratings target %.0f%%, voltage profile lifted)",
            len(net.bus), len(net.line), len(net.trafo), len(net.ext_grid),
            self.rating_target_pct,
        )
        return net


class TSORealLoader:
    """Placeholder for the real TSO dataset swap.

    Reads a pandapower-importable file from a configurable path. The
    format dispatch below covers the formats pandapower ships importers
    for; everything dataset-specific that cannot be known before the
    real data arrives is marked TODO.
    """

    name = "tso_real"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> "pp.pandapowerNet":
        if not self.path.exists():
            raise FileNotFoundError(
                f"GRID_DATA_PATH does not exist: {self.path}. Set GRID_DATASET=case118 "
                "for development or point GRID_DATA_PATH at the TSO export."
            )
        suffix = self.path.suffix.lower()
        if suffix == ".json":
            net = pp.from_json(str(self.path))
        elif suffix in (".xlsx", ".xls"):
            net = pp.from_excel(str(self.path))
        elif suffix in (".p", ".pickle"):
            net = pp.from_pickle(str(self.path))
        else:
            raise ValueError(
                f"unsupported TSO dataset format '{suffix}'; expected .json, .xlsx or .p"
            )

        # TODO(tso-swap): normalize thermal ratings. If lines carry only
        # max_i_ka the engine works as-is (pandapower computes
        # loading_percent from max_i_ka); if the export carries MVA
        # ratings in a side table, map them here.
        # TODO(tso-swap): validate switch table consistency
        # (net.switch.closed dtype, bus-bus switches); islanding code
        # already passes respect_switches=True.
        # TODO(tso-swap): time-series profiles. This loader returns one
        # snapshot; profile selection (which hour to load) belongs here
        # and nowhere downstream.
        # TODO(tso-swap): if poly_cost is missing, opf_available(net)
        # is False and OPF-dependent paths must stay on their
        # documented fallbacks; nothing to do here, just do not invent
        # cost data.
        logger.info(
            "loaded TSO dataset %s: %d buses, %d lines, %d trafos, %d ext_grids, %d switches",
            self.path.name,
            len(net.bus), len(net.line), len(net.trafo), len(net.ext_grid),
            len(net.switch),
        )
        return net


def get_loader(settings: Settings) -> GridLoader:
    """Loader selection. The ONLY place GRID_DATASET is interpreted."""
    if settings.grid_dataset == "case118":
        return Case118Loader()
    if settings.grid_dataset == "tso_real":
        if not settings.grid_data_path:
            raise ValueError("GRID_DATASET=tso_real requires GRID_DATA_PATH to be set")
        return TSORealLoader(settings.grid_data_path)
    if settings.grid_dataset == "pypsa_eur":
        if not settings.grid_data_path:
            raise ValueError(
                "GRID_DATASET=pypsa_eur requires GRID_DATA_PATH (a PyPSA CSV-folder export or .nc "
                "file); run scripts/download_pypsa_eur.py for a sample"
            )
        from src.grid.pypsa_eur_loader import PyPSAEurLoader  # local: optional pandas-only path
        return PyPSAEurLoader(settings.grid_data_path)
    raise ValueError(
        f"unknown GRID_DATASET {settings.grid_dataset!r}; expected 'case118', 'tso_real' or 'pypsa_eur'"
    )
