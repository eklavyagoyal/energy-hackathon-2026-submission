"""Freeze a deterministic case118 one-line layout into fixtures/grid_topology.json.

case118 ships no bus geodata, so we lay the network out once with a Kamada-Kawai embedding (clean
for a 118-bus mesh), normalize to a 0..1000 viewport, and tag each bus (slack / generator / load).
The frontend renders THIS so the hero grid is the real IEEE 118-bus topology, not a decorative
mock. Edges carry their native ids so per-step loadings (line_<i> / trafo_<i>) map straight on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import networkx as nx  # noqa: E402

from src.engine.network import native_index  # noqa: E402
from src.grid.loader import Case118Loader  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
VIEW = 1000.0
PAD = 40.0


def _normalize(pos: dict) -> dict:
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    sx = (VIEW - 2 * PAD) / (maxx - minx or 1.0)
    sy = (VIEW - 2 * PAD) / (maxy - miny or 1.0)
    return {n: (PAD + (x - minx) * sx, PAD + (y - miny) * sy) for n, (x, y) in pos.items()}


def main() -> None:
    FIXTURES.mkdir(exist_ok=True)
    net = Case118Loader().load()

    g = nx.Graph()
    g.add_nodes_from(int(b) for b in net.bus.index)
    for _, r in net.line.iterrows():
        g.add_edge(int(r.from_bus), int(r.to_bus))
    for _, r in net.trafo.iterrows():
        g.add_edge(int(r.hv_bus), int(r.lv_bus))
    pos = _normalize(nx.kamada_kawai_layout(g))

    slack_buses = {native_index(b) for b in net.ext_grid.bus}
    gen_buses = {native_index(b) for b in net.gen.bus}
    load_buses = {native_index(b) for b in net.load.bus}

    def role(b):
        if b in slack_buses:
            return "slack"
        if b in gen_buses:
            return "gen"
        if b in load_buses:
            return "load"
        return "bus"

    buses = []
    for b in net.bus.index:
        nb = native_index(b)
        x, y = pos[int(b)]
        buses.append({"id": nb, "x": round(x, 1), "y": round(y, 1),
                      "vn_kv": round(float(net.bus.at[b, "vn_kv"]), 1), "role": role(nb)})

    edges = []
    for i, r in net.line.iterrows():
        edges.append({"id": f"line_{native_index(i)}", "kind": "line",
                      "from": native_index(r.from_bus), "to": native_index(r.to_bus)})
    for i, r in net.trafo.iterrows():
        edges.append({"id": f"trafo_{native_index(i)}", "kind": "trafo",
                      "from": native_index(r.hv_bus), "to": native_index(r.lv_bus)})

    out = FIXTURES / "grid_topology.json"
    out.write_text(json.dumps({"view": VIEW, "buses": buses, "edges": edges}, indent=2))
    print(f"froze {len(buses)} buses, {len(edges)} edges to {out}")
    print(f"  slack={len(slack_buses)} gen={len(gen_buses)} load={len(load_buses)}")


if __name__ == "__main__":
    main()
