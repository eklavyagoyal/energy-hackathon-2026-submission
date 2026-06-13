"""FastAPI app: lifespan-loaded net, cached baseline, battery routes.

Mirrors the docs/07-api-contracts.md conventions on this branch: JSON
only, CORS allow-all (hackathon posture, not production), one uniform
error envelope {"error": {"code", "message"}}.
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.state import AppState
from src.config import get_settings
from src.engine.scenarios import SCENARIOS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message}}
    )


def _normalize_points(points: dict, view: float = 1000.0, pad: float = 40.0) -> dict:
    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    sx = (view - 2 * pad) / (maxx - minx or 1.0)
    sy = (view - 2 * pad) / (maxy - miny or 1.0)
    return {n: (pad + (x - minx) * sx, pad + (y - miny) * sy) for n, (x, y) in points.items()}


def _topology_from_net(net) -> dict:
    import networkx as nx

    from src.engine.network import native_index

    view = 1000.0
    graph = nx.Graph()
    graph.add_nodes_from(net.bus.index)
    for _, row in net.line.iterrows():
        graph.add_edge(row.from_bus, row.to_bus)
    for _, row in net.trafo.iterrows():
        graph.add_edge(row.hv_bus, row.lv_bus)

    small_network = len(net.bus) < 20
    has_xy = {"x", "y"} <= set(net.bus.columns) and not small_network
    if has_xy:
        points = {}
        for bus in net.bus.index:
            x, y = net.bus.at[bus, "x"], net.bus.at[bus, "y"]
            try:
                x, y = float(x), float(y)
            except (TypeError, ValueError):
                points = {}
                break
            if not (math.isfinite(x) and math.isfinite(y)):
                points = {}
                break
            points[bus] = (x, y)
    else:
        points = {}
    if not points:
        points = nx.kamada_kawai_layout(graph)
    pos = _normalize_points(points, view=view, pad=260.0 if small_network else 40.0)

    slack_buses = {native_index(b) for b in net.ext_grid.bus}
    gen_buses = {native_index(b) for b in net.gen.bus}
    load_buses = {native_index(b) for b in net.load.bus}

    def role(bus):
        if bus in slack_buses:
            return "slack"
        if bus in gen_buses:
            return "gen"
        if bus in load_buses:
            return "load"
        return "bus"

    buses = []
    for bus in net.bus.index:
        native = native_index(bus)
        x, y = pos[bus]
        buses.append({
            "id": native,
            "label": net.get("bus_name_map", {}).get(int(bus), str(native)) if isinstance(native, int) else str(native),
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "vn_kv": round(float(net.bus.at[bus, "vn_kv"]), 1),
            "role": role(native),
        })

    edges = []
    for i, row in net.line.iterrows():
        edges.append({"id": f"line_{native_index(i)}", "kind": "line",
                      "from": native_index(row.from_bus), "to": native_index(row.to_bus)})
    for i, row in net.trafo.iterrows():
        edges.append({"id": f"trafo_{native_index(i)}", "kind": "trafo",
                      "from": native_index(row.hv_bus), "to": native_index(row.lv_bus)})
    return {"view": view, "buses": buses, "edges": edges}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the net and compute the baseline ONCE at startup so the first
    # /battery/recommendations call is already warm (<10 s budget).
    settings = get_settings()
    logger.info("loading grid dataset %s ...", settings.grid_dataset)
    app.state.grid = AppState(settings)
    logger.info("startup complete; baseline cached")
    yield


app = FastAPI(title="Warden Battery Recommendations", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Battery routes are registered on this same app instance (see bottom).


@app.get("/api/health")
def health(request: Request) -> dict:
    grid: AppState = request.app.state.grid
    return {
        "status": "ok",
        "net_loaded": grid.net is not None,
        "dataset": grid.settings.grid_dataset,
        "scenario": grid.scenario_id,
        "baseline_cached": len(grid.baseline) > 0,
    }


@app.get("/api/state")
def state(request: Request) -> dict:
    grid: AppState = request.app.state.grid
    return grid.state_summary()


class ScenarioRequest(BaseModel):
    scenario_id: str


@app.post("/api/scenario")
def set_scenario(request: Request, body: ScenarioRequest):
    grid: AppState = request.app.state.grid
    if body.scenario_id not in SCENARIOS:
        return error_response(
            400,
            "invalid_scenario",
            f"unknown scenario {body.scenario_id!r}; known: {sorted(SCENARIOS)}",
        )
    try:
        grid.set_scenario(body.scenario_id)
    except ValueError as exc:
        return error_response(400, "scenario_incompatible", str(exc))
    return grid.state_summary()


# Register battery routes (kept in their own module per the requested
# layout: src/api/routes/battery.py).
from src.api.routes.battery import router as battery_router  # noqa: E402

app.include_router(battery_router)

# Real-time time-stepped simulation routes (Phase 2, U10).
from src.api.routes.timeseries import router as timeseries_router  # noqa: E402

app.include_router(timeseries_router)

# Agent planning route (Phase 2, U4).
from src.api.routes.agent import router as agent_router  # noqa: E402

app.include_router(agent_router)

# Baseline comparison route (challenge direction D3).
from src.api.routes.compare import router as compare_router  # noqa: E402

app.include_router(compare_router)


@app.get("/api/topology")
def topology(request: Request) -> dict:
    """The frozen one-line layout (buses + edges) the frontend renders.

    Solver-agnostic geometry; per-step loadings map on by edge id. Regenerate via
    scripts/freeze_topology.py.
    """
    grid = getattr(request.app.state, "grid", None)
    if grid is not None and grid.settings.grid_dataset != "case118":
        return _topology_from_net(grid.net)

    path = _FIXTURES_DIR / "grid_topology.json"
    if not path.is_file():
        return error_response(404, "no_topology", "grid_topology.json not found; run scripts/freeze_topology.py")
    return json.loads(path.read_text())


# Static SPA: the console at "/", assets under /static. Mounted last so the API routes win.
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))

    @app.get("/console")
    def console() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "console.html"))
