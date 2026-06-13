"""Time-stepped simulation API routes (U10).

POST /timeseries/run       run the loop, return the full SimulationTrace (per-step state, actions,
                           verify-before-commit results, narration, edge loadings for the timeline)
GET  /timeseries/profiles  the available profiles, event scenarios, and agent modes

Self-contained: each run loads a fresh net from the configured loader and (for the stress demo)
seeds the congestion pocket, so the simulation never mutates the live battery/state net. The agent
decides and EVERY action is solver-verified before it would be applied, at every timestep.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.engine.scenarios import SCENARIOS, apply_scenario, scenario_is_compatible
from src.grid.loader import get_loader
from src.timeseries.profiles import GenerationProfile, Profile
from src.timeseries.schemas import (
    AGENT_MODES,
    EVENT_SCENARIOS,
    GEOGRAPHIC_SCENARIO_NAMES,
    PROFILES,
    TimeseriesProfilesResponse,
    TimeseriesRunRequest,
)
from src.timeseries.events import build_event_scenario
from src.timeseries.scenarios import GEOGRAPHIC_SCENARIOS, build_geographic_scenario_run, scenario_options
from src.timeseries.simulator import run_simulation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/timeseries", tags=["timeseries"])

_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _load_profile(body: TimeseriesRunRequest) -> Profile:
    """Pick the load profile from profile_id. 'calm' is flat; 'entsoe_<cc>_<date>' reads
    data/entsoe/ (synthetic fallback); everything else is the synthetic daily curve."""
    pid, h = body.profile_id, body.horizon_steps
    if pid == "calm":
        return Profile.flat(hours=h, value=1.0, name="calm", kind="load")
    if pid.startswith("entsoe_"):
        parts = pid.split("_")
        if len(parts) >= 3:
            return Profile.from_entsoe(parts[1], "_".join(parts[2:]), hours=h, seed=body.seed)
    return Profile.synthetic(hours=h, seed=body.seed, name=pid, kind="load")


@router.get("/replay")
def replay():
    """Serve the frozen 24-step demo trace (fixtures/realtime_trace.json) for an instant timeline,
    no live simulation wait. Numbers are solver-produced; regenerate via scripts/freeze_realtime_demo.py."""
    path = _FIXTURES / "realtime_trace.json"
    if not path.is_file():
        return _error(404, "no_fixture", "realtime_trace.json not found; run scripts/freeze_realtime_demo.py")
    return json.loads(path.read_text())


@router.get("/profiles", response_model=TimeseriesProfilesResponse)
def profiles() -> TimeseriesProfilesResponse:
    return TimeseriesProfilesResponse(
        profiles=list(PROFILES),
        event_scenarios=list(EVENT_SCENARIOS),
        agent_modes=list(AGENT_MODES),
    )


@router.get("/scenarios")
def scenarios() -> list[dict]:
    return scenario_options()


@router.post("/run")
def run(body: TimeseriesRunRequest):
    if body.agent_mode not in AGENT_MODES:
        return _error(400, "invalid_agent_mode", f"unknown agent_mode {body.agent_mode!r}; use {list(AGENT_MODES)}")
    if body.scenario is not None and body.scenario not in GEOGRAPHIC_SCENARIO_NAMES:
        return _error(
            400,
            "invalid_scenario",
            f"unknown scenario {body.scenario!r}; use {list(GEOGRAPHIC_SCENARIO_NAMES)}",
        )

    settings = get_settings()
    base = get_loader(settings).load()

    if body.scenario and body.scenario != "custom":
        if body.scenario not in GEOGRAPHIC_SCENARIOS:
            return _error(
                400,
                "invalid_scenario",
                f"unknown scenario {body.scenario!r}; known: {sorted(GEOGRAPHIC_SCENARIOS)}",
            )
        prepared = build_geographic_scenario_run(body.scenario, base, seed=body.seed)
        trace = run_simulation(
            prepared.net,
            prepared.load_profile,
            prepared.gen_profile,
            prepared.event_stream,
            agent_mode=body.agent_mode,
            horizon_steps=prepared.scenario.duration_hours,
            seed=body.seed,
            profile_id=body.scenario,
            event_scenario=body.scenario,
            step_minutes=body.step_minutes,
            replay=not body.narrate,
            scenario_context=prepared.metadata,
        )
        return trace.to_dict()

    # Seed the net: the stress demo starts from the congestion pocket so contingencies actually
    # cascade and the agent has real work; calm runs start from the secure base.
    start = body.start_scenario
    if start is None:
        start = "demo_congestion" if body.event_scenario == "stress_demo" else None
    if start and start not in SCENARIOS:
        return _error(400, "invalid_start_scenario", f"unknown start_scenario {start!r}; known: {sorted(SCENARIOS)}")
    if start and start in SCENARIOS and not scenario_is_compatible(base, start):
        if body.start_scenario is not None:
            return _error(400, "scenario_incompatible",
                          f"start_scenario {start!r} is not compatible with dataset {settings.grid_dataset!r}")
        start = None
    net = apply_scenario(base, start) if start else base

    load_profile = _load_profile(body)
    gen_profile = GenerationProfile(hours=body.horizon_steps)
    events = build_event_scenario(body.event_scenario, net)

    trace = run_simulation(
        net,
        load_profile,
        gen_profile,
        events,
        agent_mode=body.agent_mode,
        horizon_steps=body.horizon_steps,
        seed=body.seed,
        profile_id=body.profile_id,
        event_scenario=body.event_scenario,
        step_minutes=body.step_minutes,
        replay=not body.narrate,
    )
    return trace.to_dict()
