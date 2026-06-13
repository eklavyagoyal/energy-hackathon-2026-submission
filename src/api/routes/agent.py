"""Agent planning route (U4).

POST /agent/plan  run the Warden planning loop on a scenario and return the action, the 3-sentence
                  narration, the verification report, and the tool trace.

plan() builds its own net from the scenario id, so the live AppState net is never mutated. replay
(the default) forces the deterministic template narrator: no LLM call, reproducible, free. Set
replay=false (with ANTHROPIC_API_KEY and optionally WARDEN_AGENT_MODE=llm) for real LLM narration.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.engine.scenarios import SCENARIOS, apply_scenario, scenario_is_compatible

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


class PlanRequest(BaseModel):
    scenario: str = "demo_congestion"
    replay: bool = True


@router.post("/plan")
def plan_endpoint(request: Request, body: PlanRequest):
    if body.scenario not in SCENARIOS:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "invalid_scenario",
                               "message": f"unknown scenario {body.scenario!r}; known: {sorted(SCENARIOS)}"}},
        )
    from src.agent.loop import plan  # lazy: only pull anthropic/agent deps when planning is requested

    try:
        grid = getattr(request.app.state, "grid", None)
        if grid is not None and grid.settings.grid_dataset != "case118":
            if not scenario_is_compatible(grid.base_net, body.scenario):
                return JSONResponse(
                    status_code=400,
                    content={"error": {"code": "scenario_incompatible",
                                       "message": f"scenario {body.scenario!r} is not compatible with dataset {grid.settings.grid_dataset!r}"}},
                )
            return plan(apply_scenario(grid.base_net, body.scenario), replay=body.replay)
        return plan(body.scenario, replay=body.replay)
    except Exception as exc:  # surface a clean envelope, never a 500 stacktrace to the UI
        logger.exception("agent plan failed")
        return JSONResponse(status_code=500, content={"error": {"code": "plan_failed", "message": str(exc)}})
