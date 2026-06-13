"""Battery recommendation API routes.

POST /battery/recommendations - scored, solver-verified top-K candidates
POST /battery/verify           - verify one explicit bus

Both run against the cached baseline sweep in AppState, so the expensive
N-1 physics has already run by the time these are called (the recommend
path stays under the 10 s budget for case118).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.state import AppState
from src.battery.narration import build_findings, narrate
from src.battery.recommender import recommend_battery_locations
from src.battery.schemas import (
    BatteryCandidate,
    RecommendationRequest,
    RecommendationResponse,
    VerificationResult,
    VerifyRequest,
)
from src.battery.scoring import score_buses
from src.battery.verification import (
    SlackBusError,
    UnknownBusError,
    impact_contingency_ids,
    verify_battery_candidate,
)
from src.engine.network import native_index

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/battery", tags=["battery"])


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"code": code, "message": message}}
    )


@router.post("/recommendations", response_model=RecommendationResponse)
def recommendations(request: Request, body: RecommendationRequest):
    grid: AppState = request.app.state.grid
    if body.top_k > grid.settings.battery_max_topk:
        # Cap rather than reject: the response reports the K actually used.
        logger.info(
            "top_k %d capped to BATTERY_MAX_TOPK %d",
            body.top_k, grid.settings.battery_max_topk,
        )
    resp = recommend_battery_locations(
        grid.net,
        body,
        settings=grid.settings,
        baseline=grid.baseline,
        contingencies=grid.contingencies,
    )
    return resp


@router.post("/verify", response_model=BatteryCandidate)
def verify(request: Request, body: VerifyRequest):
    """Verify one explicit bus. Returns the full candidate record
    (score context + verification + narration). A slack bus is rejected
    with 400 before any solver call (acceptance rule)."""
    grid: AppState = request.app.state.grid
    try:
        restrict_to = impact_contingency_ids(grid.baseline)
        verification: VerificationResult = verify_battery_candidate(
            grid.net,
            body.bus_idx,
            contingencies=grid.contingencies,
            baseline=grid.baseline,
            battery_p_mw=body.battery_capacity_mw,
            battery_max_e_mwh=body.battery_energy_mwh,
            restrict_to=restrict_to,
        )
    except SlackBusError as exc:
        return _error(400, "slack_bus", str(exc))
    except UnknownBusError as exc:
        return _error(400, "unknown_bus", str(exc))

    # Attach the bus's score context and a narration so the verify
    # endpoint returns the same shape the recommendation list uses.
    scores = score_buses(grid.net, grid.baseline)
    resolved = verification.bus_idx
    bus_score = next((bs for bs in scores if bs.bus_idx == resolved), None)
    rank_pos = next(
        (i for i, bs in enumerate(scores, start=1) if bs.bus_idx == resolved), 0
    )
    if bus_score is None:
        # Bus exists and is non-slack but was not in the scored set
        # (e.g. out of service). Return verification without score ctx.
        return BatteryCandidate(
            bus_idx=resolved,
            score=0.0,
            score_breakdown={"congestion": 0, "voltage": 0, "cascade": 0, "severity": 0},
            context={
                "total_scenarios": len(grid.baseline),
                "congestion_count": 0,
                "voltage_count": 0,
                "cascade_count": 0,
            },
            verification=verification,
            narration=None,
            narration_source="none",
        )
    findings = build_findings(rank_pos, len(scores), bus_score, verification)
    narration, source = narrate(findings, grid.settings)
    return BatteryCandidate(
        bus_idx=resolved,
        score=bus_score.score,
        score_breakdown=bus_score.score_breakdown,
        context=bus_score.context,
        verification=verification,
        narration=narration,
        narration_source=source,
    )
