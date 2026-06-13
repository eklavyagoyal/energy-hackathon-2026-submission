"""Baseline comparison route (challenge direction D3).

POST /compare  run three policies on the same scenario and return a head-to-head table:
  - greedy (RULE-BASED strawman): trip the most-overloaded line, no lookahead, no verification.
  - AC-OPF (OPTIMIZATION baseline): cost-optimal redispatch, a black box with no N-1 check, no words.
  - Warden agent (LLM): drives the OPF tool, VERIFIES the result on the full N-1 cascade, and EXPLAINS
    it in solver-grounded sentences.

The point of the table (the E.ON ask: "compare an LLM-driven approach against a rule-based OR
optimisation baseline") is NOT that the agent out-optimizes OPF, it uses OPF. It is that the agent
adds the two things an operator needs and a raw solver does not: a verified-secure guarantee and a
human-readable explanation. Greedy shows why naive intervention is dangerous.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import get_settings
from src.engine.network import base_case_summary, total_load_mw
from src.engine.remediation import (
    SECURITY_MARGIN_PCT,
    apply_action_to_net,
    greedy_policy,
    run_opf,
    verify_action,
)
from src.engine.scan import rank, run_contingency_sweep
from src.engine.scenarios import SCENARIOS, apply_scenario
from src.grid.loader import get_loader

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/compare", tags=["compare"])

_INSECURE_BANDS = ("CRITICAL", "HIGH")


class CompareRequest(BaseModel):
    scenario: str = "demo_overload"


def _base_secure(net) -> bool:
    m = base_case_summary(net)
    return bool(m["converged"]) and (m["n_overloads"] or 0) == 0 and (m["n_voltage_violations"] or 0) == 0


@router.post("")
def compare(body: CompareRequest):
    if body.scenario not in SCENARIOS:
        return JSONResponse(status_code=400, content={"error": {"code": "invalid_scenario",
                            "message": f"unknown scenario {body.scenario!r}; known: {sorted(SCENARIOS)}"}})

    try:
        net = apply_scenario(get_loader(get_settings()).load(), body.scenario)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"code": "scenario_incompatible",
                            "message": str(exc)}})
    base = base_case_summary(net)
    sweep = rank(run_contingency_sweep(net))
    offending = [r for r in sweep if r.severity.band in _INSECURE_BANDS
                 and r.outage.get("etype") in ("line", "trafo")][:3]
    rows: list[dict] = []

    # 1. rule-based greedy strawman
    t0 = time.perf_counter()
    br = greedy_policy(net)
    rows.append({
        "policy": "Greedy (rule-based)", "kind": "rule-based",
        "secure_after": bool(br.secure_after), "n1_verified": None,
        "load_shed_mw": round(br.load_shed_mw, 1), "action_cost_eur": 0.0,
        "lines_tripped": br.worst_cascade_depth_after, "wall_ms": round((time.perf_counter() - t0) * 1000, 0),
        "explained": False, "note": "trips the most-overloaded line repeatedly, no lookahead, no verification",
    })

    # 2. AC-OPF optimization baseline (no N-1 verification framing, no explanation)
    t0 = time.perf_counter()
    action, _opf, _bc, converged, _pred = run_opf(net, global_max_loading=SECURITY_MARGIN_PCT)
    if converged and action is not None:
        from src.engine.network import working_copy
        w = working_copy(net)
        apply_action_to_net(w, action)
        rep = verify_action(net, action, offending)
        rows.append({
            "policy": "AC-OPF (optimization)", "kind": "optimization",
            "secure_after": _base_secure(w), "n1_verified": bool(rep.verified),
            "load_shed_mw": 0.0, "action_cost_eur": action.estimated_cost_delta,
            "lines_tripped": 0, "wall_ms": round((time.perf_counter() - t0) * 1000, 0),
            "explained": False, "note": "cost-optimal generator redispatch, a black-box solve with no operator narrative",
        })
    else:
        rows.append({
            "policy": "AC-OPF (optimization)", "kind": "optimization", "secure_after": False,
            "n1_verified": False, "load_shed_mw": 0.0, "action_cost_eur": 0.0, "lines_tripped": 0,
            "wall_ms": round((time.perf_counter() - t0) * 1000, 0), "explained": False,
            "note": "OPF infeasible on this scenario",
        })

    # 3. Warden agent (LLM): the real planning loop, redispatch verified + explained
    t0 = time.perf_counter()
    from src.agent.loop import plan
    plan_res = plan(net, replay=True)
    v = plan_res.get("verification", {})
    a = plan_res.get("action", {})
    deltas = v.get("deltas", {})
    rows.append({
        "policy": "Warden agent (LLM)", "kind": "llm-agent",
        "secure_after": bool(v.get("verified")), "n1_verified": bool(v.get("verified")),
        "load_shed_mw": round(deltas.get("load_shed_avoided_mw", 0.0), 1),  # avoided, not incurred
        "load_shed_is_avoided": True,
        "action_cost_eur": a.get("estimated_cost_delta", 0.0),
        "lines_tripped": 0, "wall_ms": round((time.perf_counter() - t0) * 1000, 0),
        "explained": True, "note": "drives the OPF tool, verifies the result on the full N-1 cascade, and explains it",
        "narration": plan_res.get("narration", {}).get("sentences", []),
        "action_type": a.get("type"),
    })

    return {
        "scenario": body.scenario,
        "base_overloads": base.get("n_overloads", 0),
        "base_max_loading_pct": round(base.get("max_line_loading_pct", 0.0) or 0.0, 1),
        "total_load_mw": round(total_load_mw(net), 1),
        "rows": rows,
    }
