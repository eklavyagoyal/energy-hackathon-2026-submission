"""The Warden planning loop (ported onto the src engine, U3).

plan(scenario_id_or_net, replay=False) -> {plan_id, action, narration:{sentences:[3]}, verification, tool_trace}

Default path: deterministic orchestration (run_power_flow -> scan_contingencies -> OPF
tighten-and-verify) with the LLM used ONLY for narration. The fixed tool sequence makes letting the
model drive it pure variance, so the LLM-driven tool loop is opt-in via WARDEN_AGENT_MODE=llm. A
missing ANTHROPIC_API_KEY (or replay=True) falls back to the deterministic template narrator; the
LLM never computes a physical quantity in any path.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from src.agent import llm_model
from src.agent.prompts import render_narration_prompt, template_narration
from src.agent.state import build_grid_state_summary
from src.agent.tools import TOOLS, ToolContext, dispatch_tool
from src.engine.actions import Action, VerificationReport, to_jsonable
from src.engine.remediation import base_overloaded_lines, propose_remediation

_MAX_PLAN_ITERS = 3

_PLANNER_SYSTEM = (
    "You are Warden, a grid operation agent. Use the tools to diagnose and fix N-1 insecurity. "
    "Never state or invent a physical quantity yourself; all numbers come from tool results. "
    "Call run_power_flow first, then scan_contingencies, then if insecure call run_opf and "
    "apply_action(commit=false) up to 3 times. When you call apply_action, pass ONLY commit and "
    "verify_contingency_ids; do NOT include an action argument (the backend applies the latest "
    "run_opf proposal). Stop as soon as apply_action returns verified=true or you reach 3 iterations."
)


def _result_summary(tool_name: str, result: dict) -> str:
    try:
        if tool_name == "run_power_flow":
            return f"converged={result.get('converged')}, max loading {result.get('max_line_loading_pct')} pct, {result.get('n_overloads', 0)} overloads"
        if tool_name == "scan_contingencies":
            rs = result.get("results", [])
            insecure = sum(1 for r in rs if r.get("severity", {}).get("band") in ("CRITICAL", "HIGH"))
            worst = rs[0] if rs else None
            worst_str = ""
            if worst:
                sev = worst.get("severity", {})
                worst_str = f", worst {worst['contingency_id']} {sev.get('band', '')} {sev.get('score', '')}"
            return f"{result.get('n_scanned', len(rs))} scanned, {insecure} insecure{worst_str}"
        if tool_name == "run_opf":
            action = result.get("proposed_action")
            if not result.get("converged") or not action:
                return "OPF did not converge"
            return f"converged, {action.get('action_id')} {action.get('type', 'noop')} {len(action.get('changes', []))} changes, cost delta {action.get('estimated_cost_delta', 0.0)}"
        if tool_name == "apply_action":
            d = result.get("deltas", {})
            return f"verified={result.get('verified')}, {len(result.get('contingency_ids', []))} contingencies checked, {d.get('violations_resolved', 0)} violations resolved, {d.get('load_shed_avoided_mw', 0.0)} MW shed avoided"
    except Exception:
        pass
    return json.dumps(result)[:120]


def _noop_action() -> dict:
    return to_jsonable(Action(
        action_id="act_noop", type="noop", changes=[], source="opf",
        estimated_cost_delta=0.0,
        rationale="base case secure; no remediable insecure contingencies found",
    ))


def _noop_verification() -> dict:
    return to_jsonable(VerificationReport(
        verified=True, method="ac_cascade_rescan", contingency_ids=[], before=[], after=[],
        deltas={"violations_resolved": 0, "load_shed_avoided_mw": 0.0,
                "worst_score_before": 0.0, "worst_score_after": 0.0},
        committed=False,
    ))


def _template_plan(ctx: ToolContext, scenario_id: str, plan_id: str, narrator=None) -> dict:
    """Deterministic orchestration; narrator renders the 3 sentences (template or LLM)."""
    narrate = narrator or template_narration
    tool_trace: list[dict] = []

    t0 = time.perf_counter()
    pf_result = dispatch_tool(ctx, "run_power_flow", {"mode": "ac"})
    tool_trace.append({"tool": "run_power_flow", "args": {"mode": "ac"},
                       "result_summary": _result_summary("run_power_flow", pf_result),
                       "ms": round((time.perf_counter() - t0) * 1000, 1)})

    t0 = time.perf_counter()
    scan_result = dispatch_tool(ctx, "scan_contingencies", {"screener": "none"})
    tool_trace.append({"tool": "scan_contingencies", "args": {"screener": "none"},
                       "result_summary": _result_summary("scan_contingencies", scan_result),
                       "ms": round((time.perf_counter() - t0) * 1000, 1)})

    state = build_grid_state_summary(ctx.net, {"results": ctx.last_scan}, scenario_id)

    offending = ctx.last_offending
    if not offending:
        action = _noop_action()
        verification = _noop_verification()
        return {"plan_id": plan_id, "action": action, "narration": {"sentences": narrate(state, action, verification)},
                "verification": verification, "tool_trace": tool_trace}

    best_action: dict | None = None
    best_verification: dict | None = None
    offending_ids = [r.contingency_id for r in offending]

    for _iteration in range(_MAX_PLAN_ITERS):
        t0 = time.perf_counter()
        opf_result = dispatch_tool(ctx, "run_opf", {})
        tool_trace.append({"tool": "run_opf", "args": {},
                           "result_summary": _result_summary("run_opf", opf_result),
                           "ms": round((time.perf_counter() - t0) * 1000, 1)})
        proposed = opf_result.get("proposed_action")
        if not opf_result.get("converged") or not proposed:
            break
        best_action = proposed
        t0 = time.perf_counter()
        verify_result = dispatch_tool(ctx, "apply_action",
                                      {"commit": False, "verify_contingency_ids": offending_ids})
        tool_trace.append({"tool": "apply_action", "args": {"commit": False, "verify_contingency_ids": offending_ids},
                           "result_summary": _result_summary("apply_action", verify_result),
                           "ms": round((time.perf_counter() - t0) * 1000, 1)})
        best_verification = verify_result
        if verify_result.get("verified"):
            break

    if best_action is None:
        best_action = _noop_action()
    if best_verification is None:
        best_verification = _noop_verification()

    return {"plan_id": plan_id, "action": best_action,
            "narration": {"sentences": narrate(state, best_action, best_verification)},
            "verification": best_verification, "tool_trace": tool_trace}


def _llm_plan(ctx: ToolContext, scenario_id: str, plan_id: str) -> dict:
    """Opt-in: let the LLM drive the tool loop. Falls back to _template_plan on any error, with a
    deterministic safety net that computes remediation server-side if the model under-acts."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        return _template_plan(ctx, scenario_id, plan_id)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return _template_plan(ctx, scenario_id, plan_id)
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception:
        return _template_plan(ctx, scenario_id, plan_id)

    tool_trace: list[dict] = []
    messages: list[dict] = [{"role": "user", "content": "Diagnose and fix N-1 insecurity for the current grid state."}]
    opf_apply_count = 0
    final_opf_result: dict | None = None
    final_verify_result: dict | None = None
    try:
        while True:
            resp = client.messages.create(model=llm_model(), max_tokens=4096, temperature=0,
                                          system=_PLANNER_SYSTEM, tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break
            tool_results_content: list[dict] = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                t0 = time.perf_counter()
                try:
                    result = dispatch_tool(ctx, block.name, block.input)
                except Exception as exc:
                    result = {"error": str(exc)}
                tool_trace.append({"tool": block.name, "args": block.input,
                                   "result_summary": _result_summary(block.name, result),
                                   "ms": round((time.perf_counter() - t0) * 1000, 1)})
                if block.name == "run_opf":
                    final_opf_result = result
                if block.name == "apply_action":
                    final_verify_result = result
                    opf_apply_count += 1
                tool_results_content.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
                if opf_apply_count >= _MAX_PLAN_ITERS:
                    break
            messages.append({"role": "user", "content": tool_results_content})
            if opf_apply_count >= _MAX_PLAN_ITERS:
                break
    except Exception as exc:
        tool_trace.append({"tool": "llm_error", "args": {}, "result_summary": str(exc), "ms": 0})
        return _template_plan(ctx, scenario_id, plan_id)

    state = build_grid_state_summary(ctx.net, {"results": ctx.last_scan}, scenario_id)
    best_action: dict = _noop_action()
    best_verification: dict = _noop_verification()
    if ctx.last_action and ctx.last_action.get("changes"):
        best_action = ctx.last_action
    elif final_opf_result and final_opf_result.get("proposed_action"):
        best_action = final_opf_result["proposed_action"]
    if final_verify_result:
        best_verification = final_verify_result

    needs_fix = bool(ctx.last_offending) or bool(base_overloaded_lines(ctx.net))
    if needs_fix and (best_action.get("type") == "noop" or not best_action.get("changes")):
        act, rep = propose_remediation(ctx.net, ctx.last_offending)
        best_action = to_jsonable(act)
        best_verification = to_jsonable(rep)
        tool_trace.append({"tool": "deterministic_remediation", "args": {"reason": "LLM loop produced no verified action"},
                           "result_summary": f"verified={rep.verified}, {len(act.changes)} changes", "ms": 0})

    sentences = _narrate_with_llm(client, state, best_action, best_verification)
    return {"plan_id": plan_id, "action": best_action, "narration": {"sentences": sentences},
            "verification": best_verification, "tool_trace": tool_trace}


def _narrate_with_llm(client: Any, state: dict, action: dict, report: dict) -> list[str]:
    """One no-tool LLM call for narration; retries once on wrong sentence count; template fallback."""
    prompt = render_narration_prompt(state, action, report)
    for _attempt in range(2):
        try:
            resp = client.messages.create(model=llm_model(), max_tokens=512, temperature=0,
                                          messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            parts = [s.strip() for s in text.split(". ") if s.strip()]
            sentences = [p + "." if (i < len(parts) - 1 or not p.endswith(".")) else p for i, p in enumerate(parts)]
            if len(sentences) == 3:
                return sentences
        except Exception:
            pass
    return template_narration(state, action, report)


def plan(scenario_id_or_net: Any, replay: bool = False) -> dict:
    """Run the planning loop. scenario_id_or_net is a scenario id ("calm"/"demo_congestion") or a
    pandapower net. replay (or a missing API key) forces the deterministic template narrator."""
    plan_id = f"plan_{uuid.uuid4().hex[:8]}"

    if isinstance(scenario_id_or_net, str):
        from src.engine.scenarios import SCENARIOS, apply_scenario
        from src.grid.loader import Case118Loader
        scenario_id = scenario_id_or_net
        base = Case118Loader().load()
        net = apply_scenario(base, scenario_id) if scenario_id in SCENARIOS else base
    else:
        net = scenario_id_or_net
        scenario_id = getattr(net, "_scenario_id", "custom")

    ctx = ToolContext(net=net)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if replay or not api_key:
        return _template_plan(ctx, scenario_id, plan_id)
    if os.environ.get("WARDEN_AGENT_MODE", "").strip().lower() == "llm":
        return _llm_plan(ctx, scenario_id, plan_id)
    try:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key)

        def _llm_narrator(state: dict, action: dict, report: dict) -> list[str]:
            return _narrate_with_llm(client, state, action, report)

        return _template_plan(ctx, scenario_id, plan_id, narrator=_llm_narrator)
    except Exception:
        return _template_plan(ctx, scenario_id, plan_id)
