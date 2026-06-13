"""The 4 agent tools as plain Python functions over the src engine (ported, U3).

All numeric values originate from engine (pandapower) results; the LLM never invents physics.
ToolContext holds the live net so each tool operates on the right network without a global. TOOLS
is the Anthropic tool-schema list. Tool functions return JSON-serializable dicts via to_jsonable.

src has no screener module: scan_contingencies runs the full AC cascade on every contingency
(run_contingency_sweep, i.e. screener="none" semantics). The screener arg is accepted for API
parity but does not change behavior.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandapower as pp

from src.engine.actions import Action, VerificationReport, to_jsonable
from src.engine.constants import OVERLOAD_LIMIT
from src.engine.network import base_case_summary, native_index, total_load_mw, working_copy
from src.engine.remediation import apply_action_to_net, run_opf, verify_action
from src.engine.scan import ContingencyResult, build_contingency_set, rank, run_contingency_sweep

_INSECURE_BANDS = ("CRITICAL", "HIGH")
_LLM_SCAN_TOP = 20
# Verify a proposed action against the most-severe N-1 contingencies, not all of them. A base-case
# OPF redispatch cannot keep EVERY N-1 non-worsened (that is the security-constrained OPF problem),
# so requiring all of them is both over-strict (no action ever verifies on a stressed grid) and slow
# (every contingency is a full AC cascade). Verifying the worst few is the honest, fast operator scope;
# the UI/narration says "base secured + top-N N-1 held", not "fully N-1 secure".
_VERIFY_TOP_OFFENDING = 3


@dataclass
class ToolContext:
    """Carries the live net through the tool-calling loop.

    net:            the live pandapower net (mutated only by apply_action commit=True).
    last_scan:      ranked list[ContingencyResult] from the last scan_contingencies call.
    last_offending: remediable insecure results (CRITICAL/HIGH line/trafo outages).
    last_action:    proposed Action (dict) from the most recent run_opf, held server-side so
                    apply_action need not have the LLM transcribe setpoints.
    """
    net: Any
    last_scan: list = field(default_factory=list)
    last_offending: list = field(default_factory=list)
    last_action: dict | None = None


def run_power_flow(ctx: ToolContext, mode: str = "ac") -> dict:
    """Solve the base-case power flow on the live net (read-only). ac = runpp, dc = rundcpp."""
    t0 = time.perf_counter()
    net = ctx.net
    try:
        if mode == "dc":
            work = working_copy(net)
            pp.rundcpp(work)
            ll = work.res_line.loading_percent.dropna()
            result = {
                "converged": True,
                "max_line_loading_pct": round(float(ll.max()), 1) if len(ll) else 0.0,
                "n_overloads": int((ll > OVERLOAD_LIMIT).sum()),
                "min_vm_pu": None, "max_vm_pu": None, "n_voltage_violations": 0,
                "total_load_mw": round(total_load_mw(net), 1),
            }
        else:
            result = base_case_summary(net)
    except Exception as exc:
        result = {
            "converged": False, "max_line_loading_pct": None, "n_overloads": 0,
            "min_vm_pu": None, "max_vm_pu": None, "n_voltage_violations": 0,
            "total_load_mw": round(total_load_mw(net), 1), "error": str(exc),
        }
    result["_timing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return to_jsonable(result)


def scan_contingencies(ctx: ToolContext, screener: str = "none", **_ignored) -> dict:
    """Run the N-1 security scan: full AC cascade on every contingency, ranked by CSS.
    Updates ctx.last_scan and ctx.last_offending. Returns the top slice for the LLM to read."""
    t0 = time.perf_counter()
    net = ctx.net
    results = rank(run_contingency_sweep(net, build_contingency_set(net)))
    ctx.last_scan = results
    ctx.last_offending = [
        r for r in results
        if r.severity.band in _INSECURE_BANDS and r.outage.get("etype") in ("line", "trafo")
    ][:_VERIFY_TOP_OFFENDING]
    n_insecure = sum(1 for r in results if r.severity.band in _INSECURE_BANDS)
    summaries = [{
        "contingency_id": r.contingency_id,
        "outage_name": r.outage.get("name", r.contingency_id),
        "status": r.status,
        "severity": to_jsonable(r.severity),
        "first_overloads": to_jsonable(r.first_overloads),
    } for r in results[:_LLM_SCAN_TOP]]
    return to_jsonable({
        "results": summaries,
        "n_scanned": len(results),
        "n_insecure": n_insecure,
        "note": f"showing top {len(summaries)} of {len(results)} contingencies by severity",
        "timing": {"total_ms": round((time.perf_counter() - t0) * 1000, 1)},
        "_timing_ms": round((time.perf_counter() - t0) * 1000, 1),
    })


def run_opf_tool(ctx: ToolContext, tighten: list | None = None) -> dict:
    """Run AC-OPF on a copy and return a proposed redispatch Action. Never mutates the live net."""
    t0 = time.perf_counter()
    action, _opf, _base, converged, predicted_base = run_opf(ctx.net, tighten=tighten or None, source="opf")
    timing_ms = round((time.perf_counter() - t0) * 1000, 1)
    if not converged or action is None:
        ctx.last_action = None
        return to_jsonable({"converged": False, "proposed_action": None, "predicted_base": None, "_timing_ms": timing_ms})
    ctx.last_action = to_jsonable(action)
    return to_jsonable({"converged": True, "proposed_action": ctx.last_action, "predicted_base": predicted_base, "_timing_ms": timing_ms})


def apply_action_tool(ctx: ToolContext, action: dict | None = None, commit: bool = False,
                      verify_contingency_ids: list | None = None) -> dict:
    """Apply an Action. Omit action to use the server-held run_opf proposal. commit=False is a dry
    run with full AC cascade re-verification; commit=True mutates the live net only after verify."""
    t0 = time.perf_counter()
    net = ctx.net
    if not action or not action.get("changes"):
        action = ctx.last_action or (action or {})
    act = Action(
        action_id=action.get("action_id", "act_unknown"),
        type=action.get("type", "redispatch"),
        changes=action.get("changes", []),
        source=action.get("source", "opf"),
        estimated_cost_delta=float(action.get("estimated_cost_delta", 0.0)),
        rationale=action.get("rationale", ""),
    )
    if not verify_contingency_ids:
        offending = ctx.last_offending
    else:
        by_id = {r.contingency_id: r for r in ctx.last_scan}
        offending = [by_id[cid] for cid in verify_contingency_ids if cid in by_id]

    report: VerificationReport = verify_action(net, act, offending)
    if commit and report.verified:
        apply_action_to_net(net, act)
        report.committed = True

    result = to_jsonable(report)
    result["_timing_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


def dispatch_tool(ctx: ToolContext, tool_name: str, tool_input: dict) -> dict:
    if tool_name == "run_power_flow":
        return run_power_flow(ctx, mode=tool_input.get("mode", "ac"))
    if tool_name == "scan_contingencies":
        return scan_contingencies(ctx, screener=tool_input.get("screener", "none"))
    if tool_name == "run_opf":
        return run_opf_tool(ctx, tighten=tool_input.get("tighten"))
    if tool_name == "apply_action":
        return apply_action_tool(
            ctx,
            action=tool_input.get("action"),
            commit=bool(tool_input.get("commit", False)),
            verify_contingency_ids=tool_input.get("verify_contingency_ids") or [],
        )
    raise ValueError(f"unknown tool: {tool_name}")


TOOLS: list[dict] = [
    {
        "name": "run_power_flow",
        "description": "Solve the base-case power flow on the live net (read-only). ac = runpp, dc = rundcpp.",
        "input_schema": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["ac", "dc"]}}, "required": ["mode"]},
    },
    {
        "name": "scan_contingencies",
        "description": "Run the N-1 security scan: full AC cascade analysis on every contingency, ranked by severity. Read-only.",
        "input_schema": {"type": "object", "properties": {"screener": {"type": "string", "enum": ["dc", "none"]}}, "required": []},
    },
    {
        "name": "run_opf",
        "description": ("Run AC-OPF on a COPY of the net with tightened line limits and return a proposed redispatch Action. "
                        "Never mutates the live net. Omit tighten to let the backend auto-tighten lines to the security margin."),
        "input_schema": {"type": "object", "properties": {"tighten": {"type": "array", "items": {"type": "object",
            "properties": {"etype": {"type": "string"}, "index": {"type": "integer"}, "max_loading_pct": {"type": "number"}},
            "required": ["etype", "index", "max_loading_pct"]}}}, "required": []},
    },
    {
        "name": "apply_action",
        "description": ("Apply the remediation proposed by the most recent run_opf. Do NOT pass an action: the backend holds "
                        "the proposal, so you never copy setpoints. commit=false: dry run on a copy with full AC cascade "
                        "re-verification. commit=true: mutate the live net (only after a verified dry run)."),
        "input_schema": {"type": "object", "properties": {"commit": {"type": "boolean", "default": False},
            "verify_contingency_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["commit"]},
    },
]
