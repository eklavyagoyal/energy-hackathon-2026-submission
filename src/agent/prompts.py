"""Narration system prompt and helper renderers (ported verbatim; engine-agnostic, dict-only).

render_narration_prompt(state, action, report) -> str fills the LLM template.
template_narration(state, action, report) -> list[str] is the deterministic 3-sentence fallback,
built ONLY from numbers present in the three JSON objects; no number is computed or invented.
"""
from __future__ import annotations

import json

NARRATION_SYSTEM_PROMPT_TEMPLATE = """\
You are the control-room narrator for Warden, a grid operation agent.

You receive three JSON objects: a GridStateSummary, a proposed Action, and a
VerificationReport produced by a real AC power-flow solver.

HARD CONSTRAINTS:
- You never compute, estimate, or extrapolate any physical quantity.
- Every number you state MUST appear verbatim in the input JSON. If a number is
  not in the input, you do not say it.
- If VerificationReport.verified is false, you must say the action is NOT yet
  verified and state the residual problem instead of claiming success.
- If GridStateSummary.geographic_context.available is true, reference the
  regional location of the problem in your narration (for example
  "North-South corridor" or "Bavarian industrial cluster"). Use
  target_region_hint as guidance.

Respond with EXACTLY three sentences telling a BEFORE then AFTER then HOW story:
1. BEFORE: the problem the redispatch faced. If the base case had overloaded
   lines or voltage violations, lead with HOW MANY elements were over limit and
   the worst loading (base_case.n_overloads, base_case.max_line_loading_pct). If
   the base case was already within limits, say so and name the standing N-1 risk
   (security.n_insecure). Do NOT lead with the single-slack ext_grid blackout:
   that standing topological risk is the storage-siting question, not something
   this redispatch can fix.
2. AFTER: the solver-verified outcome. If VerificationReport.verified is true,
   say the base case is now secure with the violations cleared
   (deltas.violations_resolved) and that the most-severe N-1 contingencies held
   with none worsened. If verified is false, say the action did NOT reach a
   secure base and state the residual instead of claiming success.
3. HOW: how it was achieved, in ONE sentence: the redispatch
   (Action.total_abs_mw_shifted MW across Action.n_generator_moves generators,
   chiefly Action.top_moves), Action.estimated_cost_delta, the
   deltas.load_shed_avoided_mw it avoids, and one residual constraint to watch.
   Do NOT enumerate every setpoint.

You may round MW and euro figures to whole numbers for readability, but never
state a number that is not present in the input JSON.

Tone: terse control-room radio. No hedging, no exclamation marks, no headers.

INPUT:
GridStateSummary: {grid_state_summary_json}
Action: {action_json}
VerificationReport: {verification_report_json}\
"""


def summarize_action(action: dict) -> dict:
    """A compact, narration-ready view of an Action: aggregate MW shifted plus the few largest
    generator moves. The full per-element detail stays in action["changes"]; this only bounds what
    the narrator reads so it cannot recite every setpoint. All values derive deterministically from
    the solver-produced changes; none are invented."""
    changes = action.get("changes", [])

    def _delta(c) -> float:
        try:
            return abs(float(c["to"]) - float(c["from"]))
        except (TypeError, ValueError, KeyError):
            return 0.0

    p_moves = [c for c in changes if c.get("field") == "p_mw"]
    v_moves = [c for c in changes if c.get("field") == "vm_pu"]
    top = sorted(p_moves, key=_delta, reverse=True)[:3]
    return {
        "action_id": action.get("action_id"),
        "type": action.get("type"),
        "source": action.get("source"),
        "estimated_cost_delta": action.get("estimated_cost_delta", 0.0),
        "n_changes": len(changes),
        "n_generator_moves": len(p_moves),
        "n_voltage_setpoints": len(v_moves),
        "total_abs_mw_shifted": round(sum(_delta(c) for c in p_moves), 1),
        "top_moves": [
            {"etype": c.get("etype"), "index": c.get("index"), "from": c.get("from"), "to": c.get("to")}
            for c in top
        ],
    }


def render_narration_prompt(state: dict, action: dict, report: dict) -> str:
    """Fill the template; the Action is passed as a compact summary so the narrator states
    aggregate moves, not every setpoint."""
    return NARRATION_SYSTEM_PROMPT_TEMPLATE.format(
        grid_state_summary_json=json.dumps(state),
        action_json=json.dumps(summarize_action(action)),
        verification_report_json=json.dumps(report),
    )


def _fmt_mw(value) -> str:
    """Whole MW with thousands separators. Display rounding only; the value comes from the solver JSON."""
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(value) -> str:
    try:
        return f"{float(value):.0f}%"
    except (TypeError, ValueError):
        return str(value)


def _cost_phrase(value) -> str:
    """'a cost of EUR X' for a positive dispatch-cost delta, 'a net saving of EUR X' when negative.
    The number is the solver-produced estimated_cost_delta, rounded to whole euros for readability."""
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return f"a cost of €{value}"
    if cost < 0:
        return f"a net saving of €{abs(cost):,.0f}"
    return f"a cost of €{cost:,.0f}"


def _count(n, noun: str) -> str:
    """'1 line' / '7 lines' / '2 losses' (count from the inputs, plural is grammar not a quantity)."""
    if n == 1:
        return f"{n} {noun}"
    plural = noun + ("es" if noun.endswith(("s", "x", "z", "ch", "sh")) else "s")
    return f"{n} {plural}"


def _operational_watch_id(worst_list: list) -> str | None:
    """First worst contingency that is NOT the single-slack ext_grid blackout (which redispatch cannot
    fix); falls back to the overall worst only if every entry is the slack loss."""
    for w in worst_list:
        cid = str(w.get("contingency_id", ""))
        if cid and not cid.startswith("ext_grid"):
            return w.get("contingency_id")
    return worst_list[0].get("contingency_id") if worst_list else None


def template_narration(state: dict, action: dict, report: dict) -> list[str]:
    """Return exactly 3 control-room sentences telling a BEFORE then AFTER then HOW story, built only
    from numbers in the inputs (rounded for display, never computed or invented):

      1. BEFORE: the problem the redispatch faced (base overloads, or the standing N-1 risk).
      2. AFTER:  the solver-verified result (secure with violations cleared, or the honest residual).
      3. HOW:    the redispatch that achieved it (MW, generators, cost) and what it avoids.

    Deliberately does NOT lead with the single-slack ext_grid blackout: that standing topological risk
    is the storage-siting question, not something this redispatch can clear, and leading with it would
    misrepresent what the action does.
    """
    base = state.get("base_case", {}) or {}
    security = state.get("security", {}) or {}
    worst_list = security.get("worst", []) or []
    geo = state.get("geographic_context", {}) or {}
    region = geo.get("target_region_hint") if geo.get("available") else None

    n_over = base.get("n_overloads") or 0
    n_volt = base.get("n_voltage_violations") or 0
    max_load = base.get("max_line_loading_pct")
    total_load = base.get("total_load_mw")
    n_insecure = security.get("n_insecure")
    if n_insecure is None:
        n_insecure = len(worst_list)

    atype = action.get("type", "noop")
    changes = action.get("changes", [])
    cost_delta = action.get("estimated_cost_delta", 0.0)

    deltas = report.get("deltas", {}) or {}
    verified = bool(report.get("verified", False))
    viol_resolved = deltas.get("violations_resolved", 0)
    shed_avoided = deltas.get("load_shed_avoided_mw", 0.0)
    watch_id = _operational_watch_id(worst_list)
    watch = f"; keep {watch_id} on watch as the nearest remaining constraint." if watch_id else "."

    def _with_region(sentence: str) -> str:
        return f"{region}: {sentence}" if region else sentence

    # ---- no action: the base case is already secure, nothing to redispatch ----
    if atype == "noop" or not changes:
        worst_pct = f" (worst line at {_fmt_pct(max_load)})" if max_load is not None else ""
        s1 = _with_region(f"Before and after, the base case is secure, serving {_fmt_mw(total_load)} MW{worst_pct}.")
        s2 = "No redispatch was required this snapshot, so none was committed."
        s3 = (f"The N-1 sweep still flags {_count(n_insecure, 'single-element loss')} to keep on watch; "
              f"that standing risk is the storage-siting question, not a redispatch target.")
        return [s1, s2, s3]

    summ = summarize_action(action)
    mw = summ["total_abs_mw_shifted"]
    ngen = summ["n_generator_moves"]
    tops = summ["top_moves"][:2]
    moves = ", ".join(
        f"{m.get('etype', 'gen')} {m.get('index')} ({_fmt_mw(m.get('from'))} to {_fmt_mw(m.get('to'))} MW)"
        for m in tops
    )
    chiefly = f" (chiefly {moves})" if moves else ""
    base_insecure = (n_over > 0) or (n_volt > 0)

    # ---- 1. BEFORE: what the agent was facing ----
    if base_insecure:
        problems = []
        if n_over:
            worst_pct = f" (worst at {_fmt_pct(max_load)})" if max_load is not None else ""
            problems.append(f"{_count(n_over, 'line')} were loaded past their thermal limit{worst_pct}")
        if n_volt:
            problems.append(f"{_count(n_volt, 'bus')} sat outside the voltage band")
        s1 = _with_region(f"Before the agent acted, {' and '.join(problems)}, so the base case was insecure.")
    else:
        worst_pct = f" (worst line at {_fmt_pct(max_load)})" if max_load is not None else ""
        s1 = _with_region(
            f"Before the agent acted, the base case was already within limits{worst_pct}, but the "
            f"N-1 sweep flagged {_count(n_insecure, 'insecure single-element loss')}."
        )

    # ---- 2. AFTER and 3. HOW ----
    if verified and base_insecure:
        s2 = (f"Now the base case is secure: all {_count(viol_resolved, 'live violation')} cleared "
              f"with nothing left over its limit, and the most-severe N-1 contingencies still hold "
              f"with none worsened.")
        s3 = (f"This was achieved by redispatching {_fmt_mw(mw)} MW across {_count(ngen, 'generator')}"
              f"{chiefly}, at {_cost_phrase(cost_delta)}, which also avoids up to {_fmt_mw(shed_avoided)} "
              f"MW of cascade load shedding{watch}")
    elif verified and not base_insecure:
        s2 = ("After a precautionary redispatch the base case stays secure and the most-severe N-1 "
              "contingencies hold with none worsened.")
        s3 = (f"This was achieved by shifting {_fmt_mw(mw)} MW across {_count(ngen, 'generator')}"
              f"{chiefly}, at {_cost_phrase(cost_delta)}, pre-positioning generation to avoid up to "
              f"{_fmt_mw(shed_avoided)} MW of shedding if the worst line trips{watch}")
    else:
        s2 = (f"The redispatch did not reach a secure base: {_count(viol_resolved, 'violation')} cleared "
              f"so far, but a residual remains and is reported, not hidden.")
        s3 = (f"The proposed move shifts {_fmt_mw(mw)} MW across {_count(ngen, 'generator')}{chiefly}, at "
              f"{_cost_phrase(cost_delta)}, and avoids {_fmt_mw(shed_avoided)} MW of shedding so far; it "
              f"stays uncommitted until the AC cascade rescan shows the base fully secure{watch}")

    return [s1, s2, s3]
