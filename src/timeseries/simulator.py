"""The time-stepped simulation loop (U8): the headline real-time deliverable.

Per step: apply the load/gen profile, apply firing events, observe the solver base state, run a fast
N-1 screen, let the agent decide, VERIFY the action on a clone with the full cascade, and only then
treat it as applied for that interval. The LLM never computes physics; every number is solver
output. Reuses the Phase 1 engine unchanged (no physics reimplemented here).

Framing (KTD4, audit): this is offline sequential decision support over a horizon, not real-time
control of hardware. The pitch: every action, at every timestep, is solver-verified before it would
be applied, over hours of operation, not just one snapshot.

Performance: the 24-step / <30s budget rules out a full 187-contingency cascade sweep every step,
so each step screens the cascade-prone set (top-K most-loaded in-service lines by base loading, plus
all trafos and ext_grids) and runs the full AC cascade on that set. Narration is the deterministic
template by default (fast, free, numbers-from-solver); narrate_llm=True swaps in real LLM narration
per step for a polished demo recording (slower, costs tokens).
"""
from __future__ import annotations

from src.agent.prompts import template_narration
from src.agent.state import build_grid_state_summary
from src.engine.actions import Action, to_jsonable
from src.engine.network import base_case_summary, native_index, working_copy
from src.engine.remediation import (
    apply_action_to_net,
    greedy_policy,
    propose_secure_action,
)
from src.engine.scan import Outage, element_name, rank, run_contingency_sweep
from src.timeseries.events import EventStream
from src.timeseries.profiles import Profile, apply_profile, capture_base
from src.timeseries.trace import SimulationTrace, TimeStepRecord

_INSECURE_BANDS = ("CRITICAL", "HIGH")
_SCREEN_TOP_LINES = 8
# Cap how many offending contingencies the per-step verify re-checks. The OPF tightens ALL lines
# globally regardless, so this only bounds the verification cost (the per-step <30s budget), not
# the remediation itself; the worst-ranked few are the ones worth proving.
_VERIFY_TOP_OFFENDING = 3


def _edge_loadings(net) -> dict:
    """Per-edge loading_percent (line_<i>/trafo_<i>) of the current solved net, for the timeline UI."""
    import pandapower as pp

    work = working_copy(net)
    out: dict = {}
    try:
        pp.runpp(work)
    except Exception:
        return out
    ll = work.res_line.loading_percent
    for idx in work.res_line.index[work.line.in_service]:
        v = float(ll.at[idx])
        if v == v:
            out[f"line_{native_index(idx)}"] = round(v, 1)
    if len(work.res_trafo):
        tl = work.res_trafo.loading_percent
        for idx in work.res_trafo.index[work.trafo.in_service]:
            v = float(tl.at[idx])
            if v == v:
                out[f"trafo_{native_index(idx)}"] = round(v, 1)
    return out


def _screen_contingencies(net, k: int = _SCREEN_TOP_LINES) -> list:
    """Fast N-1 screen: the cascade-prone set = top-k in-service lines by base loading, plus every
    in-service trafo and every ext_grid (the blackout path). Honest reduction for the per-step budget;
    the most-loaded lines are the most likely to cascade on N-1."""
    import pandapower as pp

    work = working_copy(net)
    outs: list = []
    try:
        pp.runpp(work)
        ll = work.res_line.loading_percent[net.line.in_service].dropna()
        top = ll.sort_values(ascending=False).head(k).index
    except Exception:
        top = list(net.line.index[net.line.in_service])[:k]
    for i in top:
        outs.append(Outage("line", native_index(i), element_name(net, "line", i)))
    for i in net.trafo.index[net.trafo.in_service]:
        outs.append(Outage("trafo", native_index(i), element_name(net, "trafo", i)))
    for i in net.ext_grid.index:
        outs.append(Outage("ext_grid", native_index(i), element_name(net, "ext_grid", i)))
    return outs


def _worst_summary(r) -> dict:
    return {
        "contingency_id": r.contingency_id,
        "outage_name": r.outage.get("name", r.contingency_id),
        "status": r.status,
        "severity": to_jsonable(r.severity),
    }


def _empty_verification(verified: bool, method: str) -> dict:
    return {"verified": verified, "method": method, "contingency_ids": [], "before": [], "after": [],
            "deltas": {"violations_resolved": 0, "load_shed_avoided_mw": 0.0,
                       "worst_score_before": 0.0, "worst_score_after": 0.0}}


def _noop_narration(state: dict) -> list:
    """Base-secure interval with a standing N-1 risk: name the risk from solver numbers, say why no
    real-time redispatch is committed, and point the residual at storage siting. No invented numbers."""
    worst_list = state.get("security", {}).get("worst", [])
    geo = state.get("geographic_context", {}) or {}
    region = geo.get("target_region_hint") if geo.get("available") else None
    if worst_list:
        w = worst_list[0]
        sev = w.get("severity", {})
        s1 = (f"Worst standing N-1 is loss of {w.get('outage_name', w.get('contingency_id'))} "
              f"(band {sev.get('band', 'UNKNOWN')}, cascade_depth {sev.get('cascade_depth', 0)}).")
    else:
        s1 = "No HIGH or CRITICAL N-1 contingencies on the current state."
    if region:
        s1 = f"{region}: {s1}"
    s2 = "Base case is within limits this interval, so no real-time redispatch is committed."
    s3 = ("A base-case OPF cannot N-1-secure a standing contingency; that residual is the "
          "storage-siting question handled by the battery recommender, kept on watch.")
    return [s1, s2, s3]


def _llm_or_template(state: dict, action: dict, verification: dict, agent_mode: str, replay: bool) -> list:
    """Real LLM narration only for an opt-in, non-replay llm run with a key; else the deterministic
    template. Either way every number originates from the solver JSON (the LLM never computes physics)."""
    if agent_mode == "llm" and not replay:
        try:
            import anthropic

            from src.agent import anthropic_key
            from src.agent.loop import _narrate_with_llm
            key = anthropic_key()
            if key:
                return _narrate_with_llm(anthropic.Anthropic(api_key=key), state, action, verification)
        except Exception:
            pass
    return template_narration(state, action, verification)


def _decide(net, agent_mode: str, sweep: list, state: dict, replay: bool):
    """Return (action_dict, verification_dict, narration_sentences, commit_status).

    opf/llm: ONE operator OPF per interval when the base case is insecure, then VERIFY on the full
    AC cascade rescan before commit (verify-before-commit, the headline integrity property). A
    base-secure interval is a noop (standing N-1 risk is narrated, not redispatched away). An
    infeasible OPF is an honest rejection: redispatch alone cannot serve the load within limits.
    greedy: trip-and-hope strawman, commits unverified, for contrast."""
    if agent_mode == "greedy":
        br = greedy_policy(net)
        action = {"action_id": "greedy", "type": "greedy_trips", "changes": br.trace,
                  "source": "greedy", "estimated_cost_delta": 0.0}
        verification = _empty_verification(bool(br.secure_after), "greedy_no_verify")
        verification["deltas"]["violations_resolved"] = br.violations_resolved
        narration = [f"Greedy policy tripped {br.worst_cascade_depth_after} line(s) with no lookahead.",
                     f"Load shed so far {br.load_shed_mw} MW.",
                     "Strawman baseline: no solver verification before committing."]
        return action, verification, narration, "applied_unverified"

    if agent_mode not in ("opf", "llm"):
        raise ValueError(f"unknown agent_mode {agent_mode!r}; use llm | opf | greedy")

    base = state.get("base_case", {})
    base_violations = (base.get("n_overloads") or 0) + (base.get("n_voltage_violations") or 0)
    if base_violations == 0:
        action = {"action_id": "act_noop", "type": "noop", "changes": [], "source": agent_mode,
                  "estimated_cost_delta": 0.0}
        return action, _empty_verification(True, "ac_cascade_rescan"), _noop_narration(state), "noop"

    offending = [r for r in sweep if r.severity.band in _INSECURE_BANDS
                 and r.outage.get("etype") in ("line", "trafo")][:_VERIFY_TOP_OFFENDING]
    # Cheapest-first: redispatch, then curtailment as the last-resort lever (the operator's full set).
    action_obj, report = propose_secure_action(net, offending)
    action, verification = to_jsonable(action_obj), to_jsonable(report)
    if action_obj.type == "curtailment":
        narration = _curtail_narration(state, action, verification, base_violations)
    else:
        narration = _llm_or_template(state, action, verification, agent_mode, replay)
    commit = "applied" if report.verified else "rejected_infeasible"
    return action, verification, narration, commit


def _curtail_narration(state: dict, action: dict, verification: dict, base_violations: int) -> list:
    """Three solver-grounded sentences for a load-curtailment action (numbers from the report/action)."""
    d = verification.get("deltas", {})
    s1 = (f"Base case is over limits with {base_violations} violation(s); generator redispatch alone "
          "cannot restore security this interval.")
    s2 = f"Agent uses the curtailment lever: {action.get('rationale', 'shed load to restore security')}."
    if verification.get("verified"):
        s3 = (f"Full AC cascade rescan confirms the curtailment restores a secure base "
              f"({d.get('violations_resolved', 0)} violations resolved, {d.get('load_shed_avoided_mw', 0.0)} "
              "MW of cascade shedding avoided).")
    else:
        s3 = "Even maximum curtailment cannot fully secure this state; residual risk remains and is reported, not hidden."
    return [s1, s2, s3]


def run_simulation(
    net,
    load_profile: Profile,
    gen_profile: Profile,
    event_stream: EventStream,
    agent_mode: str = "opf",
    horizon_steps: int = 24,
    seed: int = 42,
    profile_id: str = "synthetic_24h",
    event_scenario: str = "default",
    step_minutes: int = 60,
    replay: bool = True,
    scenario_context: dict | None = None,
) -> SimulationTrace:
    """Run the time-stepped loop and return a SimulationTrace. Events persist across steps (until
    they expire); profiles set loads/gens from the captured base each step (idempotent, no drift);
    the agent re-decides each interval and every action is solver-verified before commit."""
    work = working_copy(net)
    base = capture_base(work)
    scenario_context = scenario_context or net.get("geographic_scenario", {})
    trace = SimulationTrace(
        profile_id=profile_id,
        event_scenario=event_scenario,
        agent_mode=agent_mode,
        horizon_steps=horizon_steps,
        step_minutes=step_minutes,
        seed=seed,
        scenario=scenario_context.get("name"),
        scenario_title=scenario_context.get("title"),
        scenario_context=scenario_context,
    )

    for t in range(horizon_steps):
        apply_profile(work, base, load_profile, gen_profile, t)
        firing = event_stream.events_at(t)
        for ev in firing:
            ev.apply(work)

        baseline_state = base_case_summary(work)
        sweep = rank(run_contingency_sweep(work, _screen_contingencies(work)))
        # Separate the load-driven OPERATIONAL risk (line/trafo N-1, remediable, what moves over the
        # day) from the standing TOPOLOGICAL risk (losing the single ext_grid is always a blackout on
        # case118: a fixed property of the test case, not something redispatch can fix). The timeline
        # tracks the operational picture; the blackout is flagged, not hidden, and is the battery's job.
        op_sweep = [r for r in sweep if r.outage.get("etype") in ("line", "trafo")]
        standing_blackout = any(r.status == "FULL_BLACKOUT" for r in sweep)
        worst = _worst_summary(op_sweep[0]) if op_sweep else None
        n_insecure = sum(1 for r in op_sweep if r.severity.band in _INSECURE_BANDS)

        state = build_grid_state_summary(work, {"results": op_sweep}, f"{event_scenario}@t{t}", scenario_context)
        action, verification, narration, commit_status = _decide(work, agent_mode, op_sweep, state, replay)

        # Verify before commit: apply the action on a clone and re-summarize (the solver is the referee).
        after = working_copy(work)
        if action.get("type") in ("redispatch", "curtailment") and action.get("changes"):
            apply_action_to_net(after, Action(action["action_id"], action["type"], action["changes"],
                                              action.get("source", "opf"), action.get("estimated_cost_delta", 0.0)))
        elif action.get("type") == "greedy_trips":
            for trip in action.get("changes", []):
                li = trip.get("tripped_line")
                if li is not None and li in after.line.index:
                    after.line.loc[li, "in_service"] = False
        verified_state = base_case_summary(after)

        trace.steps.append(TimeStepRecord(
            t=t,
            timestamp=f"{(t * step_minutes) // 60:02d}:{(t * step_minutes) % 60:02d}",
            events=[ev.to_dict() for ev in firing],
            baseline_state=baseline_state,
            worst_contingency=worst,
            agent_action=action,
            commit_status=commit_status,
            verification=verification,
            verified_state=verified_state,
            narration=narration,
            edge_loadings=_edge_loadings(work),
            islanded_buses=(op_sweep[0].all_islanded_buses if op_sweep else []),
            n_insecure=n_insecure,
            standing_blackout_risk=standing_blackout,
        ))
        event_stream.cleanup(t, work)

    return trace
