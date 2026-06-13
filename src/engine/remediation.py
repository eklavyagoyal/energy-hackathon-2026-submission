"""Redispatch remediation: OPF tighten-and-verify plus the greedy strawman (U2).

Ported from the warden build onto the canonical src engine. pandapower runopp is a BASE-CASE
AC-OPF, not security-constrained; we approximate SCOPF by tightening line limits, running OPF on a
working copy, then VERIFYING the proposed dispatch by re-running the full src cascade analysis
(scan.analyze_contingency) per offending contingency, never a static post-outage check. The OPF
solution sets generator MW and voltage setpoints, so the extracted Action captures gen p_mw AND
gen / ext_grid vm_pu, or the verification re-solve drifts from the OPF state.

The LLM never sees any of this; it is pure solver work. Dataset-agnostic: every element reference
goes through native_index and iterates net.<table>.index, never range(len()) and never an int-coerced
position. runopp is guarded by opf_available (closes audit A7) and a broad try/except.
"""
from __future__ import annotations

import hashlib
import logging
import time

import pandapower as pp

from src.engine.actions import Action, BaselineResult, VerificationReport
from src.engine.constants import OVERLOAD_LIMIT, VOLTAGE_BAND_HIGH, VOLTAGE_BAND_LOW
from src.engine.network import (
    base_case_summary,
    native_index,
    opf_available,
    total_load_mw,
    working_copy,
)
from src.engine.preflight import deenergize_and_count, unsupplied
from src.engine.scan import ContingencyResult, Outage, analyze_contingency

logger = logging.getLogger(__name__)

# Tighten-and-verify tunables (mirror the warden constants; local to remediation).
SECURITY_MARGIN_PCT = 85.0  # global line max_loading_percent target inside the OPF loop
OPF_RETRY_LIMIT = 3
OPF_RELAX_STEP_PCT = 5.0
_P_TOL = 0.1   # MW change below this is not worth listing
_V_TOL = 1e-3  # p.u. setpoint change below this is ignored
# Constrain the OPF voltage band a hair INSIDE the verifier band so a setpoint pinned to the OPF
# limit does not graze over it on the plain re-solve (the OPF returns e.g. 1.0500 but runpp settles
# at 1.050012, which the 1e-6-tolerance verifier would otherwise flag). 5e-3 p.u. is ample headroom.
_OPF_VM_MARGIN = 0.005


def _marginal_costs(net: "pp.pandapowerNet") -> dict:
    """Per-generator marginal cost (cp1, eur/MWh) from poly_cost, keyed by native gen index."""
    mc: dict = {}
    if "poly_cost" in net and len(net.poly_cost):
        for _, row in net.poly_cost.iterrows():
            if str(row["et"]) == "gen":
                mc[native_index(row["element"])] = float(row["cp1_eur_per_mw"])
    return mc


def _redispatch_cost(net: "pp.pandapowerNet", changes: list) -> float:
    """Counter-trade cost of a redispatch: sum of |MW moved| x marginal cost over the generator moves.
    Always >= 0 and is the figure an operator reads as 'how much generation did this action pay to
    shift', unlike opf_cost - base_cost which re-optimizes the whole dispatch and reads negative."""
    mc = _marginal_costs(net)
    total = 0.0
    for c in changes:
        if c.get("etype") == "gen" and c.get("field") == "p_mw":
            total += abs(float(c["to"]) - float(c["from"])) * mc.get(c["index"], 0.0)
    return round(total, 1)


def _action_id(source: str, changes: list) -> str:
    return f"act_{source}_{hashlib.md5(repr(changes).encode()).hexdigest()[:6]}"


def dispatch_cost(net: "pp.pandapowerNet") -> float:
    """Total generation cost (eur) of the current solved dispatch, from poly_cost polynomials."""
    if "poly_cost" not in net or len(net.poly_cost) == 0:
        return float("nan")
    cost = 0.0
    for _, row in net.poly_cost.iterrows():
        et, el = row["et"], row["element"]
        if et == "gen" and el in net.res_gen.index:
            p = float(net.res_gen.at[el, "p_mw"])
        elif et == "ext_grid" and el in net.res_ext_grid.index:
            p = float(net.res_ext_grid.at[el, "p_mw"])
        else:
            continue
        cost += float(row["cp0_eur"]) + float(row["cp1_eur_per_mw"]) * p + float(row["cp2_eur_per_mw2"]) * p * p
    return cost


def base_overloaded_lines(net: "pp.pandapowerNet") -> list:
    """Indices (native dtype) of in-service lines over OVERLOAD_LIMIT on the base case."""
    work = working_copy(net)
    try:
        pp.runpp(work)
    except Exception:
        return []
    ll = work.res_line.loading_percent
    return [native_index(i) for i in work.res_line.index[(ll > OVERLOAD_LIMIT) & work.line.in_service]]


def run_opf(
    net: "pp.pandapowerNet",
    tighten: list | None = None,
    global_max_loading: float | None = None,
    source: str = "opf",
):
    """Run AC-OPF on a working copy and extract a redispatch Action (gen p_mw + gen/ext_grid vm_pu).

    tighten: per-line overrides [{"etype":"line","index":i,"max_loading_pct":x}].
    global_max_loading: constrain every line to this first (a broad security margin that absorbs the
        PV-setpoint reproduction gap on the verification re-solve); per-line tighten overrides on top.
    Returns (action, opf_cost, base_cost, converged, predicted_base). Never mutates net. Returns
    converged=False (no crash) when cost data is missing or OPF does not converge (audit A7).
    """
    if not opf_available(net):
        logger.warning("run_opf: opf_available is False (no poly_cost / no controllable gen); skipping OPF")
        return None, float("nan"), float("nan"), False, None

    o = working_copy(net)
    # Constrain the OPF bus voltages to the SAME band the verifier enforces (src
    # VOLTAGE_BAND_LOW/HIGH), not pandapower's wider OPF defaults (~0.94 to 1.06). Without this the
    # OPF returns ~1.06 p.u. solutions that the 0.95 to 1.05 verification band then rejects as
    # over-voltage, so no redispatch could ever verify. This closes the audit DEVIATION the honest
    # way: the OPF respects the band rather than the report admitting a residual it cannot fix.
    o.bus["max_vm_pu"] = VOLTAGE_BAND_HIGH - _OPF_VM_MARGIN
    o.bus["min_vm_pu"] = VOLTAGE_BAND_LOW + _OPF_VM_MARGIN
    if global_max_loading is None and not tighten:
        o.line["max_loading_percent"] = SECURITY_MARGIN_PCT
    else:
        if global_max_loading is not None:
            o.line["max_loading_percent"] = float(global_max_loading)
        for t in (tighten or []):
            o[t["etype"]].loc[t["index"], "max_loading_percent"] = float(t["max_loading_pct"])

    try:
        pp.runpp(o)
        base_cost = dispatch_cost(o)
    except Exception:
        base_cost = float("nan")

    try:
        pp.runopp(o)
    except Exception:
        return None, float("nan"), base_cost, False, None

    changes: list = []
    for idx in o.gen.index:
        old_p, new_p = float(net.gen.at[idx, "p_mw"]), float(o.res_gen.at[idx, "p_mw"])
        if abs(new_p - old_p) > _P_TOL:
            changes.append({"etype": "gen", "index": native_index(idx), "field": "p_mw",
                            "from": round(old_p, 2), "to": round(new_p, 2)})
        old_v, new_v = float(net.gen.at[idx, "vm_pu"]), float(o.res_gen.at[idx, "vm_pu"])
        if abs(new_v - old_v) > _V_TOL:
            changes.append({"etype": "gen", "index": native_index(idx), "field": "vm_pu",
                            "from": round(old_v, 4), "to": round(new_v, 4)})
    for idx in o.ext_grid.index:
        bus_raw = o.ext_grid.at[idx, "bus"]
        old_v, new_v = float(net.ext_grid.at[idx, "vm_pu"]), float(o.res_bus.at[bus_raw, "vm_pu"])
        if abs(new_v - old_v) > _V_TOL:
            changes.append({"etype": "ext_grid", "index": native_index(idx), "field": "vm_pu",
                            "from": round(old_v, 4), "to": round(new_v, 4)})

    opf_cost = float(o.res_cost)
    # Report the counter-trade cost of the redispatch (positive, marginal-cost-weighted MW moved),
    # not opf_cost - base_cost (which re-optimizes the whole dispatch and reads as a negative
    # "saving" while the grid is congested, a credibility tell for a grid engineer).
    action = Action(
        action_id=_action_id(source, changes),
        type="redispatch" if changes else "noop",
        changes=changes, source=source,
        estimated_cost_delta=_redispatch_cost(net, changes),
    )
    predicted_base = {
        "converged": True,
        "max_line_loading_pct": round(float(o.res_line.loading_percent.max()), 1),
        "min_vm_pu": round(float(o.res_bus.vm_pu.min()), 4),
    }
    return action, opf_cost, base_cost, True, predicted_base


def apply_action_to_net(net: "pp.pandapowerNet", action: Action) -> None:
    """Apply an Action's changes to net in place (gen/ext_grid setpoints, load shed)."""
    for c in action.changes:
        et, field = c["etype"], c["field"]
        if et == "gen" and field == "p_mw":
            net.gen.loc[c["index"], "p_mw"] = float(c["to"])
        elif et == "gen" and field == "vm_pu":
            net.gen.loc[c["index"], "vm_pu"] = float(c["to"])
        elif et == "ext_grid" and field == "vm_pu":
            net.ext_grid.loc[c["index"], "vm_pu"] = float(c["to"])
        elif et == "load" and field == "p_mw":
            net.load.loc[c["index"], "p_mw"] = float(c["to"])
        elif et == "load" and field == "in_service":
            net.load.loc[c["index"], "in_service"] = bool(c["to"])


def _base_entry(net: "pp.pandapowerNet") -> dict:
    m = base_case_summary(net)
    if not m["converged"]:
        status = "DIVERGED"
    elif (m["n_overloads"] or 0) == 0 and (m["n_voltage_violations"] or 0) == 0:
        status = "SECURE"
    else:
        status = "VIOLATIONS"
    score = float((m["n_overloads"] or 0) + (m["n_voltage_violations"] or 0))
    return {"contingency_id": "base", "status": status, "score": score}


def _cont_entry(net: "pp.pandapowerNet", outage: Outage) -> dict:
    r = analyze_contingency(net, outage)
    return {"contingency_id": r.contingency_id, "status": r.status, "score": r.severity.score}


def verify_action(
    net: "pp.pandapowerNet",
    action: Action,
    offending: list,
    method: str = "ac_cascade_rescan",
) -> VerificationReport:
    """Apply the action on a working copy and re-check the base case + each offending contingency
    with the full AC cascade. verified = base SECURE AND no offending contingency made worse.
    Topological risks the action cannot fix (blackout / islanding) stay equal and so pass here; they
    are residual risk, not a block."""
    outages = {
        r.contingency_id: Outage(r.outage["etype"], r.outage["index"], r.outage["name"])
        for r in offending
    }
    before = [_base_entry(net)] + [_cont_entry(net, o) for o in outages.values()]
    after_net = working_copy(net)
    apply_action_to_net(after_net, action)
    after = [_base_entry(after_net)] + [_cont_entry(after_net, o) for o in outages.values()]

    base_before = before[0]
    base_after = after[0]
    base_ok = base_after["status"] == "SECURE"
    cont_ok = all(a["score"] <= b["score"] + 1e-6 for b, a in zip(before[1:], after[1:]))
    verified = bool(base_ok and cont_ok)

    # Load-shed avoided is reported PER WORST single contingency, not summed across the offending set:
    # N-1 events are mutually exclusive (one element trips at a time), so summing their independent
    # shed double-counts the same load and can exceed total grid load (a physics-impossible figure).
    # The honest number is "if the worst N-1 occurs, this action avoids X MW of shedding".
    avoided_by_cont = [
        r.severity.load_shed_mw - analyze_contingency(after_net, o).severity.load_shed_mw
        for r, o in zip(offending, outages.values())
    ]
    load_shed_avoided = max([0.0, *avoided_by_cont])
    worst_before = max((e["score"] for e in before), default=0.0)
    worst_after = max((e["score"] for e in after), default=0.0)
    return VerificationReport(
        verified=verified, method=method,
        contingency_ids=list(outages.keys()),
        before=before, after=after,
        deltas={
            "violations_resolved": int(max(0.0, base_before["score"] - base_after["score"])),
            "load_shed_avoided_mw": round(max(0.0, load_shed_avoided), 1),
            "worst_score_before": round(worst_before, 1),
            "worst_score_after": round(worst_after, 1),
        },
        committed=False,
    )


def propose_remediation(net: "pp.pandapowerNet", offending: list | None = None):
    """Tighten-and-verify loop. Tighten all lines to a global security margin, run OPF on a copy,
    verify with the full cascade rescan, iterate up to OPF_RETRY_LIMIT pushing the target harder.
    Returns the best (Action, VerificationReport) found, with an honest verified flag."""
    offending = offending or []
    target = SECURITY_MARGIN_PCT
    best = None
    for _ in range(OPF_RETRY_LIMIT + 1):
        action, _opf, _base, converged, _pred = run_opf(net, global_max_loading=target)
        if not converged or action is None:
            target = min(98.0, target + OPF_RELAX_STEP_PCT)  # OPF infeasible: relax and retry
            continue
        report = verify_action(net, action, offending)
        best = (action, report)
        if report.verified:
            return best
        target = max(60.0, target - OPF_RELAX_STEP_PCT)  # not yet secure: push harder
    if best is not None:
        return best
    noop = Action(_action_id("opf", []), "noop", [], "opf", 0.0)
    return noop, verify_action(net, noop, offending)


CURTAIL_MAX_FRACTION = 0.9     # shed at most 90% of the TARGETED (congested) loads; beyond that, uncontrollable
CURTAIL_BISECT_STEPS = 12      # ~0.02 percent resolution on the shed fraction
VOLL_EUR_PER_MWH = 5000.0      # value of lost load: curtailment is the EXPENSIVE last-resort lever


def _congested_load_buses(net: "pp.pandapowerNet") -> set:
    """Buses at the endpoints of the currently overloaded lines/trafos, plus their one-hop neighbours.
    Curtailing load HERE relieves the binding constraint; uniform grid-wide shedding does not (it
    wastes shed on loads that do not feed the congested corridor). Empty if the base solve fails."""
    w = working_copy(net)
    try:
        pp.runpp(w)
    except Exception:
        return set()
    buses: set = set()
    ll = w.res_line.loading_percent
    for i in w.res_line.index[(ll > OVERLOAD_LIMIT) & w.line.in_service]:
        buses.add(w.line.at[i, "from_bus"])
        buses.add(w.line.at[i, "to_bus"])
    if len(w.res_trafo):
        tl = w.res_trafo.loading_percent
        for i in w.res_trafo.index[(tl > OVERLOAD_LIMIT) & w.trafo.in_service]:
            buses.add(w.trafo.at[i, "hv_bus"])
            buses.add(w.trafo.at[i, "lv_bus"])
    neighbours = set(buses)
    for _, r in net.line[net.line.in_service].iterrows():
        if r["from_bus"] in buses:
            neighbours.add(r["to_bus"])
        if r["to_bus"] in buses:
            neighbours.add(r["from_bus"])
    return neighbours


def _secure_after_targeted_shed(net, base_p, target_loads, fraction: float) -> bool:
    """Is the base case SECURE if the TARGETED loads' active power is scaled by (1 - fraction)? Scales
    only p_mw on target_loads, matching what the curtailment Action applies (bisection == verify)."""
    w = working_copy(net)
    for idx in target_loads:
        w.load.at[idx, "p_mw"] = base_p[idx] * (1.0 - fraction)
    m = base_case_summary(w)
    return bool(m["converged"]) and (m["n_overloads"] or 0) == 0 and (m["n_voltage_violations"] or 0) == 0


def propose_curtailment(net: "pp.pandapowerNet", offending: list | None = None, source: str = "curtail"):
    """Targeted load curtailment lever (the operator's 'curtail output' option): identify the loads on
    the congested corridor (endpoints of the overloaded elements + one-hop neighbours), then bisect the
    SMALLEST shed fraction on JUST those loads that restores base security, build a load-shed Action,
    and verify it on the full AC cascade. Targeted shedding relieves the binding constraint efficiently
    where uniform grid-wide shedding cannot. This is the last resort after redispatch; its cost is the
    value of lost load (VOLL), which keeps it correctly more expensive than redispatch.
    Returns (Action, VerificationReport)."""
    offending = offending or []
    target_buses = _congested_load_buses(net)
    target_loads = [idx for idx in net.load.index
                    if (net.load.at[idx, "bus"] in target_buses if target_buses else True)
                    and net.load.at[idx, "in_service"]]
    if not target_loads:
        noop = Action(_action_id(source, []), "noop", [], source, 0.0)
        return noop, verify_action(net, noop, offending, method="curtailment_rescan")

    base_p = net.load["p_mw"].copy()
    if _secure_after_targeted_shed(net, base_p, target_loads, CURTAIL_MAX_FRACTION):
        lo, hi = 0.0, CURTAIL_MAX_FRACTION
        for _ in range(CURTAIL_BISECT_STEPS):
            mid = (lo + hi) / 2.0
            if _secure_after_targeted_shed(net, base_p, target_loads, mid):
                hi = mid
            else:
                lo = mid
        fraction = hi
    else:
        fraction = CURTAIL_MAX_FRACTION  # best effort; verify will report it not fully secure

    changes: list = []
    shed_mw = 0.0
    for idx in target_loads:
        old = float(base_p[idx])
        new = round(old * (1.0 - fraction), 2)
        if abs(old - new) > _P_TOL:
            changes.append({"etype": "load", "index": native_index(idx), "field": "p_mw",
                            "from": round(old, 2), "to": new})
            shed_mw += old - new
    shed_mw = round(shed_mw, 1)
    action = Action(
        action_id=_action_id(source, changes),
        type="curtailment" if changes else "noop",
        changes=changes, source=source,
        estimated_cost_delta=round(shed_mw * VOLL_EUR_PER_MWH, 1),
        rationale=(f"curtail {shed_mw} MW ({round(fraction * 100, 1)}%) of load on the congested corridor "
                   f"({len(changes)} buses) to restore base security"),
    )
    return action, verify_action(net, action, offending, method="curtailment_rescan")


def propose_secure_action(net: "pp.pandapowerNet", offending: list | None = None):
    """The agent's full lever set, cheapest-first: try generator REDISPATCH (one OPF), and only if
    that cannot be verified secure, fall back to load CURTAILMENT. Returns (Action, VerificationReport)
    for whichever is verified, preferring redispatch; if neither verifies, return the redispatch
    attempt when it did something, else the curtailment attempt (honest best effort, verified=False)."""
    offending = offending or []
    action, _opf, _base, converged, _pred = run_opf(net, global_max_loading=SECURITY_MARGIN_PCT)
    if converged and action is not None:
        report = verify_action(net, action, offending)
        if report.verified:
            return action, report
    else:
        action, report = None, None

    c_action, c_report = propose_curtailment(net, offending)
    if c_report.verified:
        return c_action, c_report
    if action is not None and action.changes:
        return action, report
    return c_action, c_report


def greedy_policy(net: "pp.pandapowerNet", trip_at: float = OVERLOAD_LIMIT, max_steps: int = 20) -> BaselineResult:
    """Strawman baseline: repeatedly trip the single most-overloaded in-service line, no lookahead.
    Demonstrates how naive intervention cascades. Records a trace for animation."""
    t0 = time.perf_counter()
    g = working_copy(net)
    initial_overloads = len(base_overloaded_lines(net))
    steps: list = []
    diverged = False
    shed = 0.0
    worst = 0.0
    for it in range(max_steps):
        try:
            pp.runpp(g)
        except Exception:
            diverged = True
            shed += total_load_mw(g)
            break
        ll = g.res_line.loading_percent
        in_svc = g.line.in_service
        worst_idx = ll[in_svc].idxmax()
        worst = float(ll[worst_idx])
        if worst <= trip_at:
            break
        g.line.loc[worst_idx, "in_service"] = False
        steps.append({"iter": it + 1, "tripped_line": native_index(worst_idx), "loading_pct": round(worst, 1)})
        stranded = unsupplied(g)
        if stranded:
            shed += deenergize_and_count(g, stranded)

    secure_after = (not diverged) and worst <= OVERLOAD_LIMIT
    return BaselineResult(
        policy="greedy",
        secure_after=bool(secure_after),
        violations_resolved=int(initial_overloads if secure_after else 0),
        load_shed_mw=round(shed, 1),
        worst_cascade_depth_after=len(steps),
        redispatch_cost=0.0,
        wall_ms=round((time.perf_counter() - t0) * 1000, 1),
        explanation_available=False,
        trace=steps,
    )
