"""Solver-verified battery counterfactual: the Verification Loop.

For a candidate bus, re-run the SAME full N-1 cascade sweep (slack
guard, islanding, cascade loop: src/engine/scan.run_contingency_sweep,
no duplicated physics anywhere) on a deep copy of the net that carries
one extra element: a pandapower-native storage unit
(pp.create_storage) at the candidate bus. The recommendation is then
the measured delta between the two sweeps, never an estimate.

Battery dispatch model (logged as D51): the storage is held at full
rated discharge (p_mw = -battery_p_mw; pandapower storage uses the
load sign convention, negative = injection) for every contingency of
the sweep. This is the deterministic redispatch heuristic: it answers
"does a battery discharging here help", runs through the exact same
runpp path as every production element, and needs no cost data.
controllable=True is set so future OPF-based dispatch can pick the
setpoint instead; the battery loop itself never calls runopp, and
opf_available(net) is checked and logged by the recommender so the
fallback is explicit on datasets without poly_cost.

Honesty: scenarios_worsened is measured and reported. A battery behind
the wrong side of a congested corridor, or one that pushes a lightly
loaded path over a limit, shows up here instead of being dropped.
"""

from __future__ import annotations

import copy
import logging
import time

import pandapower as pp

from src.battery.schemas import (
    PerContingencyDelta,
    VerificationResult,
)
from src.engine.network import native_index, slack_bus_set
from src.engine.scan import run_contingency_sweep

logger = logging.getLogger(__name__)

# CSS deadband for calling a scenario improved or worsened. 1 CSS point
# equals one residual violation; 0.5 keeps sub-violation-scale wobble
# (a fraction of a percent of system load shed) out of the verdict.
IMPROVEMENT_DEADBAND_CSS = 0.5

STABLE_STATUSES = ("SECURE", "VIOLATIONS")
CASCADING_STATUSES = ("CASCADE", "DIVERGED")

# A contingency is in the battery's "impact set" if its baseline state
# is insecure, OR any branch ends this near the trip threshold. A
# moderate battery cannot flip a contingency whose worst element sits
# comfortably below this, so such contingencies are carried forward
# unchanged instead of being re-solved (see impact_contingency_ids and
# the restrict_to path in verify_battery_candidate). 90 percent = within
# 30 points of the 120 percent trip threshold.
IMPACT_NEAR_LIMIT_PCT = 90.0


class SlackBusError(ValueError):
    """Battery placement on a slack bus is meaningless: the slack
    already provides unbounded P and Q at fixed voltage."""


class UnknownBusError(ValueError):
    """The requested bus does not exist in the active net."""


def resolve_bus(net, bus_idx):
    """Map an API-supplied bus reference onto the net's native index.

    Accepts the native value (int or str) and, as a convenience, the
    string form of an integer index ("23" for 23). Raises
    UnknownBusError otherwise. Never assumes the index dtype.
    """
    if bus_idx in net.bus.index:
        return native_index(bus_idx)
    if isinstance(bus_idx, str):
        try:
            as_int = int(bus_idx)
        except ValueError:
            raise UnknownBusError(f"bus {bus_idx!r} not found in the active net")
        if as_int in net.bus.index:
            return native_index(as_int)
    raise UnknownBusError(f"bus {bus_idx!r} not found in the active net")


def assert_not_slack(net, bus_idx) -> None:
    """The acceptance rule: verifying any bus in net.ext_grid.bus.values
    MUST raise before any solve. Set membership, never equality against
    a hardcoded index."""
    if bus_idx in slack_bus_set(net):
        raise SlackBusError(
            f"bus {bus_idx} is a slack (ext_grid) bus; battery placement "
            "there is invalid: the slack already provides unbounded P and Q"
        )


def add_virtual_battery(
    net, bus_idx, battery_p_mw: float, battery_max_e_mwh: float
):
    """Deep-copy the net and attach the virtual storage element at the
    candidate bus. Returns (net_with_battery, storage_idx). The input
    net is never mutated."""
    net_with_battery = copy.deepcopy(net)
    storage_idx = pp.create_storage(
        net_with_battery,
        bus=bus_idx,
        p_mw=-battery_p_mw,  # load sign convention: negative = discharge
        max_e_mwh=battery_max_e_mwh,
        q_mvar=0.0,
        min_p_mw=-battery_p_mw,
        max_p_mw=battery_p_mw,
        controllable=True,
        name="virtual_battery_candidate",
    )
    return net_with_battery, storage_idx


def impact_contingency_ids(baseline: list) -> set:
    """Contingency ids whose outcome a moderate battery could change:
    insecure (band CRITICAL or HIGH) or carrying a branch near the trip
    threshold in the baseline. Used to bound the verification re-solve
    for large grids (docs/battery-feature.md, performance).

    Honesty: this only ever EXCLUDES contingencies that are comfortably
    secure in the baseline, where a battery cannot manufacture a new
    cascade, so no worsened case is hidden. The deadband is also why the
    excluded set carries forward as a true zero delta. (Logged as D54.)
    """
    ids: set = set()
    for r in baseline:
        ml = r.final_state.get("max_loading_pct") or 0.0
        if r.severity.band in ("CRITICAL", "HIGH") or ml > IMPACT_NEAR_LIMIT_PCT:
            ids.add(r.contingency_id)
    return ids


def compute_verification_deltas(baseline: list, with_battery: list) -> dict:
    """Pure delta computation between two sweeps over the same
    contingency list. No solver, fully unit-testable."""
    base_by_id = {r.contingency_id: r for r in baseline}
    with_by_id = {r.contingency_id: r for r in with_battery}
    common_ids = [cid for cid in base_by_id if cid in with_by_id]
    if len(common_ids) != len(base_by_id) or len(common_ids) != len(with_by_id):
        logger.warning(
            "verification sweeps differ in contingency ids: %d baseline, "
            "%d with battery, %d common",
            len(base_by_id), len(with_by_id), len(common_ids),
        )

    improved = worsened = unchanged = prevented = 0
    per_contingency: list[PerContingencyDelta] = []
    vm_deltas: list[float] = []
    max_loading_reduction = 0.0
    shed_before = shed_after = 0.0

    for cid in common_ids:
        b = base_by_id[cid]
        w = with_by_id[cid]
        ds = w.severity.score - b.severity.score
        if ds < -IMPROVEMENT_DEADBAND_CSS:
            improved += 1
        elif ds > IMPROVEMENT_DEADBAND_CSS:
            worsened += 1
        else:
            unchanged += 1
        if b.status in CASCADING_STATUSES and w.status in STABLE_STATUSES:
            prevented += 1

        shed_before += b.severity.load_shed_mw
        shed_after += w.severity.load_shed_mw

        if abs(ds) > IMPROVEMENT_DEADBAND_CSS or b.status != w.status:
            per_contingency.append(
                PerContingencyDelta(
                    contingency_id=cid,
                    status_before=b.status,
                    status_after=w.status,
                    score_before=b.severity.score,
                    score_after=w.severity.score,
                    load_shed_before_mw=b.severity.load_shed_mw,
                    load_shed_after_mw=w.severity.load_shed_mw,
                )
            )

        # Voltage deltas over (scenario, bus) pairs present in both
        # end states; buses dead in one run have no comparable voltage.
        for bus_idx, vm_w in w.final_bus_vm.items():
            vm_b = b.final_bus_vm.get(bus_idx)
            if vm_b is not None:
                vm_deltas.append(vm_w - vm_b)

        # Largest single-line loading relief across (scenario, line).
        for line_idx, pct_b in b.final_line_loading.items():
            pct_w = w.final_line_loading.get(line_idx)
            if pct_w is not None:
                reduction = pct_b - pct_w
                if reduction > max_loading_reduction:
                    max_loading_reduction = reduction

    return {
        "n_scenarios": len(common_ids),
        "scenarios_improved": improved,
        "scenarios_unchanged": unchanged,
        "scenarios_worsened": worsened,
        "cascades_prevented": prevented,
        "mw_load_shed_avoided": shed_before - shed_after,
        "avg_voltage_improvement": (
            sum(vm_deltas) / len(vm_deltas) if vm_deltas else 0.0
        ),
        "max_loading_reduction": max_loading_reduction,
        "per_contingency": per_contingency,
    }


def classify(
    scenarios_improved: int,
    scenarios_worsened: int,
    cascades_prevented: int = 0,
    mw_load_shed_avoided: float = 0.0,
) -> str:
    """Verdict from measured deltas. NO_IMPACT and NOT_RECOMMENDED are
    legitimate, reported outcomes (honesty constraint).

    The verdict is net-benefit aware, not a raw improved-vs-worsened
    tally (logged as D58): a battery that prevents cascades and avoids
    load shedding on net is RECOMMENDED even if it marginally worsens a
    few lower-severity cases, because the per_contingency deltas report
    every worsened case explicitly (nothing is hidden). The bands:

    - NO_IMPACT      nothing changed either way
    - RECOMMENDED    clean win (no worsened cases) OR net security gain
                     (prevents a cascade, avoids shed on net, and helps
                     at least as many cases as it harms)
    - NOT_RECOMMENDED  net negative: harms more than it helps and saves
                     no load on net
    - MIXED          helps and harms with no clear net direction
    """
    if scenarios_improved == 0 and scenarios_worsened == 0:
        return "NO_IMPACT"
    if scenarios_worsened == 0 and scenarios_improved > 0:
        return "RECOMMENDED"
    net_security_gain = (
        cascades_prevented > 0
        and mw_load_shed_avoided > 0
        and scenarios_improved >= scenarios_worsened
    )
    if net_security_gain:
        return "RECOMMENDED"
    if scenarios_worsened > scenarios_improved and mw_load_shed_avoided <= 0:
        return "NOT_RECOMMENDED"
    return "MIXED"


def verify_battery_candidate(
    net,
    bus_idx,
    contingencies: list,
    baseline: list,
    battery_p_mw: float = 10.0,
    battery_max_e_mwh: float = 40.0,
    restrict_to: set | None = None,
) -> VerificationResult:
    """The Verification Loop for one candidate bus.

    Order of operations is load-bearing: resolve and slack-check the
    bus BEFORE any copy or solve (the slack rejection must cost zero
    solver calls), then attach the storage element on a deep copy and
    re-run the Phase 1 sweep with the battery in place.

    restrict_to: optional set of contingency ids to actually re-solve.
    When given, contingencies outside the set are carried forward from
    the baseline unchanged (a true zero delta, because a moderate
    battery cannot alter a comfortably-secure contingency, see
    impact_contingency_ids). The returned n_scenarios still counts the
    FULL set, so coverage is preserved; only provably-unchanged solves
    are skipped. restrict_to=None re-solves the full set (the honest
    default for tests and small grids).
    """
    t0 = time.perf_counter()
    resolved = resolve_bus(net, bus_idx)
    assert_not_slack(net, resolved)

    net_with_battery, _storage_idx = add_virtual_battery(
        net, resolved, battery_p_mw, battery_max_e_mwh
    )

    if restrict_to is None:
        to_solve = contingencies
    else:
        to_solve = [o for o in contingencies if o.contingency_id in restrict_to]
    solved = run_contingency_sweep(net_with_battery, to_solve)

    if restrict_to is None:
        with_battery = solved
    else:
        # Carry forward the baseline result for every contingency not
        # re-solved (delta 0); splice in the re-solved ones.
        solved_by_id = {r.contingency_id: r for r in solved}
        with_battery = [
            solved_by_id.get(b.contingency_id, b) for b in baseline
        ]

    deltas = compute_verification_deltas(baseline, with_battery)
    return VerificationResult(
        bus_idx=resolved,
        battery_p_mw=battery_p_mw,
        battery_max_e_mwh=battery_max_e_mwh,
        verdict=classify(
            deltas["scenarios_improved"],
            deltas["scenarios_worsened"],
            deltas["cascades_prevented"],
            deltas["mw_load_shed_avoided"],
        ),
        computation_time_ms=(time.perf_counter() - t0) * 1000.0,
        **deltas,
    )
