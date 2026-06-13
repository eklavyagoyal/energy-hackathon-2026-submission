"""Per-bus battery placement scoring.

Pure aggregation over Phase 1 ContingencyResult records: NO solver
call, NO LLM call, fully unit-testable with synthetic results. The
exact formula (docs/battery-feature.md):

  congestion_score(b) = fraction of N-1 scenarios where ANY line
      connected to b has post-cascade loading_percent > 80
  voltage_score(b)    = fraction of scenarios where post-cascade vm_pu
      at b is < 0.95 or > 1.05
  cascade_score(b)    = fraction of scenarios where b is in an islanded
      component OR part of the cascade chain (a connected line or
      trafo tripped during the cascade)
  severity_weight(b)  = sum of total_load_shed_mw over scenarios where
      b had any issue, normalized by the max across all buses

  score(b) = w_congestion * congestion_score(b)
           + w_voltage    * voltage_score(b)
           + w_cascade    * cascade_score(b)
           + w_severity   * severity_weight(b)

All components live in [0, 1]; weights are normalized to sum to 1, so
the final score is in [0, 1] as well.

Interpretation decisions (logged as D56/D57):
- the congestion criterion is lines-only, exactly as the formula says;
  the cascade-chain criterion counts line AND trafo trips (a tripped
  trafo at b is unambiguously cascade involvement at b)
- the initiating outage itself is not a cascade trip; only elements
  tripped DURING the cascade count
- FULL_BLACKOUT scenarios affect every bus uniformly (all buses
  islanded, full system load shed); they carry no per-line or
  per-voltage data and shift all candidates equally
- a bus with no voltage result in a scenario (islanded or never
  converged) is counted by the cascade component, not double-counted
  by the voltage component

Candidate set: every IN-SERVICE bus that is NOT in
net.ext_grid.bus.values (set membership; a battery on a slack bus is
meaningless because the slack already provides unbounded P and Q).

Scaling note: the pass below is O(scenarios x (lines + buses)). For
5000+ bus grids with full N-1 sets this should be vectorized over
numpy arrays; the record shapes already support that.
"""

from __future__ import annotations

from collections import defaultdict

from src.battery.schemas import (
    BusScore,
    BusScoreContext,
    ScoreBreakdown,
    ScoreWeights,
)
from src.engine.constants import VOLTAGE_BAND_HIGH, VOLTAGE_BAND_LOW, VOLTAGE_TOL
from src.engine.network import native_index, slack_bus_set

# A line above this post-cascade loading marks its endpoint buses as
# congestion-affected in that scenario. Deliberately below
# OVERLOAD_LIMIT (100): the score looks for chronic stress, not only
# hard violations.
CONGESTION_LINE_LOADING_PCT = 80.0


def _branch_endpoint_maps(net) -> tuple[dict, dict]:
    """index -> (endpoint buses) maps for lines and trafos, keyed and
    valued with native index types."""
    line_ends: dict = {}
    for idx, row in net.line.iterrows():
        line_ends[native_index(idx)] = (
            native_index(row.from_bus),
            native_index(row.to_bus),
        )
    trafo_ends: dict = {}
    if len(net.trafo) > 0:
        for idx, row in net.trafo.iterrows():
            trafo_ends[native_index(idx)] = (
                native_index(row.hv_bus),
                native_index(row.lv_bus),
            )
    return line_ends, trafo_ends


def _worst_voltage(lo: float | None, hi: float | None) -> tuple[float | None, str]:
    """Pick the voltage extreme furthest outside the band, with its
    direction, so narration says 'rose to' for overvoltage and 'dropped
    to' for undervoltage. Returns (value, 'low'|'high'|'none')."""
    under = (
        (VOLTAGE_BAND_LOW - lo)
        if lo is not None and lo < VOLTAGE_BAND_LOW - VOLTAGE_TOL
        else 0.0
    )
    over = (
        (hi - VOLTAGE_BAND_HIGH)
        if hi is not None and hi > VOLTAGE_BAND_HIGH + VOLTAGE_TOL
        else 0.0
    )
    if under == 0.0 and over == 0.0:
        return None, "none"
    if under >= over:
        return lo, "low"
    return hi, "high"


def candidate_buses(net) -> tuple[list, list]:
    """(candidates, excluded_slack) with the slack exclusion applied as
    set membership against ALL ext_grid buses."""
    slack = slack_bus_set(net)
    candidates = []
    for b in net.bus.index[net.bus.in_service]:
        nb = native_index(b)
        if nb not in slack:
            candidates.append(nb)
    return candidates, sorted(slack, key=str)


def score_buses(
    net, results: list, weights: ScoreWeights | None = None
) -> list[BusScore]:
    """Aggregate a full N-1 sweep into ranked per-bus scores.

    `results` is the list of Phase 1 ContingencyResult records (or any
    objects exposing the same fields; the scoring tests use synthetic
    stand-ins). Returns BusScore records sorted by score descending,
    tiebreak bus_idx ascending as strings (deterministic on mixed key
    types).
    """
    weights = (weights or ScoreWeights()).normalized()
    line_ends, trafo_ends = _branch_endpoint_maps(net)
    candidates, _ = candidate_buses(net)
    candidate_set = set(candidates)
    n = len(results)

    congestion_count: dict = defaultdict(int)
    voltage_count: dict = defaultdict(int)
    cascade_count: dict = defaultdict(int)
    severity_sum: dict = defaultdict(float)
    worst_line: dict = {}
    worst_line_pct: dict = defaultdict(float)
    min_voltage: dict = {}
    max_voltage: dict = {}

    for r in results:
        if r.severity.blackout:
            # Uniform impact: every candidate sits in the dead system.
            for b in candidate_set:
                cascade_count[b] += 1
                severity_sum[b] += r.severity.load_shed_mw
            continue

        # Congestion membership, at most once per bus per scenario.
        congested_buses: set = set()
        for line_idx, pct in r.final_line_loading.items():
            ends = line_ends.get(line_idx)
            if ends is None:
                continue
            for b in ends:
                if b not in candidate_set:
                    continue
                # Narration context: the worst connected line ever seen.
                if pct > worst_line_pct[b]:
                    worst_line_pct[b] = pct
                    worst_line[b] = line_idx
                if pct > CONGESTION_LINE_LOADING_PCT:
                    congested_buses.add(b)

        # Voltage membership and context.
        voltage_buses: set = set()
        for bus_idx, vm in r.final_bus_vm.items():
            if bus_idx not in candidate_set:
                continue
            prev_lo = min_voltage.get(bus_idx)
            if prev_lo is None or vm < prev_lo:
                min_voltage[bus_idx] = vm
            prev_hi = max_voltage.get(bus_idx)
            if prev_hi is None or vm > prev_hi:
                max_voltage[bus_idx] = vm
            if vm < VOLTAGE_BAND_LOW - VOLTAGE_TOL or vm > VOLTAGE_BAND_HIGH + VOLTAGE_TOL:
                voltage_buses.add(bus_idx)

        # Cascade-chain membership: islanded, or endpoint of a tripped
        # branch element.
        cascade_buses: set = {
            b for b in r.all_islanded_buses if b in candidate_set
        }
        for item in r.tripped_elements:
            ends = (
                line_ends.get(item["index"])
                if item["etype"] == "line"
                else trafo_ends.get(item["index"])
            )
            if ends is None:
                continue
            for b in ends:
                if b in candidate_set:
                    cascade_buses.add(b)

        for b in congested_buses:
            congestion_count[b] += 1
        for b in voltage_buses:
            voltage_count[b] += 1
        for b in cascade_buses:
            cascade_count[b] += 1

        affected = congested_buses | voltage_buses | cascade_buses
        shed = r.severity.load_shed_mw
        if shed > 0:
            for b in affected:
                severity_sum[b] += shed

    max_severity = max(severity_sum.values()) if severity_sum else 0.0

    out: list[BusScore] = []
    for b in candidates:
        c = congestion_count.get(b, 0) / n if n else 0.0
        v = voltage_count.get(b, 0) / n if n else 0.0
        k = cascade_count.get(b, 0) / n if n else 0.0
        s = severity_sum.get(b, 0.0) / max_severity if max_severity > 0 else 0.0
        lo = min_voltage.get(b)
        hi = max_voltage.get(b)
        worst_v, worst_dir = _worst_voltage(lo, hi)
        score = (
            weights.congestion * c
            + weights.voltage * v
            + weights.cascade * k
            + weights.severity * s
        )
        out.append(
            BusScore(
                bus_idx=b,
                score=score,
                score_breakdown=ScoreBreakdown(
                    congestion=c, voltage=v, cascade=k, severity=s
                ),
                context=BusScoreContext(
                    total_scenarios=n,
                    congestion_count=congestion_count.get(b, 0),
                    voltage_count=voltage_count.get(b, 0),
                    cascade_count=cascade_count.get(b, 0),
                    worst_line=worst_line.get(b),
                    worst_line_loading_pct=(
                        worst_line_pct.get(b) if b in worst_line else None
                    ),
                    min_voltage_pu=lo,
                    max_voltage_pu=hi,
                    worst_voltage_pu=worst_v,
                    worst_voltage_dir=worst_dir,
                    severity_shed_sum_mw=severity_sum.get(b, 0.0),
                ),
            )
        )
    out.sort(key=lambda bs: (-bs.score, str(bs.bus_idx)))
    return out
