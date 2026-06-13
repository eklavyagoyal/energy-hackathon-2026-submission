"""Recommendation orchestrator: scoring -> ranking -> verification.

Pipeline (docs/battery-feature.md):

  1. build the contingency set and run the baseline N-1 sweep ONCE
     (src/engine/scan.run_contingency_sweep). This single sweep is the
     baseline cache: it feeds BOTH the scoring aggregation and every
     candidate's verification delta, so the expensive physics runs
     once per request, not once per candidate.
  2. score every non-slack bus (src/battery/scoring.score_buses).
  3. take the top-K candidates (K capped by BATTERY_MAX_TOPK).
  4. VERIFY each candidate by re-running the full sweep with a virtual
     battery at its bus (src/battery/verification). The top-K
     verifications are independent full sweeps and run in parallel when
     BATTERY_VERIFICATION_PARALLEL is set (mandatory for large grids,
     harmless overkill for case118).
  5. narrate each verified candidate (src/battery/narration), LLM with
     deterministic fallback.

Baseline-cache note (logged as D52): Phase 1 ships no shared result
cache on this branch, so the recommender owns the cache for the
duration of one request (the baseline sweep computed in step 1 and
reused in steps 2 and 4). A process-wide cache keyed by net revision
is the natural next step once the scenario endpoint and live net are
wired; the API layer holds the single live net and can supply a
pre-computed baseline to recommend_battery_locations to skip the
recompute entirely.

Parallelism note (logged as D53): the top-K verifications run on a
ThreadPoolExecutor. Each task deep-copies the net first, so there is
no shared mutable state; pandapower releases the GIL inside its
numba/scipy Newton-Raphson kernels, so threads give real overlap
without the pickling overhead and "pandapower + multiprocessing"
debugging tarpit flagged in docs/03-phase1-engine.md. The baseline
sweep in step 1 also warms numba's JIT before any thread starts, which
sidesteps first-call compilation races. Swap to ProcessPoolExecutor
here if a future profile shows the GIL dominating on very large grids;
the task function is already a pure function of picklable inputs.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from src.battery.narration import build_findings, narrate
from src.battery.schemas import (
    BatteryCandidate,
    RecommendationRequest,
    RecommendationResponse,
    VerificationResult,
)
from src.battery.scoring import score_buses
from src.battery.verification import impact_contingency_ids, verify_battery_candidate
from src.config import Settings
from src.engine.network import native_index, opf_available, slack_bus_set
from src.engine.scan import build_contingency_set, run_contingency_sweep

logger = logging.getLogger(__name__)

DISPATCH_MODEL = "fixed_full_discharge_heuristic"


def _bus_name(net, bus_idx) -> str | None:
    if "name" in net.bus.columns and bus_idx in net.bus.index:
        name = net.bus.at[bus_idx, "name"]
        if name is not None and str(name) != "nan":
            return str(name)
    return None


def _verify_all(
    net,
    candidates,
    contingencies,
    baseline,
    battery_p_mw,
    battery_max_e_mwh,
    parallel,
    restrict_to,
) -> dict:
    """Verify each candidate bus; returns {bus_idx: VerificationResult}.
    Runs the K independent sweeps in parallel when requested."""

    def task(bus_idx):
        return verify_battery_candidate(
            net,
            bus_idx,
            contingencies=contingencies,
            baseline=baseline,
            battery_p_mw=battery_p_mw,
            battery_max_e_mwh=battery_max_e_mwh,
            restrict_to=restrict_to,
        )

    bus_ids = [bs.bus_idx for bs in candidates]
    results: dict = {}
    if parallel and len(bus_ids) > 1:
        with ThreadPoolExecutor(max_workers=len(bus_ids)) as pool:
            for bus_idx, vr in zip(bus_ids, pool.map(task, bus_ids)):
                results[bus_idx] = vr
    else:
        for bus_idx in bus_ids:
            results[bus_idx] = task(bus_idx)
    return results


def recommend_battery_locations(
    net,
    request: RecommendationRequest,
    settings: Settings | None = None,
    baseline: list | None = None,
    contingencies: list | None = None,
) -> RecommendationResponse:
    """Full recommendation for the given net and request.

    `baseline` and `contingencies` may be supplied by the caller (the
    API layer caches them against the live net); when omitted they are
    computed here and `baseline_cached` is reported False.
    """
    settings = settings or Settings.from_env()
    t0 = time.perf_counter()

    # Time-aware siting (U16): for a horizon, evaluate at the PEAK load level (where storage value is
    # highest), not the current snapshot. horizon_steps=1 keeps the snapshot path byte-for-byte.
    peak_load_factor = 1.0
    horizon_steps = getattr(request, "horizon_steps", 1)
    if horizon_steps > 1:
        from src.engine.network import working_copy
        from src.timeseries.profiles import Profile

        peak_load_factor = max(Profile.synthetic(hours=horizon_steps).multipliers)
        net = working_copy(net)
        if len(net.load):
            net.load["p_mw"] = net.load["p_mw"] * peak_load_factor
            net.load["q_mvar"] = net.load["q_mvar"] * peak_load_factor
        # The caller's cached baseline is for the unscaled net; it no longer applies.
        baseline, contingencies = None, None
        logger.info("time-aware siting: horizon %d steps, peak load factor %.3f",
                    horizon_steps, peak_load_factor)

    if contingencies is None:
        contingencies = build_contingency_set(net)

    baseline_cached = baseline is not None
    if baseline is None:
        baseline = run_contingency_sweep(net, contingencies)

    opf_ok = opf_available(net)
    if not opf_ok:
        # Honesty / portability rule 8: real TSO exports often lack
        # poly_cost. The battery dispatch is the fixed-discharge
        # heuristic regardless, so this is a logged note, not a crash.
        logger.warning(
            "poly_cost/controllable not available on this net; battery "
            "dispatch uses the %s (no OPF). This is expected on datasets "
            "without generator cost data.",
            DISPATCH_MODEL,
        )

    weights = request.weights.normalized()
    scores = score_buses(net, baseline, weights)

    top_k = min(request.top_k, settings.battery_max_topk)
    top = scores[:top_k]

    verifications: dict = {}
    if request.verify and top:
        # Bound each verification re-solve to the contingencies a battery
        # can physically affect; the rest carry forward from the baseline
        # unchanged. Full-set coverage is preserved (n_scenarios counts
        # all baseline contingencies); only provably-unchanged solves are
        # skipped, which is what keeps the per-candidate sweep fast.
        restrict_to = impact_contingency_ids(baseline)
        verifications = _verify_all(
            net,
            top,
            contingencies,
            baseline,
            request.battery_capacity_mw,
            request.battery_energy_mwh,
            settings.battery_verification_parallel,
            restrict_to,
        )

    total_buses = len(scores)
    candidates: list[BatteryCandidate] = []
    for rank_pos, bs in enumerate(top, start=1):
        verification: VerificationResult | None = verifications.get(bs.bus_idx)
        findings = build_findings(rank_pos, total_buses, bs, verification)
        narration, source = narrate(findings, settings)
        candidates.append(
            BatteryCandidate(
                bus_idx=bs.bus_idx,
                bus_name=_bus_name(net, bs.bus_idx),
                score=bs.score,
                score_breakdown=bs.score_breakdown,
                context=bs.context,
                verification=verification,
                narration=narration,
                narration_source=source,
            )
        )

    excluded = sorted(slack_bus_set(net), key=str)
    return RecommendationResponse(
        candidates=candidates,
        computation_time_ms=(time.perf_counter() - t0) * 1000.0,
        baseline_cached=baseline_cached,
        n_scenarios=len(baseline),
        n_buses_scored=total_buses,
        excluded_slack_buses=excluded,
        weights_used=weights,
        opf_available=opf_ok,
        dispatch_model=DISPATCH_MODEL,
        horizon_steps=horizon_steps,
        peak_load_factor=round(peak_load_factor, 4),
    )
