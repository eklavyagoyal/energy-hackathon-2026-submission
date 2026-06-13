"""Freeze the battery demo scenario into fixtures and print a demo report.

Run: python scripts/freeze_battery_demo.py   (from an activated .venv; see README)

Produces (under fixtures/):
  battery_state.json            - /api/state for the demo scenario
  battery_recommendations.json  - /battery/recommendations response (top-3)
  battery_counterfactual.json   - the headline cascade WITH vs WITHOUT the
                                  battery, for the split-screen demo beat

Everything is regenerated from the live engine, so the frozen numbers are
real (solver-produced), never illustrative. Deterministic: same scenario
in, same fixtures out.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from src.api.state import AppState  # noqa: E402
from src.battery.recommender import recommend_battery_locations  # noqa: E402
from src.battery.schemas import RecommendationRequest  # noqa: E402
from src.battery.verification import add_virtual_battery  # noqa: E402
from src.config import Settings  # noqa: E402
from src.engine.scan import analyze_contingency  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
DEMO_BATTERY_MW = 50.0
DEMO_BATTERY_MWH = 200.0


def _outage_by_id(contingencies, cid):
    return next(o for o in contingencies if o.contingency_id == cid)


def _contingency_view(result) -> dict:
    """Compact view of one contingency outcome for the counterfactual."""
    return {
        "contingency_id": result.contingency_id,
        "status": result.status,
        "cascade_depth": result.severity.cascade_depth,
        "load_shed_mw": round(result.severity.load_shed_mw, 1),
        "lines_tripped": [
            t["index"] for t in result.tripped_elements if t["etype"] == "line"
        ],
        "n_elements_tripped": len(result.tripped_elements),
        "max_loading_pct": (
            round(result.final_state["max_loading_pct"], 1)
            if result.final_state.get("max_loading_pct") is not None
            else None
        ),
        "cascade_trace": [
            {
                "iter": s.iter,
                "tripped": s.tripped,
                "load_shed_mw": round(s.load_shed_mw, 1),
            }
            for s in result.cascade_trace
        ],
    }


def main() -> None:
    FIXTURES.mkdir(exist_ok=True)
    settings = Settings.from_env()

    print("loading demo scenario and caching baseline ...")
    t0 = time.perf_counter()
    grid = AppState(settings)
    print(f"  baseline: {len(grid.baseline)} contingencies in {grid.baseline_ms:.0f} ms")

    # 1. state fixture
    state = grid.state_summary()
    (FIXTURES / "battery_state.json").write_text(json.dumps(state, indent=2, default=str))

    # 2. recommendation fixture (the warm path the API serves)
    req = RecommendationRequest(
        top_k=3, battery_capacity_mw=DEMO_BATTERY_MW,
        battery_energy_mwh=DEMO_BATTERY_MWH, verify=True,
    )
    t1 = time.perf_counter()
    resp = recommend_battery_locations(
        grid.net, req, settings, baseline=grid.baseline, contingencies=grid.contingencies
    )
    rec_ms = (time.perf_counter() - t1) * 1000.0
    (FIXTURES / "battery_recommendations.json").write_text(
        resp.model_dump_json(indent=2)
    )
    print(f"  recommendation (warm): {rec_ms:.0f} ms, {len(resp.candidates)} candidates")

    # 3. counterfactual: pick the headline = the candidate+contingency with the
    #    largest cascade->stable load-shed avoided.
    headline = None  # (candidate, per_contingency_delta)
    for cand in resp.candidates:
        if not cand.verification:
            continue
        for pc in cand.verification.per_contingency:
            if pc.status_before in ("CASCADE", "DIVERGED") and pc.status_after in (
                "SECURE", "VIOLATIONS"
            ):
                avoided = pc.load_shed_before_mw - pc.load_shed_after_mw
                if headline is None or avoided > headline[2]:
                    headline = (cand, pc, avoided)

    counterfactual = None
    if headline is not None:
        cand, pc, avoided = headline
        outage = _outage_by_id(grid.contingencies, pc.contingency_id)
        without = next(r for r in grid.baseline if r.contingency_id == pc.contingency_id)
        net_batt, _ = add_virtual_battery(
            grid.net, cand.bus_idx, DEMO_BATTERY_MW, DEMO_BATTERY_MWH
        )
        with_batt = analyze_contingency(net_batt, outage)
        counterfactual = {
            "recommended_bus": cand.bus_idx,
            "battery_mw": DEMO_BATTERY_MW,
            "battery_mwh": DEMO_BATTERY_MWH,
            "contingency": {
                "id": outage.contingency_id,
                "name": outage.name,
            },
            "without_battery": _contingency_view(without),
            "with_battery": _contingency_view(with_batt),
            "load_shed_avoided_mw": round(avoided, 1),
        }
        (FIXTURES / "battery_counterfactual.json").write_text(
            json.dumps(counterfactual, indent=2, default=str)
        )

    print(f"  total freeze wall: {time.perf_counter() - t0:.1f} s")
    print("\n=== DEMO REPORT ===")
    print(f"scenario: {state['scenario_id']} | base secure: "
          f"{state['base_case']['converged'] and state['base_case']['n_overloads'] == 0}"
          f" | insecure contingencies: {state['security']['n_insecure']}")
    print("top-3 recommended buses (solver-verified):")
    for c in resp.candidates:
        v = c.verification
        print(f"  bus {c.bus_idx}: score {c.score:.3f} | {v.verdict} | "
              f"prevents {v.cascades_prevented} cascades, saves "
              f"{v.mw_load_shed_avoided:.0f} MW, worsens {v.scenarios_worsened}")
    if counterfactual:
        wo = counterfactual["without_battery"]
        wb = counterfactual["with_battery"]
        print(f"\nheadline counterfactual: bus {counterfactual['recommended_bus']}, "
              f"{DEMO_BATTERY_MW:.0f} MW battery vs loss of {counterfactual['contingency']['name']}")
        print(f"  WITHOUT: {wo['status']}, {wo['n_elements_tripped']} elements trip, "
              f"{wo['load_shed_mw']:.0f} MW shed")
        print(f"  WITH:    {wb['status']}, {wb['n_elements_tripped']} elements trip, "
              f"{wb['load_shed_mw']:.0f} MW shed")
        print(f"  load shedding avoided: {counterfactual['load_shed_avoided_mw']:.0f} MW")
    else:
        print("\nno clean cascade->stable headline found at this battery size; "
              "increase DEMO_BATTERY_MW or revisit the scenario")
    print(f"\nfixtures written to {FIXTURES}/")


if __name__ == "__main__":
    main()
