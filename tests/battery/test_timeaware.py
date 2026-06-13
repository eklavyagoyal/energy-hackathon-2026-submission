"""Time-aware battery siting (U16): horizon_steps=1 is the unchanged snapshot; >1 sites at peak."""
from __future__ import annotations

from src.battery.recommender import recommend_battery_locations
from src.battery.schemas import RecommendationRequest


def test_horizon_1_is_unchanged_snapshot(demo_net, demo_baseline):
    cont, base = demo_baseline
    r = recommend_battery_locations(
        demo_net, RecommendationRequest(top_k=2, verify=False, horizon_steps=1),
        baseline=base, contingencies=cont,
    )
    assert r.horizon_steps == 1
    assert r.peak_load_factor == 1.0
    assert r.baseline_cached is True  # the passed snapshot baseline is used as-is


def test_default_request_omits_horizon(demo_net, demo_baseline):
    cont, base = demo_baseline
    r = recommend_battery_locations(
        demo_net, RecommendationRequest(top_k=2, verify=False),
        baseline=base, contingencies=cont,
    )
    assert r.horizon_steps == 1 and r.peak_load_factor == 1.0


def test_horizon_peak_scales_and_recomputes(demo_net):
    r = recommend_battery_locations(
        demo_net, RecommendationRequest(top_k=2, verify=False, horizon_steps=24),
    )
    assert r.horizon_steps == 24
    assert r.peak_load_factor > 1.0          # synthetic daily curve peaks above unity
    assert r.baseline_cached is False        # snapshot baseline discarded, recomputed at peak
    assert len(r.candidates) == 2
