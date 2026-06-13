"""Verification delta math and verdict classification.

Pure-function tests: compute_verification_deltas and classify never
touch a solver. The solver-backed path (verify_battery_candidate on a
real net) is exercised in test_dataset_portability and test_api.
"""

from __future__ import annotations

import pytest

from src.battery.verification import (
    classify,
    compute_verification_deltas,
    impact_contingency_ids,
    resolve_bus,
)
from tests.battery.factories import line_net, make_result


@pytest.fixture
def baseline():
    return [
        make_result("c1", status="CASCADE", cascade_depth=2, score=600.0,
                    load_shed_mw=300.0, line_loading={0: 130.0}, bus_vm={1: 0.93}),
        make_result("c2", status="VIOLATIONS", residual_violations=1, score=5.0,
                    line_loading={0: 105.0}, bus_vm={1: 0.96}),
        make_result("c3", status="SECURE", score=0.0,
                    line_loading={0: 50.0}, bus_vm={1: 0.96}),
    ]


@pytest.fixture
def with_battery():
    return [
        # c1: cascade resolved
        make_result("c1", status="SECURE", score=0.0,
                    line_loading={0: 95.0}, bus_vm={1: 0.97}),
        # c2: unchanged
        make_result("c2", status="VIOLATIONS", residual_violations=1, score=5.0,
                    line_loading={0: 104.0}, bus_vm={1: 0.96}),
        # c3: worsened (battery overloads a previously secure case)
        make_result("c3", status="VIOLATIONS", residual_violations=1, score=8.0,
                    line_loading={0: 110.0}, bus_vm={1: 0.94}),
    ]


def test_delta_counts(baseline, with_battery):
    d = compute_verification_deltas(baseline, with_battery)
    assert d["n_scenarios"] == 3
    assert d["scenarios_improved"] == 1   # c1
    assert d["scenarios_worsened"] == 1   # c3
    assert d["scenarios_unchanged"] == 1  # c2
    assert d["cascades_prevented"] == 1   # c1 CASCADE -> SECURE


def test_delta_mw_and_physical_metrics(baseline, with_battery):
    d = compute_verification_deltas(baseline, with_battery)
    assert d["mw_load_shed_avoided"] == pytest.approx(300.0)
    # max single-line loading reduction is c1: 130 -> 95 = 35
    assert d["max_loading_reduction"] == pytest.approx(35.0)
    # voltage deltas (after - before) over bus1: c1 +0.04, c2 0, c3 -0.02
    # -> mean +0.02/3 (battery helps the cascade bus more than it dips the
    # worsened one)
    assert d["avg_voltage_improvement"] == pytest.approx(0.02 / 3)


def test_per_contingency_only_lists_changed(baseline, with_battery):
    d = compute_verification_deltas(baseline, with_battery)
    changed = {pc.contingency_id for pc in d["per_contingency"]}
    assert changed == {"c1", "c3"}  # c2 unchanged is omitted
    c1 = next(pc for pc in d["per_contingency"] if pc.contingency_id == "c1")
    assert c1.status_before == "CASCADE" and c1.status_after == "SECURE"
    assert c1.load_shed_before_mw == pytest.approx(300.0)
    assert c1.load_shed_after_mw == pytest.approx(0.0)


def test_classify_bands():
    # clean win
    assert classify(3, 0, 0, 0.0) == "RECOMMENDED"
    # net security gain despite minor worsening
    assert classify(5, 2, 2, 1200.0) == "RECOMMENDED"
    # nothing changed
    assert classify(0, 0, 0, 0.0) == "NO_IMPACT"
    # clearly harmful
    assert classify(1, 5, 0, -300.0) == "NOT_RECOMMENDED"
    # helps and harms with no clear net direction
    assert classify(2, 3, 0, 50.0) == "MIXED"


def test_classify_full_recommendation_from_deltas(baseline, with_battery):
    d = compute_verification_deltas(baseline, with_battery)
    verdict = classify(
        d["scenarios_improved"], d["scenarios_worsened"],
        d["cascades_prevented"], d["mw_load_shed_avoided"],
    )
    # improved 1 == worsened 1, but prevents a cascade and saves 300 MW net
    assert verdict == "RECOMMENDED"


def test_impact_set_excludes_only_comfortably_secure():
    baseline = [
        make_result("casc", status="CASCADE", cascade_depth=1, score=500.0,
                    load_shed_mw=10.0, line_loading={0: 130.0}),
        make_result("near", status="VIOLATIONS", line_loading={0: 95.0}),
        make_result("secure", status="SECURE", line_loading={0: 50.0}),
    ]
    impact = impact_contingency_ids(baseline)
    assert impact == {"casc", "near"}  # 'secure' (max 50) is excluded


def test_resolve_bus_accepts_native_and_string_int():
    net = line_net(4)  # int indices 0..3
    assert resolve_bus(net, 2) == 2
    assert resolve_bus(net, "2") == 2  # convenience string-of-int
    from src.battery.verification import UnknownBusError
    with pytest.raises(UnknownBusError):
        resolve_bus(net, 999)
