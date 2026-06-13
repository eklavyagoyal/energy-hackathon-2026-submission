"""Scoring math: known synthetic contingency results -> known scores.

Network: radial line 0-1-2-3, ext_grid at bus 0.
  line 0 = (0,1), line 1 = (1,2), line 2 = (2,3)
  candidate buses = {1, 2, 3} (bus 0 is slack, excluded)

Four synthetic scenarios with hand-chosen post-cascade states; every
expected fraction below is computed by hand in the comments so a
regression in the aggregation is caught immediately.
"""

from __future__ import annotations

import math

import pytest

from src.battery.schemas import ScoreWeights
from src.battery.scoring import candidate_buses, score_buses
from tests.battery.factories import line_net, make_result


@pytest.fixture
def net():
    return line_net(4)


@pytest.fixture
def results():
    return [
        # s1: line 0 congested, bus 1 mildly low (in band)
        make_result("c1", line_loading={0: 85.0, 1: 50.0, 2: 10.0},
                    bus_vm={0: 1.0, 1: 0.96, 2: 0.99, 3: 1.0}),
        # s2: lines 1,2 congested, bus 1 undervoltage
        make_result("c2", line_loading={0: 50.0, 1: 90.0, 2: 90.0},
                    bus_vm={0: 1.0, 1: 0.90, 2: 0.99, 3: 1.0}),
        # s3: full blackout, 400 MW shed (affects every candidate bus)
        make_result("c3", status="FULL_BLACKOUT", blackout=True,
                    load_shed_mw=400.0, load_shed_pct=100.0, score=2000.0),
        # s4: line 2 congested + trips, bus 3 overvoltage, 20 MW shed
        make_result("c4", line_loading={0: 10.0, 1: 10.0, 2: 85.0},
                    bus_vm={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.07},
                    cascade_depth=1, load_shed_mw=20.0,
                    tripped=[{"etype": "line", "index": 2, "loading_pct": 130.0}]),
    ]


def _by_bus(scores):
    return {bs.bus_idx: bs for bs in scores}


def test_slack_bus_excluded_from_candidates(net):
    candidates, excluded = candidate_buses(net)
    assert set(candidates) == {1, 2, 3}
    assert excluded == [0]


def test_congestion_scores(net, results):
    s = _by_bus(score_buses(net, results))
    # bus1 lines {0,1}: s1(85),s2(90) -> 2/4 ; bus2 lines {1,2}: s2,s4 -> 2/4 ;
    # bus3 line {2}: s2(90),s4(85) -> 2/4
    assert s[1].score_breakdown.congestion == pytest.approx(0.5)
    assert s[2].score_breakdown.congestion == pytest.approx(0.5)
    assert s[3].score_breakdown.congestion == pytest.approx(0.5)


def test_voltage_scores(net, results):
    s = _by_bus(score_buses(net, results))
    # bus1: s2 (0.90) -> 1/4 ; bus2: none -> 0 ; bus3: s4 (1.07) -> 1/4
    assert s[1].score_breakdown.voltage == pytest.approx(0.25)
    assert s[2].score_breakdown.voltage == pytest.approx(0.0)
    assert s[3].score_breakdown.voltage == pytest.approx(0.25)


def test_cascade_scores(net, results):
    s = _by_bus(score_buses(net, results))
    # blackout s3 marks every candidate; s4 trips line2 (endpoints 2,3).
    # bus1: {s3} -> 1/4 ; bus2: {s3,s4} -> 2/4 ; bus3: {s3,s4} -> 2/4
    assert s[1].score_breakdown.cascade == pytest.approx(0.25)
    assert s[2].score_breakdown.cascade == pytest.approx(0.5)
    assert s[3].score_breakdown.cascade == pytest.approx(0.5)


def test_severity_weight_normalized(net, results):
    s = _by_bus(score_buses(net, results))
    # severity_sum: blackout adds 400 to all; s4 adds 20 to affected {2,3}.
    # bus1=400, bus2=420, bus3=420 -> max 420 -> bus2,bus3=1.0, bus1=400/420
    assert s[2].score_breakdown.severity == pytest.approx(1.0)
    assert s[3].score_breakdown.severity == pytest.approx(1.0)
    assert s[1].score_breakdown.severity == pytest.approx(400.0 / 420.0)


def test_final_score_weighted_sum_and_ranking(net, results):
    scores = score_buses(net, results)
    s = _by_bus(scores)
    w = ScoreWeights()  # 0.35 / 0.25 / 0.25 / 0.15, sums to 1
    for b in (1, 2, 3):
        bd = s[b].score_breakdown
        expected = (
            w.congestion * bd.congestion + w.voltage * bd.voltage
            + w.cascade * bd.cascade + w.severity * bd.severity
        )
        assert s[b].score == pytest.approx(expected)
        assert 0.0 <= s[b].score <= 1.0
    # bus3 (0.5125) > bus2 (0.45) > bus1 (~0.4429)
    assert [bs.bus_idx for bs in scores] == [3, 2, 1]


def test_context_reports_worst_line_and_voltage_direction(net, results):
    s = _by_bus(score_buses(net, results))
    # bus3's worst connected line loading is 90 (line 2 in s2)
    assert s[3].context.worst_line == 2
    assert s[3].context.worst_line_loading_pct == pytest.approx(90.0)
    # bus3 saw overvoltage 1.07 -> direction 'high'
    assert s[3].context.worst_voltage_dir == "high"
    assert s[3].context.worst_voltage_pu == pytest.approx(1.07)
    # bus1 saw undervoltage 0.90 -> direction 'low'
    assert s[1].context.worst_voltage_dir == "low"
    assert s[1].context.worst_voltage_pu == pytest.approx(0.90)


def test_weights_normalized_when_not_summing_to_one(net, results):
    # Weights are normalized internally; doubling all weights changes nothing.
    a = score_buses(net, results, ScoreWeights(congestion=0.35, voltage=0.25, cascade=0.25, severity=0.15))
    b = score_buses(net, results, ScoreWeights(congestion=0.70, voltage=0.50, cascade=0.50, severity=0.30))
    sa, sb = _by_bus(a), _by_bus(b)
    for bus in (1, 2, 3):
        assert sa[bus].score == pytest.approx(sb[bus].score)


def test_extreme_weights_keep_score_in_unit_interval(net, results):
    scores = score_buses(net, results, ScoreWeights(congestion=1.0, voltage=0.0, cascade=0.0, severity=0.0))
    for bs in scores:
        assert 0.0 <= bs.score <= 1.0
        # pure-congestion weighting -> score equals congestion component
        assert bs.score == pytest.approx(bs.score_breakdown.congestion)
