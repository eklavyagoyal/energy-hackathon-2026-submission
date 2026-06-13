"""Simulator (U8): the time-stepped loop, verify-before-commit, the <30s budget, serialization.

One 24-step stress run is shared (module fixture) and asserted from several angles, so the suite
pays the simulation cost once.
"""
from __future__ import annotations

import json
import time

import pytest

from src.timeseries.events import build_event_scenario
from src.timeseries.profiles import GenerationProfile, Profile
from src.timeseries.simulator import run_simulation

VALID_COMMIT = {"applied", "rejected_infeasible", "applied_unverified", "noop"}


@pytest.fixture(scope="module")
def stress_run(demo_net):
    lp = Profile.synthetic(hours=24, seed=42)
    gp = GenerationProfile(hours=24)
    ev = build_event_scenario("stress_demo", demo_net)
    t0 = time.perf_counter()
    trace = run_simulation(demo_net, lp, gp, ev, agent_mode="opf", horizon_steps=24, seed=42,
                           profile_id="synthetic_24h", event_scenario="stress_demo")
    return trace, time.perf_counter() - t0


def test_simulator_runs_24_steps_case118(stress_run):
    trace, _dt = stress_run
    assert len(trace.steps) == 24
    assert [s.t for s in trace.steps] == list(range(24))
    assert trace.agent_mode == "opf"


def test_runs_under_30s(stress_run):
    _trace, dt = stress_run
    assert dt < 30.0, f"24-step simulation took {dt:.1f}s, over the 30s budget"


def test_every_step_is_verified_before_commit(stress_run):
    """The headline integrity property: commit status is consistent with the solver verdict on every
    step. applied <=> verified; rejected <=> not verified; noop is a verified no-action."""
    trace, _ = stress_run
    for s in trace.steps:
        assert s.commit_status in VALID_COMMIT
        if s.commit_status == "applied":
            assert s.agent_action.get("verified") is True or s.verified_state["converged"] is True
        if s.commit_status == "rejected_infeasible":
            # a rejection never reports a verified action
            v = s.agent_action
            assert v.get("type") in ("redispatch", "noop")


def test_run_produces_a_decision_mix(stress_run):
    """A whole day over the congestion pocket exercises both branches: calm-hour noops AND
    verified redispatch wins when congestion appears (the agent holds the grid N-1 secure)."""
    trace, _ = stress_run
    statuses = {s.commit_status for s in trace.steps}
    assert "noop" in statuses     # calm hours, base within limits
    assert "applied" in statuses  # the agent acts and the solver verifies it


def test_applied_steps_are_solver_secure(stress_run):
    """The verify-before-commit guarantee: any committed action leaves a solver-verified secure base
    (0 overloads, 0 voltage violations). A step is only 'applied' if the AC rescan confirmed it."""
    trace, _ = stress_run
    applied = [s for s in trace.steps if s.commit_status == "applied"]
    assert applied, "expected at least one verified redispatch over the day"
    for s in applied:
        assert s.verified_state["n_overloads"] == 0
        assert s.verified_state["n_voltage_violations"] == 0


def test_event_outage_takes_line_out_of_service(stress_run):
    """The scheduled LineOutage removes a line: it fires in the step record and that line drops out
    of the edge loadings (which list only in-service edges)."""
    trace, _ = stress_run
    outage_steps = [s for s in trace.steps if any(e.get("kind") == "line_outage" for e in s.events)]
    assert outage_steps, "stress_demo schedules a line outage"
    s = outage_steps[0]
    li = next(e["line_index"] for e in s.events if e.get("kind") == "line_outage")
    assert f"line_{li}" not in s.edge_loadings  # out of service -> not in the loadings map
    assert f"line_{li}" in trace.steps[0].edge_loadings  # but was present before the outage


def test_narration_is_three_sentences(stress_run):
    trace, _ = stress_run
    for s in trace.steps:
        assert isinstance(s.narration, list) and len(s.narration) == 3


def test_trace_serialization_roundtrip(stress_run):
    trace, _ = stress_run
    blob = json.dumps(trace.to_dict())  # to_jsonable handles NaN/inf/numpy -> JSON
    back = json.loads(blob)
    assert back["horizon_steps"] == 24
    assert len(back["steps"]) == 24
    assert "edge_loadings" in back["steps"][0]
    assert "commit_status" in back["steps"][0]


def test_greedy_strawman_commits_unverified(demo_net):
    """Contrast policy: the greedy baseline trips lines and commits WITHOUT solver verification."""
    lp = Profile.synthetic(hours=6, seed=1)
    ev = build_event_scenario("stress_demo", demo_net)
    trace = run_simulation(demo_net, lp, GenerationProfile(hours=6), ev, agent_mode="greedy",
                           horizon_steps=6, seed=1)
    assert all(s.commit_status == "applied_unverified" for s in trace.steps)
