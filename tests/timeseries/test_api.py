"""API acceptance for /timeseries (U10): discovery, a short run, the error envelope.

Runs use a short horizon so the endpoint test stays quick; the 24-step budget is covered in
test_simulator. replay defaults true, so no network/LLM call happens here.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_profiles_discovery(client):
    body = client.get("/timeseries/profiles").json()
    assert "synthetic_24h" in body["profiles"]
    assert "stress_demo" in body["event_scenarios"]
    assert set(body["agent_modes"]) >= {"opf", "greedy", "llm"}


def test_geographic_scenarios_discovery(client):
    body = client.get("/timeseries/scenarios").json()
    names = {row["name"] for row in body}
    assert {"dunkelflaute", "solar_peak_south", "heatwave"} <= names
    dunkelflaute = next(row for row in body if row["name"] == "dunkelflaute")
    assert dunkelflaute["duration_hours"] == 24
    assert dunkelflaute["fallback_available"] is True


def test_run_returns_trace(client):
    r = client.post("/timeseries/run", json={
        "profile_id": "synthetic_24h", "event_scenario": "stress_demo",
        "agent_mode": "opf", "horizon_steps": 6,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["horizon_steps"] == 6
    assert len(body["steps"]) == 6
    step = body["steps"][0]
    for key in ("t", "timestamp", "baseline_state", "agent_action", "commit_status",
                "verified_state", "narration", "edge_loadings"):
        assert key in step
    assert step["commit_status"] in (
        "applied", "rejected_infeasible", "applied_unverified", "noop"
    )


def test_run_calm_profile_is_all_noop(client):
    r = client.post("/timeseries/run", json={
        "profile_id": "calm", "event_scenario": "calm", "agent_mode": "opf",
        "horizon_steps": 4, "start_scenario": "calm",
    })
    body = r.json()
    assert all(s["commit_status"] == "noop" for s in body["steps"])


def test_invalid_agent_mode_returns_400(client):
    r = client.post("/timeseries/run", json={"agent_mode": "telepathy", "horizon_steps": 2})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_agent_mode"


def test_replay_serves_frozen_trace(client):
    body = client.get("/timeseries/replay").json()
    assert body["horizon_steps"] == 24
    assert len(body["steps"]) == 24
    assert "verification" in body["steps"][0]  # structured deltas present for the UI


def test_topology_has_buses_and_edges(client):
    body = client.get("/api/topology").json()
    assert len(body["buses"]) == 118
    assert len(body["edges"]) == 186
    roles = {b["role"] for b in body["buses"]}
    assert "slack" in roles and "gen" in roles
    b0 = body["buses"][0]
    assert {"id", "x", "y", "role"} <= set(b0)


def test_index_html_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "WARDEN" in r.text


def test_agent_plan_replay(client):
    r = client.post("/agent/plan", json={"scenario": "demo_congestion", "replay": True})
    assert r.status_code == 200
    body = r.json()
    assert "plan_id" in body
    assert len(body["narration"]["sentences"]) == 3
    assert body["action"]["type"] in ("redispatch", "noop")
    assert "verified" in body["verification"]


def test_agent_plan_invalid_scenario_returns_400(client):
    r = client.post("/agent/plan", json={"scenario": "does_not_exist"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_scenario"


def test_agent_plan_demo_overload_verifies(client):
    """The headline agent-plan beat: an overloaded base is taken to a verified-secure state."""
    r = client.post("/agent/plan", json={"scenario": "demo_overload", "replay": True})
    assert r.status_code == 200
    body = r.json()
    assert body["action"]["type"] == "redispatch"
    assert body["verification"]["verified"] is True
    assert body["verification"]["deltas"]["violations_resolved"] >= 1
    # the shed-avoided figure is bounded by total grid load (no physics-impossible number)
    assert body["verification"]["deltas"]["load_shed_avoided_mw"] <= 7000


def test_compare_three_policies(client):
    r = client.post("/compare", json={"scenario": "demo_overload"})
    assert r.status_code == 200
    body = r.json()
    assert body["base_overloads"] >= 1
    kinds = {row["kind"] for row in body["rows"]}
    assert kinds == {"rule-based", "optimization", "llm-agent"}
    agent = next(row for row in body["rows"] if row["kind"] == "llm-agent")
    assert agent["explained"] is True            # only the agent explains
    assert agent["secure_after"] is True
    opf = next(row for row in body["rows"] if row["kind"] == "optimization")
    assert opf["explained"] is False             # the baseline does not


def test_compare_invalid_scenario_returns_400(client):
    r = client.post("/compare", json={"scenario": "nope"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_scenario"
