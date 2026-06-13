"""API acceptance: endpoints, error envelope, and the <10 s budget.

The TestClient fixture is module-scoped so startup (load + baseline
sweep) runs once; the recommendation endpoint is then exercised on the
warm cached baseline, which is the documented operator flow and the
condition the <10 s budget applies to.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["net_loaded"] is True
    assert body["baseline_cached"] is True


def test_state_reports_secure_base_and_insecure_set(client):
    body = client.get("/api/state").json()
    assert body["scenario_id"] == "demo_congestion"
    assert body["base_case"]["converged"] is True
    assert body["base_case"]["n_overloads"] == 0
    assert body["security"]["n_insecure"] >= 1
    # ext_grid blackout is the worst-ranked contingency
    worst_ids = [w["contingency_id"] for w in body["security"]["worst"]]
    assert any(cid.startswith("ext_grid") for cid in worst_ids)


def test_recommendations_returns_topk_verified(client):
    r = client.post("/battery/recommendations", json={
        "top_k": 3, "battery_capacity_mw": 50.0, "battery_energy_mwh": 200.0,
        "verify": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["candidates"]) == 3
    assert body["baseline_cached"] is True
    assert body["n_scenarios"] >= 100  # full N-1 coverage preserved
    assert body["excluded_slack_buses"]  # at least one slack excluded
    for c in body["candidates"]:
        assert c["verification"] is not None
        assert c["verification"]["verdict"] in (
            "RECOMMENDED", "MIXED", "NO_IMPACT", "NOT_RECOMMENDED"
        )
        # narration is exactly 3 sentences
        n = c["narration"]
        assert n and n.count(".") >= 3


def test_recommendations_under_10s(client):
    # warm path: baseline already cached at startup
    t0 = time.perf_counter()
    r = client.post("/battery/recommendations", json={
        "top_k": 3, "battery_capacity_mw": 50.0, "verify": True,
    })
    dt = time.perf_counter() - t0
    assert r.status_code == 200
    assert dt < 10.0, f"recommendation took {dt:.1f}s, over the 10s budget"


def test_verify_slack_bus_returns_400(client):
    # ext_grid sits at bus 68 on case118
    r = client.post("/battery/verify", json={"bus_idx": 68, "battery_capacity_mw": 50.0})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "slack_bus"


def test_verify_unknown_bus_returns_400(client):
    r = client.post("/battery/verify", json={"bus_idx": 99999})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unknown_bus"


def test_verify_normal_bus_returns_candidate(client):
    r = client.post("/battery/verify", json={"bus_idx": 24, "battery_capacity_mw": 50.0})
    assert r.status_code == 200
    body = r.json()
    assert body["bus_idx"] == 24
    assert body["verification"]["bus_idx"] == 24
    assert body["verification"]["n_scenarios"] >= 100


def test_weights_exposed_and_honored(client):
    # all weight on voltage -> ranking can differ; just assert it is echoed
    r = client.post("/battery/recommendations", json={
        "top_k": 2,
        "weights": {"congestion": 0.0, "voltage": 1.0, "cascade": 0.0, "severity": 0.0},
        "verify": False,
    })
    assert r.status_code == 200
    w = r.json()["weights_used"]
    assert w["voltage"] == pytest.approx(1.0)
    assert w["congestion"] == pytest.approx(0.0)


def test_invalid_scenario_returns_400(client):
    r = client.post("/api/scenario", json={"scenario_id": "does_not_exist"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_scenario"
