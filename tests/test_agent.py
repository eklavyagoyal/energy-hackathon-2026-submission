"""U3: the LLM agent ported onto the src engine.

Covers the plan's U3 scenarios on the deterministic (no-API-key) path: plan() returns the full
shape with exactly 3 narration sentences and never raises without a key; the state summary has the
contract keys; narration states only numbers present in the inputs (no invention).
"""
from __future__ import annotations

import pytest

from src.agent import loop as L
from src.agent.prompts import template_narration
from src.agent.state import build_grid_state_summary
from src.agent.tools import ToolContext, scan_contingencies
from src.engine.scenarios import apply_scenario
from src.grid.loader import Case118Loader


@pytest.fixture(scope="module")
def congested_net():
    """A localized-congestion net where some N-1 contingencies cascade (the agent has work to do)."""
    return apply_scenario(Case118Loader().load(), "demo_congestion")


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Force the deterministic template path: drop any key loaded from .env so no real API call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("WARDEN_AGENT_MODE", raising=False)


def test_plan_returns_full_shape_without_key(congested_net):
    result = L.plan(congested_net)
    assert set(result) >= {"plan_id", "action", "narration", "verification", "tool_trace"}
    assert isinstance(result["narration"]["sentences"], list)
    assert len(result["narration"]["sentences"]) == 3
    tools_called = [t["tool"] for t in result["tool_trace"]]
    assert tools_called[0] == "run_power_flow"
    assert "scan_contingencies" in tools_called


def test_plan_produces_a_decision(congested_net):
    """The agent either redispatches (offending found) or reports a clean noop; never crashes."""
    result = L.plan(congested_net)
    assert result["action"]["type"] in ("redispatch", "noop")
    assert "verified" in result["verification"]


def test_plan_str_scenario_replay():
    result = L.plan("demo_congestion", replay=True)
    assert len(result["narration"]["sentences"]) == 3


def test_build_grid_state_summary_contract(congested_net):
    sweep = scan_contingencies(ToolContext(net=congested_net))  # populates a sweep
    ctx = ToolContext(net=congested_net)
    scan_contingencies(ctx)
    state = build_grid_state_summary(congested_net, {"results": ctx.last_scan}, "demo_congestion")
    assert state["scenario_id"] == "demo_congestion"
    assert set(state["base_case"]) >= {"converged", "max_line_loading_pct", "total_load_mw"}
    assert set(state["security"]) == {"n_contingencies_scanned", "n_insecure", "worst"}
    assert set(state["options"]) == {"redispatchable_gens", "opf_available", "sheddable_load_mw"}
    assert state["security"]["n_contingencies_scanned"] >= 1


def test_narration_states_only_input_numbers():
    """A built state/action/report: the 3 sentences must quote numbers that appear in the inputs."""
    state = {
        "base_case": {"total_load_mw": 4242.0},
        "security": {"worst": [{"contingency_id": "line_89", "outage_name": "line 89 (bus 88 to bus 89)",
                                "severity": {"band": "HIGH", "cascade_depth": 2, "load_shed_mw": 184.0, "blackout": False}}]},
    }
    action = {"type": "redispatch", "estimated_cost_delta": -410.0,
              "changes": [{"etype": "gen", "index": 4, "field": "p_mw", "from": 450.0, "to": 340.0}]}
    report = {"verified": True, "contingency_ids": ["line_89"],
              "after": [{"contingency_id": "base", "status": "SECURE"}, {"contingency_id": "line_89", "status": "VIOLATIONS"}],
              "deltas": {"violations_resolved": 2, "load_shed_avoided_mw": 184.0, "worst_score_after": 43.4}}
    sentences = template_narration(state, action, report)
    assert len(sentences) == 3
    blob = " ".join(sentences)
    # numbers/ids drawn from the inputs (load_shed_avoided_mw is rounded to whole MW for display)
    assert "184" in blob and "line_89" in blob
