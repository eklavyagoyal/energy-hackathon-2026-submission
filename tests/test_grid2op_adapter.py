"""Grid2Op adapter + L2RPN runner (U14, U15): isolation and the duck-typed mapping.

grid2op is intentionally absent from this environment (it pins pandapower < 3, KTD7), so these
tests assert the isolation holds and exercise the mapping against a FAKE observation, with no
grid2op installed.
"""
from __future__ import annotations

import pytest

from src.integrations.grid2op_adapter import (
    agent_action_to_grid2op,
    grid2op_available,
    observation_to_grid_state,
)
from src.integrations.l2rpn_runner import run_l2rpn_episode


class FakeObs:
    """Minimal duck-typed Grid2Op observation."""
    def __init__(self):
        self.rho = [0.45, 1.20, 0.0, 0.80]      # ratios; index 1 is overloaded, index 2 is out
        self.line_status = [True, True, False, True]
        self.load_p = [100.0, 60.0]
        self.gen_p = [180.0, 40.0]
        self.current_step = 7


def test_grid2op_isolated_in_main_env():
    """The whole point of KTD7: the module imports and reports availability without grid2op."""
    assert grid2op_available() is False


def test_rho_maps_to_loading_percent():
    state = observation_to_grid_state(FakeObs())
    # rho * 100, out-of-service line excluded from the loadings map
    assert state["line_loadings"]["line_0"] == pytest.approx(45.0)
    assert state["line_loadings"]["line_1"] == pytest.approx(120.0)
    assert "line_2" not in state["line_loadings"]
    assert state["lines_out_of_service"] == [2]
    assert state["n_overloads"] == 1  # only line_1 over 100%
    assert state["max_loading_pct"] == pytest.approx(120.0)
    assert state["total_load_mw"] == pytest.approx(160.0)
    assert state["time_step"] == 7
    assert state["source"] == "grid2op"


def test_action_maps_to_redispatch_deltas():
    captured = {}

    def fake_action_space(d):
        captured["d"] = d
        return d

    action = {"changes": [
        {"etype": "gen", "index": 3, "field": "p_mw", "from": 100.0, "to": 130.0},   # +30
        {"etype": "gen", "index": 5, "field": "p_mw", "from": 80.0, "to": 60.0},     # -20
        {"etype": "gen", "index": 2, "field": "vm_pu", "from": 1.0, "to": 1.02},     # dropped (vm)
    ]}
    agent_action_to_grid2op(action, fake_action_space)
    deltas = dict(captured["d"]["redispatch"])
    assert deltas[3] == pytest.approx(30.0)
    assert deltas[5] == pytest.approx(-20.0)
    assert 2 not in deltas  # voltage setpoints have no redispatch equivalent
    assert all(isinstance(k, int) for k in deltas)  # integer gen ids (KTD7)


def test_noop_action_maps_to_donothing():
    sent = {}
    agent_action_to_grid2op({"changes": []}, lambda d: sent.update({"d": d}))
    assert sent["d"] == {}  # empty -> do-nothing


def test_runner_raises_actionable_error_without_grid2op():
    with pytest.raises(RuntimeError, match="separate venv|SEPARATE venv|grid2op"):
        run_l2rpn_episode()
