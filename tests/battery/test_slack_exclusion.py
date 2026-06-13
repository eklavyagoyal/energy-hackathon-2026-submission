"""Slack-bus exclusion: generalized to the ext_grid.bus set, enforced
before any solver call.

Acceptance rules:
- verifying any bus in net.ext_grid.bus.values raises ValueError BEFORE
  a single power flow runs (a monkeypatched runpp counter must read 0)
- the exclusion is set membership over ALL ext_grids, not equality to a
  hardcoded bus 0, so multi-slack nets exclude every slack bus
"""

from __future__ import annotations

import pandapower as pp
import pytest

from src.battery.scoring import candidate_buses
from src.battery.verification import (
    SlackBusError,
    assert_not_slack,
    verify_battery_candidate,
)
from src.engine.network import slack_bus_set
from tests.battery.factories import line_net, make_result


def two_slack_net():
    """4-bus line with ext_grids at BOTH ends (buses 0 and 3)."""
    net = pp.create_empty_network()
    for i in range(4):
        pp.create_bus(net, vn_kv=110.0, index=i)
    pp.create_ext_grid(net, bus=0, vm_pu=1.0)
    pp.create_ext_grid(net, bus=3, vm_pu=1.0)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        pp.create_line_from_parameters(
            net, a, b, length_km=10, r_ohm_per_km=0.1, x_ohm_per_km=0.3,
            c_nf_per_km=10, max_i_ka=1.0,
        )
    pp.create_load(net, bus=1, p_mw=10, q_mvar=2)
    pp.create_load(net, bus=2, p_mw=10, q_mvar=2)
    return net


def test_slack_set_is_a_set_of_all_ext_grids():
    net = two_slack_net()
    assert slack_bus_set(net) == {0, 3}


def test_both_slacks_excluded_from_candidates():
    net = two_slack_net()
    candidates, excluded = candidate_buses(net)
    assert set(candidates) == {1, 2}
    assert excluded == [0, 3]
    assert 0 not in candidates and 3 not in candidates


def test_verify_slack_raises_valueerror_before_any_solve(monkeypatch):
    net = line_net(4)  # ext_grid at bus 0
    calls = {"runpp": 0}
    orig = pp.runpp

    def counting_runpp(*args, **kwargs):
        calls["runpp"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(pp, "runpp", counting_runpp)

    with pytest.raises(ValueError):  # SlackBusError is a ValueError
        verify_battery_candidate(
            net, 0, contingencies=[], baseline=[], battery_p_mw=10.0,
        )
    assert calls["runpp"] == 0  # the slack guard never reached the solver


def test_verify_each_slack_in_multislack_net_raises(monkeypatch):
    net = two_slack_net()
    calls = {"runpp": 0}
    orig = pp.runpp
    monkeypatch.setattr(
        pp, "runpp",
        lambda *a, **k: (calls.__setitem__("runpp", calls["runpp"] + 1), orig(*a, **k))[1],
    )
    for slack in (0, 3):
        with pytest.raises(ValueError):
            verify_battery_candidate(net, slack, contingencies=[], baseline=[])
    assert calls["runpp"] == 0


def test_assert_not_slack_passes_for_normal_bus():
    net = two_slack_net()
    # non-slack buses do not raise
    assert_not_slack(net, 1)
    assert_not_slack(net, 2)


def test_slack_bus_never_in_scored_output():
    """Even with synthetic results that reference the slack bus, scoring
    never emits a candidate for it."""
    from src.battery.scoring import score_buses

    net = two_slack_net()
    results = [
        make_result("c1", line_loading={0: 95.0, 1: 95.0, 2: 95.0},
                    bus_vm={0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9}),
    ]
    scored = score_buses(net, results)
    scored_buses = {bs.bus_idx for bs in scored}
    assert scored_buses == {1, 2}
    assert 0 not in scored_buses and 3 not in scored_buses
