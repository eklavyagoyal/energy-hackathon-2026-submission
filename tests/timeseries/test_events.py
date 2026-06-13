"""Events (U7): firing, auto-restore, load spikes, maintenance windows, prepared scenarios."""
from __future__ import annotations

import pandapower as pp
import pytest

from src.timeseries.events import (
    EventStream,
    LineOutage,
    LoadSpike,
    Maintenance,
    build_event_scenario,
)


def _net():
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, b0)
    pp.create_line_from_parameters(net, b0, b1, length_km=1.0, r_ohm_per_km=0.1,
                                   x_ohm_per_km=0.3, c_nf_per_km=0.0, max_i_ka=1.0)
    pp.create_load(net, b1, p_mw=100.0, q_mvar=20.0)
    return net


def test_line_outage_fires_then_auto_restores():
    net = _net()
    stream = EventStream({1: [LineOutage(0, duration_steps=2)]})
    assert stream.events_at(0) == []
    fired = stream.events_at(1)
    assert len(fired) == 1 and fired[0].kind == "line_outage"
    fired[0].apply(net)
    assert net.line.at[0, "in_service"] is False or not net.line.at[0, "in_service"]
    stream.cleanup(1, net)  # t+1=2 < expires 3: still out
    assert not net.line.at[0, "in_service"]
    stream.cleanup(2, net)  # t+1=3 >= expires 3: restored
    assert net.line.at[0, "in_service"]


def test_load_spike_scales_then_restores():
    net = _net()
    spike = LoadSpike(1, factor=1.5, duration_steps=1)  # bus 1 carries the load
    spike.apply(net)
    assert net.load.at[0, "p_mw"] == pytest.approx(150.0)
    spike.restore(net)
    assert net.load.at[0, "p_mw"] == pytest.approx(100.0)


def test_maintenance_window_active_range():
    m = Maintenance(0, start=3, end=6)
    assert not m.active_at(2)
    assert m.active_at(3)
    assert m.active_at(5)
    assert not m.active_at(6)


def test_event_to_dict_is_json_safe():
    for ev in (LineOutage(0, 2), LoadSpike(1, 1.4, 3), Maintenance(0, 1, 4)):
        d = ev.to_dict()
        assert "kind" in d
        assert isinstance(d["kind"], str)


def test_build_scenarios(case118_net):
    calm = build_event_scenario("calm", case118_net)
    assert calm.schedule == {}
    stress = build_event_scenario("stress_demo", case118_net)
    assert stress.schedule  # has a known cascade-risk window
    kinds = {ev.kind for evs in stress.schedule.values() for ev in evs}
    assert "line_outage" in kinds
    with pytest.raises(KeyError):
        build_event_scenario("nonsense", case118_net)


def test_unknown_line_index_is_safe_noop():
    """An event referencing an index absent on this dataset must not raise (portability)."""
    net = _net()
    LineOutage(9999, 1).apply(net)  # no such line; silently does nothing
    assert net.line.at[0, "in_service"]
