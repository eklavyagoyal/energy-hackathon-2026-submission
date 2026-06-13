"""Profiles (U6): determinism, clamping, and the no-drift invariant of apply_profile."""
from __future__ import annotations

import pandapower as pp
import pytest

from src.timeseries.profiles import (
    GenerationProfile,
    Profile,
    apply_profile,
    capture_base,
)


def test_synthetic_is_deterministic_for_seed():
    a = Profile.synthetic(hours=24, seed=42)
    b = Profile.synthetic(hours=24, seed=42)
    assert a.multipliers == b.multipliers
    assert Profile.synthetic(hours=24, seed=7).multipliers != a.multipliers


def test_synthetic_peaks_in_the_evening():
    p = Profile.synthetic(hours=24, seed=42, noise=0.0)  # noise off so the curve is exact
    peak_hour = max(range(24), key=lambda t: p.at(t))
    assert 16 <= peak_hour <= 20  # bimodal curve: the higher (evening) peak sits ~19:00
    assert all(m > 0 for m in p.multipliers)


def test_synthetic_is_bimodal_morning_and_evening():
    """Real demand is bimodal: a morning peak (~08:00) and a higher evening peak (~19:00), with a
    midday dip between them and an overnight trough. Guards against regressing to a single afternoon
    hump (the old cos() shape that put the heaviest load at the wrong hours)."""
    p = Profile.synthetic(hours=24, seed=42, noise=0.0)
    morning, midday, evening, overnight = p.at(8), p.at(13), p.at(19), p.at(3)
    assert morning > midday      # a real morning peak rises above the midday dip
    assert evening > morning     # the evening peak is the day's highest
    assert overnight < morning   # overnight is the trough, not (as before) near the daily high


def test_flat_profile_is_unity():
    p = Profile.flat(hours=12, value=1.0)
    assert len(p) == 12
    assert all(p.at(t) == 1.0 for t in range(12))


def test_at_clamps_past_the_end():
    p = Profile([1.0, 2.0, 3.0])
    assert p.at(0) == 1.0
    assert p.at(2) == 3.0
    assert p.at(99) == 3.0  # clamped, never IndexError


def _two_bus_net_with_load():
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, b0)
    pp.create_line_from_parameters(net, b0, b1, length_km=1.0, r_ohm_per_km=0.1,
                                   x_ohm_per_km=0.3, c_nf_per_km=0.0, max_i_ka=1.0)
    pp.create_load(net, b1, p_mw=100.0, q_mvar=20.0)
    return net


def test_profile_scaling_applied():
    net = _two_bus_net_with_load()
    base = capture_base(net)
    apply_profile(net, base, Profile.flat(value=1.5, kind="load"), GenerationProfile(), t=0)
    assert net.load.at[0, "p_mw"] == pytest.approx(150.0)
    assert net.load.at[0, "q_mvar"] == pytest.approx(30.0)


def test_apply_profile_is_idempotent_no_drift():
    """Reapplying any step re-derives from the captured base, never compounding (the classic
    cumulative-scaling drift bug)."""
    net = _two_bus_net_with_load()
    base = capture_base(net)
    lp = Profile([0.8, 1.2, 0.5], kind="load")
    gp = GenerationProfile()
    apply_profile(net, base, lp, gp, t=1)  # x1.2
    apply_profile(net, base, lp, gp, t=1)  # again: still x1.2, not x1.44
    assert net.load.at[0, "p_mw"] == pytest.approx(120.0)
    apply_profile(net, base, lp, gp, t=0)  # back to x0.8 from base, not from x1.2
    assert net.load.at[0, "p_mw"] == pytest.approx(80.0)
