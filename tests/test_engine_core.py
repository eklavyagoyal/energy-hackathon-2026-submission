"""Phase 1 engine core: the invariants the battery feature relies on.

These mirror the doc 03 acceptance tests (AT-1, AT-2, AT-3, AT-6) at the
level this branch needs: the slack guard never solves, islanding is
counted before the solve, a real cascade propagates, and CSS ranking is
ordered. Pure-Python, no LLM.
"""

from __future__ import annotations

import pandapower as pp
import pytest

from src.engine.constants import CSS_MAX
from src.engine.scan import (
    Outage,
    analyze_contingency,
    build_contingency_set,
    rank,
)
from src.engine.severity import band_from_flags, css, status_from_flags
from tests.battery.factories import make_result


# --------------------------------------------------------------------------
# CSS formula and status/band derivation (pure)
# --------------------------------------------------------------------------
def test_css_formula_matches_doc():
    # CSS = 1000*blackout + 500*diverged + 10*shed_pct + 20*min(depth,20) + residual
    assert css(False, False, 0.0, 0, 0) == 0.0
    assert css(False, False, 10.0, 3, 5) == 10 * 10 + 20 * 3 + 5
    assert css(False, True, 100.0, 25, 0) == 500 + 1000 + 20 * 20  # depth capped at 20
    # blackout pin
    assert css(True, False, 100.0, 0, 0) == 1000 + 1000  # == CSS_MAX
    assert css(True, False, 100.0, 0, 0) == CSS_MAX


def test_status_derivation():
    assert status_from_flags(True, False, 0, 0.0, 0) == "FULL_BLACKOUT"
    assert status_from_flags(False, True, 0, 0.0, 0) == "DIVERGED"
    assert status_from_flags(False, False, 2, 0.0, 0) == "CASCADE"
    assert status_from_flags(False, False, 0, 0.0, 3) == "VIOLATIONS"
    assert status_from_flags(False, False, 0, 0.0, 0) == "SECURE"


def test_band_derivation():
    assert band_from_flags(True, False, 0, 0.0, 0) == "CRITICAL"
    assert band_from_flags(False, True, 0, 0.0, 0) == "CRITICAL"
    assert band_from_flags(False, False, 1, 0.0, 0) == "HIGH"
    assert band_from_flags(False, False, 0, 50.0, 0) == "HIGH"
    assert band_from_flags(False, False, 0, 0.0, 2) == "MEDIUM"
    assert band_from_flags(False, False, 0, 0.0, 0) == "LOW"


def test_rank_orders_blackout_over_diverged_over_cascade():
    results = [
        make_result("cascade", status="CASCADE", cascade_depth=2, score=540.0,
                    load_shed_mw=100.0),
        make_result("blackout", status="FULL_BLACKOUT", blackout=True, score=CSS_MAX,
                    load_shed_mw=4000.0),
        make_result("diverged", status="DIVERGED", diverged=True, score=1500.0,
                    load_shed_mw=2000.0),
        make_result("secure", status="SECURE", score=0.0),
    ]
    ordered = [r.contingency_id for r in rank(results)]
    assert ordered == ["blackout", "diverged", "cascade", "secure"]


# --------------------------------------------------------------------------
# AT-1: ext_grid outage is a solver-free FULL_BLACKOUT
# --------------------------------------------------------------------------
def test_ext_grid_outage_full_blackout_no_solver(case118_net, monkeypatch):
    ext_idx = int(case118_net.ext_grid.index[0])
    outage = Outage("ext_grid", ext_idx, f"ext_grid_{ext_idx}")

    calls = {"runpp": 0}
    orig = pp.runpp

    def counting(*a, **k):
        calls["runpp"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(pp, "runpp", counting)
    r = analyze_contingency(case118_net, outage)

    assert r.status == "FULL_BLACKOUT"
    assert r.severity.score == CSS_MAX
    assert r.severity.blackout is True
    assert calls["runpp"] == 0  # dedicated topological path, never solves


# --------------------------------------------------------------------------
# AT-2: islanding is detected and counted, no exception on the dead island
# --------------------------------------------------------------------------
def test_islanding_radial_stub_counts_shed():
    net = pp.create_empty_network()
    for i in range(4):
        pp.create_bus(net, vn_kv=110.0, index=i)
    pp.create_ext_grid(net, bus=0, vm_pu=1.02)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        pp.create_line_from_parameters(
            net, a, b, length_km=10, r_ohm_per_km=0.06, x_ohm_per_km=0.2,
            c_nf_per_km=10, max_i_ka=0.4,
        )
    pp.create_load(net, bus=2, p_mw=15, q_mvar=5)
    pp.create_load(net, bus=3, p_mw=15, q_mvar=5)

    # removing line 1 (1-2) islands buses 2 and 3 (30 MW)
    outage = Outage("line", 1, "line 1 (bus 1 to bus 2)")
    r = analyze_contingency(net, outage)
    assert set(r.preflight.islanded_buses) == {2, 3}
    assert r.severity.load_shed_mw == pytest.approx(30.0)
    assert r.status != "FULL_BLACKOUT"  # bus 0 (and 1) still supplied


# --------------------------------------------------------------------------
# AT-3: a real cascade propagates on the demo scenario
# --------------------------------------------------------------------------
def test_demo_scenario_has_a_real_cascade(demo_baseline):
    _cont, results = demo_baseline
    cascading = [
        r for r in results
        if r.outage["etype"] in ("line", "trafo")
        and r.status in ("CASCADE", "DIVERGED")
    ]
    assert cascading, "demo scenario must contain at least one line/trafo cascade"
    worst = max(cascading, key=lambda r: r.severity.score)
    assert worst.severity.cascade_depth >= 1
    assert worst.cascade_trace  # trace recorded


def test_demo_base_case_is_secure(demo_net):
    from src.engine.network import base_case_summary

    bc = base_case_summary(demo_net)
    assert bc["converged"] is True
    assert bc["n_overloads"] == 0
    assert bc["min_vm_pu"] >= 0.95
