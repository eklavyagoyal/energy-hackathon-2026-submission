"""U5: executable record of the Phase 1 integrity audit (docs/audit-findings.md).

Pins the audit conclusions so a regression cannot silently reintroduce a failure mode:
no hardcoded slack index (A2), the cascade is iterative not a static scan (A1), multiple ext_grids
are handled (A2), slack loss is a blackout with NO solver call (A1/A5), divergence is marked not
faked (A5), and stress injection is reproducible (A9).
"""
from __future__ import annotations

import glob
import re

import pandapower as pp
import pytest

from src.engine.constants import CSS_MAX
from src.engine.network import working_copy
from src.engine.preflight import slack_lost
from src.engine.scan import analyze_contingency, build_contingency_set, run_contingency_sweep
from src.engine.scenarios import apply_scenario
from src.engine.network import inject_stress
from src.grid.loader import Case118Loader

REPO = __file__.rsplit("/tests/", 1)[0]


@pytest.fixture(scope="module")
def base_net():
    return Case118Loader().load()


@pytest.fixture(scope="module")
def congested_net(base_net):
    return apply_scenario(base_net, "demo_congestion")


def test_audit_no_hardcoded_slack_zero():
    """A2: no slack/bus equality against a hardcoded index anywhere in engine or agent code."""
    pattern = re.compile(r"\b(slack|bus_idx|ext_grid)\s*==\s*0\b|\.bus\s*==\s*0\b|\bslack\s*=\s*0\b")
    offenders = []
    for path in glob.glob(f"{REPO}/src/engine/*.py") + glob.glob(f"{REPO}/src/agent/*.py"):
        for n, line in enumerate(open(path, encoding="utf-8"), 1):
            if pattern.search(line):
                offenders.append(f"{path}:{n}: {line.strip()}")
    assert offenders == [], f"hardcoded slack/bus literals found: {offenders}"


def test_cascade_is_iterative(congested_net):
    """A1: at least one contingency drives the iterative cascade loop past depth 0 (a trip and
    re-solve). A single static post-outage solve could never report cascade_depth >= 1."""
    sweep = run_contingency_sweep(congested_net)
    assert any(r.severity.cascade_depth >= 1 for r in sweep), \
        "no contingency cascaded; the loop may have degenerated to a static scan"
    # the trace corroborates: a depth-d result carries d recorded iterations
    deep = [r for r in sweep if r.severity.cascade_depth >= 1][0]
    assert len(deep.cascade_trace) == deep.severity.cascade_depth


def test_multiple_ext_grids_handled():
    """A2: a 2-slack net. Losing one ext_grid is NOT a blackout; losing both is."""
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=110.0)
    b1 = pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, b0)
    pp.create_ext_grid(net, b1)
    pp.create_line_from_parameters(net, b0, b1, length_km=1.0, r_ohm_per_km=0.1,
                                   x_ohm_per_km=0.3, c_nf_per_km=0.0, max_i_ka=1.0)
    assert slack_lost(net) is False           # both live
    net.ext_grid.at[0, "in_service"] = False
    assert slack_lost(net) is False           # one remains: an islanding question, not a blackout
    net.ext_grid.at[1, "in_service"] = False
    assert slack_lost(net) is True            # none left


def test_slack_loss_is_blackout_without_solver(base_net, monkeypatch):
    """A1/A5: the ext_grid outage returns FULL_BLACKOUT at CSS_MAX with ZERO solver calls."""
    calls = {"n": 0}
    orig = pp.runpp

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(pp, "runpp", counting)
    extg = next(o for o in build_contingency_set(base_net) if o.etype == "ext_grid")
    calls["n"] = 0
    r = analyze_contingency(base_net, extg)
    assert r.status == "FULL_BLACKOUT"
    assert r.severity.score == CSS_MAX
    assert calls["n"] == 0  # topological verdict, no power flow


def test_divergence_marked_not_faked(base_net, monkeypatch):
    """A5: a solver that fails to converge is recorded as diverged with the load counted as shed,
    never swallowed into a fake success."""
    from src.engine import cascade as C
    from src.engine.network import LoadflowNotConverged

    def boom(net, *a, **k):
        raise LoadflowNotConverged("forced non-convergence")

    monkeypatch.setattr(C.pp, "runpp", boom)
    out = C.run_cascade(working_copy(base_net), islanded_load_mw=0.0)
    assert out.diverged is True
    assert out.shed_mw > 0
    assert out.final_state["converged"] is False


def test_seed_reproducible(base_net):
    """A9: inject_stress is deterministic for a fixed seed (demo scenarios reproduce exactly)."""
    a = inject_stress(base_net, load_scale=2.0, seed=42)
    b = inject_stress(base_net, load_scale=2.0, seed=42)
    assert a.load["p_mw"].tolist() == b.load["p_mw"].tolist()
