"""Shared fixtures. Session-scoped so the expensive case118 load and the
baseline N-1 sweep run once for the whole suite."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def case118_net():
    from src.grid.loader import Case118Loader

    return Case118Loader().load()


@pytest.fixture(scope="session")
def demo_net(case118_net):
    from src.engine.scenarios import apply_scenario

    return apply_scenario(case118_net, "demo_congestion")


@pytest.fixture(scope="session")
def demo_baseline(demo_net):
    from src.engine.scan import build_contingency_set, run_contingency_sweep

    cont = build_contingency_set(demo_net)
    results = run_contingency_sweep(demo_net, cont)
    return cont, results
