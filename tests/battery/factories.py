"""Synthetic ContingencyResult factories for battery unit tests.

These build real engine dataclasses (Severity, Preflight,
ContingencyResult) with hand-chosen fields so scoring and verification
math can be asserted against known inputs WITHOUT running a solver. That
isolation is the point: the scoring and delta layers never touch
pandapower, so they are tested as pure functions.
"""

from __future__ import annotations

import pandapower as pp

from src.engine.scan import ContingencyResult
from src.engine.severity import Severity


def make_result(
    contingency_id: str,
    *,
    status: str = "SECURE",
    blackout: bool = False,
    diverged: bool = False,
    cascade_depth: int = 0,
    load_shed_mw: float = 0.0,
    load_shed_pct: float = 0.0,
    residual_violations: int = 0,
    score: float = 0.0,
    line_loading: dict | None = None,
    trafo_loading: dict | None = None,
    bus_vm: dict | None = None,
    islanded: list | None = None,
    tripped: list | None = None,
) -> ContingencyResult:
    sev = Severity(
        score=score,
        band=_band(blackout, diverged, cascade_depth, load_shed_mw, residual_violations),
        blackout=blackout,
        diverged=diverged,
        cascade_depth=cascade_depth,
        load_shed_mw=load_shed_mw,
        load_shed_pct=load_shed_pct,
        residual_violations=residual_violations,
    )
    ll = line_loading or {}
    tl = trafo_loading or {}
    all_loadings = list(ll.values()) + list(tl.values())
    return ContingencyResult(
        contingency_id=contingency_id,
        outage={"etype": "line", "index": 0, "name": contingency_id,
                "from_bus": None, "to_bus": None},
        status=status,
        severity=sev,
        preflight=_preflight(islanded or []),
        cascade_trace=[],
        first_overloads=[],
        final_state={
            "converged": not (blackout or diverged),
            "max_loading_pct": max(all_loadings) if all_loadings else 0.0,
            "min_vm_pu": min(bus_vm.values()) if bus_vm else None,
            "violations": [],
        },
        timing_ms=0.0,
        final_line_loading=ll,
        final_trafo_loading=tl,
        final_bus_vm=bus_vm or {},
        all_islanded_buses=islanded or [],
        tripped_elements=tripped or [],
    )


def _band(blackout, diverged, depth, shed, residual):
    if blackout or diverged:
        return "CRITICAL"
    if depth >= 1 or shed > 0:
        return "HIGH"
    if residual > 0:
        return "MEDIUM"
    return "LOW"


def _preflight(islanded):
    from src.engine.preflight import Preflight

    return Preflight(slack_lost=False, islanded_buses=list(islanded), islanded_load_mw=0.0)


def line_net(n_buses: int = 4, slack_bus=0, index=None):
    """A simple radial line network for scoring/endpoint tests.

    Creation only; this net is NOT solved (scoring never solves). `index`
    overrides the bus indices (e.g. non-sequential ints or strings) to
    exercise dtype-agnostic code paths.
    """
    net = pp.create_empty_network()
    idx = index if index is not None else list(range(n_buses))
    for i in idx:
        pp.create_bus(net, vn_kv=110.0, index=i)
    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0)
    for a, b in zip(idx[:-1], idx[1:]):
        pp.create_line_from_parameters(
            net, a, b, length_km=10, r_ohm_per_km=0.1, x_ohm_per_km=0.3,
            c_nf_per_km=10, max_i_ka=1.0,
        )
    return net
