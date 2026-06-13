"""Network helpers: working copies, stress injection, base-case summary.

Doc reference: docs/03-phase1-engine.md section 3 (inject_stress) and
docs/02-architecture.md section 6 (working_copy, single live net).

Dataset portability rules enforced here and engine-wide:
- never iterate `range(len(net.bus))`; always `net.bus.index`
- never hardcode a bus index; slack buses come from `net.ext_grid.bus`
- never assume the index dtype; `native_index` preserves strings
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandapower as pp
import pandapower.topology as top

from src.engine.constants import (
    OVERLOAD_LIMIT,
    RATING_BASE_LOADING_PCT,
    RATING_FLOOR_QUANTILE,
    SEED,
    VOLTAGE_BAND_HIGH,
    VOLTAGE_BAND_LOW,
    VOLTAGE_TOL,
)

logger = logging.getLogger(__name__)

try:  # pandapower re-exports this at top level in recent versions
    from pandapower.powerflow import LoadflowNotConverged
except ImportError:  # pragma: no cover
    LoadflowNotConverged = pp.LoadflowNotConverged


def native_index(idx: Any) -> int | str:
    """Return a pandas index value as a plain Python scalar.

    pandapower DataFrame indices are integers on every solver-backed
    net, but the portability contract types every element/bus reference
    as int | str. numpy integer types are converted to int (for JSON
    serializability); strings pass through untouched.
    """
    if isinstance(idx, (np.integer, int)):
        return int(idx)
    return str(idx)


def working_copy(net: "pp.pandapowerNet") -> "pp.pandapowerNet":
    """Every analysis starts from a deep copy; the live net is read-only
    to all analysis code paths (docs/02-architecture.md section 6)."""
    return copy.deepcopy(net)


def slack_bus_set(net: "pp.pandapowerNet") -> set:
    """ALL ext_grid buses, in service or not, as a set.

    This is the battery-placement exclusion set: a battery on any slack
    bus is meaningless because the slack already provides unbounded P
    and Q at fixed voltage. Set membership, never an equality check
    against a hardcoded index (portability rule 6).
    """
    return {native_index(b) for b in net.ext_grid.bus.values}


def live_slack_bus_set(net: "pp.pandapowerNet") -> set:
    """Buses of IN-SERVICE ext_grids. The slack guard works on this set
    so multi-slack nets behave correctly: losing one of two slacks is an
    islanding question, not an automatic blackout."""
    es = net.ext_grid
    return {native_index(b) for b in es.bus[es.in_service].values}


def opf_available(net: "pp.pandapowerNet") -> bool:
    """Structural check that pp.runopp could be attempted at all:
    poly_cost present and at least one controllable element.

    Required guard before ANY runopp code path (dataset portability
    rule 8: real TSO exports often ship without cost data). The battery
    feature itself never calls runopp (its dispatch model is the fixed
    full-discharge heuristic, see docs/battery-feature.md), but it logs
    this flag so the fallback is explicit.
    """
    if "poly_cost" not in net or len(net.poly_cost) == 0:
        return False
    if len(net.gen) > 0 and "controllable" in net.gen.columns:
        if not net.gen.controllable.fillna(False).any():
            return False
    return True


def total_load_mw(net: "pp.pandapowerNet") -> float:
    """Sum of in-service load p_mw (scaling-aware)."""
    ld = net.load
    if len(ld) == 0:
        return 0.0
    scaling = ld.scaling if "scaling" in ld.columns else 1.0
    return float((ld.p_mw * scaling)[ld.in_service].sum())


def inject_stress(
    net: "pp.pandapowerNet",
    load_scale: float,
    targets: str = "top10_loads",
    seed: int = SEED,
) -> "pp.pandapowerNet":
    """Scale p_mw and q_mvar of the target loads by load_scale.

    The slack (ext_grid) absorbs the imbalance. Returns a modified deep
    copy; the input net is never mutated. Target selection is
    deterministic: top 10 loads by p_mw, ties broken by index
    (mergesort is stable), so reruns reproduce scenarios exactly
    (SEED 42 convention; no stochastic choice is actually needed here).
    """
    if targets != "top10_loads":
        raise ValueError(f"unknown stress target preset: {targets!r}")
    work = copy.deepcopy(net)
    n_targets = min(10, len(work.load))
    idx = (
        work.load.sort_values("p_mw", ascending=False, kind="mergesort")
        .head(n_targets)
        .index
    )
    work.load.loc[idx, ["p_mw", "q_mvar"]] *= load_scale
    return work


def apply_proportional_line_ratings(
    net: "pp.pandapowerNet",
    base_loading_pct: float = RATING_BASE_LOADING_PCT,
    floor_quantile: float = RATING_FLOOR_QUANTILE,
) -> "pp.pandapowerNet":
    """Assign deterministic per-line emergency ratings when the dataset
    ships placeholders (case118: 9900 MVA on everything). Mutates the
    passed net IN PLACE and returns it; call once at load time, never
    during analysis.

    Method (Motter-Lai 2002 style, logged as D59): solve the base case,
    set max_i_ka = base current / (base_loading_pct / 100), then floor
    the result at the floor_quantile of the assigned distribution so
    lines with near-zero base flow keep usable headroom instead of
    becoming hair triggers. Every number derives from the net itself;
    no per-line knowledge is invented.

    Trafo ratings are deliberately NOT touched: pandapower defines
    trafo impedance (vk_percent) relative to sn_mva, so rescaling
    sn_mva would change the physics, not just the rating. On case118
    the cascade story is carried by the 173 lines; the 13 trafos stay
    effectively unconstrained and this is documented honestly.
    """
    work = copy.deepcopy(net)
    pp.runpp(work)
    base_i = work.res_line.i_ka
    in_service = net.line.in_service & base_i.notna()
    assigned = base_i[in_service] / (base_loading_pct / 100.0)
    if len(assigned) == 0:
        return net
    floor = float(assigned.quantile(floor_quantile))
    net.line.loc[in_service, "max_i_ka"] = assigned.clip(lower=floor)
    logger.info(
        "assigned proportional line ratings: target base loading %.0f%%, "
        "floor %.4f kA (q%.0f), %d lines",
        base_loading_pct, floor, floor_quantile * 100, int(in_service.sum()),
    )
    return net


def lift_voltage_profile(
    net: "pp.pandapowerNet",
    gen_floor: float = 1.01,
    gen_ceiling: float = 1.04,
    ext_grid_vm: float = 1.04,
) -> "pp.pandapowerNet":
    """Raise generator and ext_grid voltage setpoints to lift the bus
    voltage profile into the 0.95 to 1.05 p.u. band. Mutates the passed
    net IN PLACE and returns it; call once at load time.

    Why (logged as D60): pandapower's case118 ships a voltage profile
    that dips below 0.95 p.u. at load buses far from generation once the
    grid is loaded, so the unstressed base case already shows voltage
    violations. That muddies both the demo (the base case is supposed to
    be secure) and the battery scoring (every contingency inherits the
    base undervoltage). Setting generator and ext_grid voltage setpoints
    into a narrow band just inside the limits is an operational choice,
    not a topology change, and gives a clean in-band base case. The
    setpoints are kept off the 1.05 edge so buses pinned to a generator
    are not borderline violations. Real TSO datasets bring their own
    profile and skip this.
    """
    if len(net.gen) > 0 and "vm_pu" in net.gen.columns:
        net.gen["vm_pu"] = net.gen["vm_pu"].clip(lower=gen_floor, upper=gen_ceiling)
    if len(net.ext_grid) > 0:
        net.ext_grid["vm_pu"] = ext_grid_vm
    return net


def buses_within(net: "pp.pandapowerNet", center, radius: int) -> set:
    """Buses within `radius` electrical hops of `center` on the
    in-service topology. respect_switches=True so switch state is
    honored on real datasets (dataset portability rule 9)."""
    import networkx as nx

    g = top.create_nxgraph(net, respect_switches=True)
    if center not in g:
        raise ValueError(f"center bus {center!r} is not in the network graph")
    reachable = nx.single_source_shortest_path_length(g, center, cutoff=radius)
    return set(reachable.keys())


def inject_local_stress(
    net: "pp.pandapowerNet",
    center,
    radius: int,
    load_scale: float,
) -> "pp.pandapowerNet":
    """Scale loads within `radius` hops of `center` by load_scale.

    Returns a modified deep copy; the input is never mutated. Unlike
    inject_stress (top-10 loads grid-wide), this concentrates the stress
    in one electrical neighborhood, producing LOCALIZED congestion: a
    handful of N-1 contingencies near the pocket cascade while the rest
    of the grid keeps headroom. That is what lets a single battery
    relieve the corridor without overloading distant lines, and what
    keeps the verification impact set (and runtime) small. The center
    and radius are deterministic inputs, never a hardcoded bus index.
    """
    work = copy.deepcopy(net)
    region = buses_within(work, center, radius)
    mask = work.load.bus.isin(region)
    work.load.loc[mask, ["p_mw", "q_mvar"]] *= load_scale
    return work


def base_case_summary(net: "pp.pandapowerNet") -> dict:
    """runpp on a working copy and summarize. Never mutates the input.

    Shape matches the run_power_flow tool return in docs/04-agent.md.
    """
    work = working_copy(net)
    try:
        pp.runpp(work)
        converged = bool(work.converged)
    except LoadflowNotConverged:
        converged = False
    if not converged:
        return {
            "converged": False,
            "max_line_loading_pct": None,
            "n_overloads": None,
            "min_vm_pu": None,
            "max_vm_pu": None,
            "n_voltage_violations": None,
            "total_load_mw": total_load_mw(net),
        }
    loading = work.res_line.loading_percent.dropna()
    vm = work.res_bus.vm_pu.dropna()
    return {
        "converged": True,
        "max_line_loading_pct": float(loading.max()) if len(loading) else 0.0,
        "n_overloads": int((loading > OVERLOAD_LIMIT).sum()),
        "min_vm_pu": float(vm.min()) if len(vm) else None,
        "max_vm_pu": float(vm.max()) if len(vm) else None,
        "n_voltage_violations": int(
            (
                (vm < VOLTAGE_BAND_LOW - VOLTAGE_TOL)
                | (vm > VOLTAGE_BAND_HIGH + VOLTAGE_TOL)
            ).sum()
        ),
        "total_load_mw": total_load_mw(net),
    }
