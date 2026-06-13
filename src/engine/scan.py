"""Contingency set, per-contingency analysis pipeline, full N-1 sweep.

Doc reference: docs/03-phase1-engine.md sections 2 and 4. The ordered
pipeline is mandatory and load-bearing:

  STEP 1  slack guard FIRST, dedicated code path, NO SOLVER CALL
  STEP 2  islanding BEFORE any solve; de-energize and count the shed
  STEP 3  cascade loop on the supplied component
  STEP 4  severity (CSS) and record assembly

THE STATIC-SCAN TRAP (doc 03 section 1) is the forbidden anti-pattern:
no severity here ever derives from a single post-outage solve; every
contingency runs the full iterative cascade loop.

Scope note for this branch (D49): the Screener slot and the
screener-promoted run_full_scan of doc 03 section 7 belong to the main
Phase 1 build. The battery bolt-on needs per-contingency results for
the WHOLE set, so this module ships run_contingency_sweep, which is
doc 07's screener="none" semantics: full AC cascade analysis on every
contingency in the list.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import pandapower as pp

from src.engine.cascade import CascadeOutcome, run_cascade
from src.engine.constants import CSS_MAX
from src.engine.network import native_index, total_load_mw
from src.engine.preflight import (
    Preflight,
    all_buses_unsupplied,
    preflight_islanding,
    slack_lost,
)
from src.engine.severity import (
    Severity,
    score_severity,
    status_from_flags,
)


@dataclass(frozen=True)
class Outage:
    etype: Literal["line", "trafo", "gen", "ext_grid"]
    index: int | str  # pandapower DataFrame index, native dtype preserved
    name: str  # e.g. "line 8 (bus 8 to bus 9)", built from net tables

    @property
    def contingency_id(self) -> str:
        return f"{self.etype}_{self.index}"


@dataclass
class ContingencyResult:
    contingency_id: str
    outage: dict  # {etype, index, name, from_bus, to_bus}
    status: str  # SECURE | VIOLATIONS | CASCADE | DIVERGED | FULL_BLACKOUT
    severity: Severity
    preflight: Preflight
    cascade_trace: list  # list[TraceStep]
    first_overloads: list
    final_state: dict
    timing_ms: float
    # Additive per-element end state for downstream consumers, D50.
    # The battery bolt-on reads ONLY these plus the fields above; it
    # never reaches back into the solver from a result record.
    final_line_loading: dict = field(default_factory=dict)
    final_trafo_loading: dict = field(default_factory=dict)
    final_bus_vm: dict = field(default_factory=dict)
    all_islanded_buses: list = field(default_factory=list)
    tripped_elements: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def element_name(net: "pp.pandapowerNet", etype: str, idx) -> str:
    """Human name built from net tables, never invented."""
    if etype == "line":
        row = net.line.loc[idx]
        return f"line {native_index(idx)} (bus {native_index(row.from_bus)} to bus {native_index(row.to_bus)})"
    if etype == "trafo":
        row = net.trafo.loc[idx]
        return f"trafo {native_index(idx)} (bus {native_index(row.hv_bus)} to bus {native_index(row.lv_bus)})"
    if etype == "ext_grid":
        row = net.ext_grid.loc[idx]
        return f"ext_grid {native_index(idx)} (bus {native_index(row.bus)})"
    if etype == "gen":
        row = net.gen.loc[idx]
        return f"gen {native_index(idx)} (bus {native_index(row.bus)})"
    return f"{etype} {native_index(idx)}"


def build_contingency_set(net: "pp.pandapowerNet") -> list:
    """All in-service lines + all in-service trafos + every ext_grid
    (doc 03 section 2; the ext_grid outage exists to demo the blackout
    path). Deterministic order: etype, then table order. Element counts
    are read from the net at runtime, never hardcoded (D26)."""
    out: list[Outage] = []
    for etype in ("line", "trafo"):
        table = getattr(net, etype)
        if len(table) == 0:
            continue
        for idx in table.index[table.in_service]:
            out.append(Outage(etype, native_index(idx), element_name(net, etype, idx)))
    for idx in net.ext_grid.index:
        out.append(Outage("ext_grid", native_index(idx), element_name(net, "ext_grid", idx)))
    return out


def set_out_of_service(net: "pp.pandapowerNet", outage: Outage) -> None:
    net[outage.etype].loc[outage.index, "in_service"] = False


def _outage_dict(net: "pp.pandapowerNet", outage: Outage) -> dict:
    from_bus = to_bus = None
    if outage.etype == "line" and outage.index in net.line.index:
        row = net.line.loc[outage.index]
        from_bus = native_index(row.from_bus)
        to_bus = native_index(row.to_bus)
    return {
        "etype": outage.etype,
        "index": outage.index,
        "name": outage.name,
        "from_bus": from_bus,
        "to_bus": to_bus,
    }


def full_blackout_result(
    net: "pp.pandapowerNet", outage: Outage, started_at: float
) -> ContingencyResult:
    """Dedicated blackout constructor. MUST NOT share a code path with
    the solver: the verdict is topological (AT-1 asserts a runpp call
    counter of ZERO on this branch). CSS pinned to CSS_MAX (D9)."""
    system_load = total_load_mw(net)
    severity = Severity(
        score=CSS_MAX,
        band="CRITICAL",
        blackout=True,
        diverged=False,
        cascade_depth=0,
        load_shed_mw=system_load,
        load_shed_pct=100.0,
        residual_violations=0,
    )
    all_buses = sorted(
        (native_index(b) for b in net.bus.index[net.bus.in_service]), key=str
    )
    return ContingencyResult(
        contingency_id=outage.contingency_id,
        outage=_outage_dict(net, outage),
        status="FULL_BLACKOUT",
        severity=severity,
        preflight=Preflight(slack_lost=True, islanded_buses=[], islanded_load_mw=0.0),
        cascade_trace=[],
        first_overloads=[],
        final_state={
            "converged": False,
            "max_loading_pct": None,
            "min_vm_pu": None,
            "violations": [],
        },
        timing_ms=(time.perf_counter() - started_at) * 1000.0,
        all_islanded_buses=all_buses,
    )


def analyze_contingency(net: "pp.pandapowerNet", outage: Outage) -> ContingencyResult:
    """The full per-contingency pipeline on a deep copy of the input
    net. The input net is never mutated."""
    t0 = time.perf_counter()
    work = copy.deepcopy(net)
    set_out_of_service(work, outage)

    # STEP 1: SLACK GUARD. Dedicated code path, no solver call.
    if slack_lost(work):
        return full_blackout_result(net, outage, started_at=t0)

    # STEP 2: ISLANDING. Before every power flow.
    pf, unsup = preflight_islanding(work)
    if all_buses_unsupplied(work, unsup):
        return full_blackout_result(net, outage, started_at=t0)

    # STEP 3: CASCADE LOOP on the supplied component.
    cascade: CascadeOutcome = run_cascade(work, pf.islanded_load_mw)

    # STEP 4: severity and record assembly.
    severity = score_severity(cascade, total_load_mw=total_load_mw(net))
    status = status_from_flags(
        blackout=False,
        diverged=severity.diverged,
        cascade_depth=severity.cascade_depth,
        load_shed_mw=severity.load_shed_mw,
        residual_violations=severity.residual_violations,
    )
    all_islanded = sorted(
        set(pf.islanded_buses) | set(cascade.islanded_during_cascade), key=str
    )
    return ContingencyResult(
        contingency_id=outage.contingency_id,
        outage=_outage_dict(net, outage),
        status=status,
        severity=severity,
        preflight=pf,
        cascade_trace=cascade.trace,
        first_overloads=cascade.first_overloads,
        final_state=cascade.final_state,
        timing_ms=(time.perf_counter() - t0) * 1000.0,
        final_line_loading=cascade.final_line_loading,
        final_trafo_loading=cascade.final_trafo_loading,
        final_bus_vm=cascade.final_bus_vm,
        all_islanded_buses=all_islanded,
        tripped_elements=cascade.tripped_elements,
    )


def run_contingency_sweep(
    net: "pp.pandapowerNet", contingencies: list | None = None
) -> list:
    """Full AC cascade analysis on every contingency in the list
    (doc 07 screener="none" semantics). This is the single physics
    pipeline both the baseline and the battery counterfactual run
    through; there is no second implementation anywhere.
    """
    if contingencies is None:
        contingencies = build_contingency_set(net)
    return [analyze_contingency(net, outage) for outage in contingencies]


def rank(results: list) -> list:
    """Doc 03 section 6.3 ordering: CSS descending, tiebreak
    load_shed_mw descending, then contingency_id ascending."""
    return sorted(
        results,
        key=lambda r: (-r.severity.score, -r.severity.load_shed_mw, r.contingency_id),
    )
