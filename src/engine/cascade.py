"""The iterative cascade loop.

Doc reference: docs/03-phase1-engine.md section 5 (canonical reference
semantics, implemented exactly):

  loop (max MAX_CASCADE_ITERS):
    runpp; on non-convergence: diverged, remaining supplied load
      counted as shed (conservative, logged decision D7), break
    tripping = elements with loading_percent > TRIP_THRESHOLD
      (lines AND trafos)
    if none: break (stable end state, possibly with 100 to 120 pct
      residual violations)
    ALL tripping elements go out of service simultaneously (D5)
    re-run the islanding check (mid-cascade islanding is real, D38)

The two named bugs NOT to write (doc section 5.1): tripping at
OVERLOAD_LIMIT instead of TRIP_THRESHOLD, and checking islanding only
once before the loop.

Additive end-state fields (logged as D50): final_line_loading,
final_trafo_loading and final_bus_vm carry the full per-element vectors
of the last converged solve, and all_islanded_buses / tripped_elements
accumulate across the run. The battery bolt-on consumes these; the wire
schemas of docs/07-api-contracts.md are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandapower as pp

from src.engine.constants import (
    MAX_CASCADE_ITERS,
    OVERLOAD_LIMIT,
    TRIP_THRESHOLD,
    VOLTAGE_BAND_HIGH,
    VOLTAGE_BAND_LOW,
    VOLTAGE_TOL,
)
from src.engine.network import LoadflowNotConverged, native_index
from src.engine.preflight import deenergize_and_count, unsupplied

logger = logging.getLogger(__name__)

# Branch element tables that carry loading_percent results and can trip.
_BRANCH_ETYPES = ("line", "trafo")


@dataclass
class TraceStep:
    iter: int
    tripped: list  # [{"etype", "index", "loading_pct"}]
    converged: bool
    max_loading_pct: float
    min_vm_pu: float
    load_shed_mw: float  # cumulative shed at end of this iteration


@dataclass
class CascadeOutcome:
    depth: int
    shed_mw: float
    diverged: bool
    trace: list  # list[TraceStep]
    first_overloads: list  # elements > OVERLOAD_LIMIT after the FIRST solve
    final_state: dict  # {"converged", "max_loading_pct", "min_vm_pu", "violations"}
    # Additive per-element end state from the last converged solve (D50).
    final_line_loading: dict = field(default_factory=dict)
    final_trafo_loading: dict = field(default_factory=dict)
    final_bus_vm: dict = field(default_factory=dict)
    # Buses islanded DURING the cascade (preflight islanding is recorded
    # separately on the ContingencyResult).
    islanded_during_cascade: list = field(default_factory=list)
    # Every element tripped across all iterations: [{"etype", "index",
    # "loading_pct"}].
    tripped_elements: list = field(default_factory=list)


def elements_over(net: "pp.pandapowerNet", threshold: float) -> list:
    """In-service lines and trafos whose loading_percent exceeds the
    threshold. Out-of-service elements carry NaN results and drop out."""
    out = []
    for etype in _BRANCH_ETYPES:
        table = getattr(net, etype)
        if len(table) == 0:
            continue
        res = net[f"res_{etype}"]
        loading = res.loading_percent.dropna()
        for idx, pct in loading[loading > threshold].items():
            if bool(table.at[idx, "in_service"]):
                out.append(
                    {
                        "etype": etype,
                        "index": native_index(idx),
                        "loading_pct": float(pct),
                    }
                )
    return out


def set_all_out_of_service(net: "pp.pandapowerNet", tripping: list) -> None:
    """Simultaneous-trip policy (D5): every element past the threshold
    trips in the same iteration."""
    for item in tripping:
        net[item["etype"]].loc[item["index"], "in_service"] = False


def remaining_supplied_load_mw(net: "pp.pandapowerNet") -> float:
    """In-service load only: islanded loads were already de-energized,
    so what is still in service is what the diverging component was
    supplying. Conservative shed accounting (D7)."""
    ld = net.load
    if len(ld) == 0:
        return 0.0
    scaling = ld.scaling if "scaling" in ld.columns else 1.0
    return float((ld.p_mw * scaling)[ld.in_service].sum())


def snapshot_results(net: "pp.pandapowerNet") -> dict:
    """Capture loadings and voltages of a converged solve as plain
    dicts keyed by native element index."""
    line_loading = {
        native_index(i): float(v)
        for i, v in net.res_line.loading_percent.dropna().items()
    }
    trafo_loading = {}
    if len(net.trafo) > 0:
        trafo_loading = {
            native_index(i): float(v)
            for i, v in net.res_trafo.loading_percent.dropna().items()
        }
    bus_vm = {
        native_index(i): float(v) for i, v in net.res_bus.vm_pu.dropna().items()
    }
    all_loadings = list(line_loading.values()) + list(trafo_loading.values())
    return {
        "line_loading": line_loading,
        "trafo_loading": trafo_loading,
        "bus_vm": bus_vm,
        "max_loading_pct": max(all_loadings) if all_loadings else 0.0,
        "min_vm_pu": min(bus_vm.values()) if bus_vm else None,
    }


def _violations(snapshot: dict) -> list:
    """Residual violations at the stable end state: lines above
    OVERLOAD_LIMIT plus buses outside VOLTAGE_BAND (doc 03 section 5)."""
    out = []
    for idx, pct in snapshot["line_loading"].items():
        if pct > OVERLOAD_LIMIT:
            out.append({"etype": "line", "index": idx, "loading_pct": pct})
    for idx, vm in snapshot["bus_vm"].items():
        if vm < VOLTAGE_BAND_LOW - VOLTAGE_TOL or vm > VOLTAGE_BAND_HIGH + VOLTAGE_TOL:
            out.append({"etype": "bus", "index": idx, "vm_pu": vm})
    return out


def build_final_state(last_good: dict | None, diverged: bool) -> dict:
    if last_good is None:
        return {
            "converged": False,
            "max_loading_pct": None,
            "min_vm_pu": None,
            "violations": [],
        }
    return {
        "converged": not diverged,
        "max_loading_pct": last_good["max_loading_pct"],
        "min_vm_pu": last_good["min_vm_pu"],
        "violations": _violations(last_good),
    }


def run_cascade(work: "pp.pandapowerNet", islanded_load_mw: float) -> CascadeOutcome:
    """Run the cascade loop on the supplied component of a WORKING COPY
    (the caller owns the copy; this function mutates it freely)."""
    depth = 0
    shed_mw = islanded_load_mw
    trace: list = []
    diverged = False
    first_overloads: list = []
    last_good: dict | None = None
    islanded_during: set = set()
    tripped_all: list = []

    for it in range(1, MAX_CASCADE_ITERS + 1):
        try:
            pp.runpp(work)
        except Exception as exc:
            if not isinstance(exc, LoadflowNotConverged):
                # pandapower can raise auxiliary errors on degenerate
                # post-trip topologies; treat any solve failure as
                # divergence (conservative, same as D7) but log it
                # loudly so a programming error cannot hide here.
                logger.warning("cascade solve failed (%s): %s", type(exc).__name__, exc)
            diverged = True
            shed_mw += remaining_supplied_load_mw(work)
            break
        snap = snapshot_results(work)
        last_good = snap
        if it == 1:
            first_overloads = elements_over(work, OVERLOAD_LIMIT)

        tripping = elements_over(work, TRIP_THRESHOLD)
        if not tripping:
            break  # stable end state reached

        set_all_out_of_service(work, tripping)
        tripped_all.extend(tripping)
        depth += 1

        # Mid-cascade islanding is real: recheck EVERY iteration (D38).
        newly_unsupplied = unsupplied(work)
        if newly_unsupplied:
            shed_mw += deenergize_and_count(work, newly_unsupplied)
            islanded_during.update(native_index(b) for b in newly_unsupplied)

        trace.append(
            TraceStep(
                iter=depth,
                tripped=tripping,
                converged=True,
                max_loading_pct=snap["max_loading_pct"],
                min_vm_pu=snap["min_vm_pu"],
                load_shed_mw=shed_mw,
            )
        )
    else:
        diverged = True  # iteration cap hit = diverged (D6)

    final_state = build_final_state(last_good, diverged)
    return CascadeOutcome(
        depth=depth,
        shed_mw=shed_mw,
        diverged=diverged,
        trace=trace,
        first_overloads=first_overloads,
        final_state=final_state,
        final_line_loading=last_good["line_loading"] if last_good else {},
        final_trafo_loading=last_good["trafo_loading"] if last_good else {},
        final_bus_vm=last_good["bus_vm"] if last_good else {},
        islanded_during_cascade=sorted(islanded_during, key=str),
        tripped_elements=tripped_all,
    )
