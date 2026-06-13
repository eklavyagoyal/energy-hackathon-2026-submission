"""Per-contingency preflight: slack guard + islanding.

Doc reference: docs/03-phase1-engine.md section 4. The order is
mandatory and load-bearing:

  STEP 1  slack guard, runs FIRST, NO solver call on the blackout path
  STEP 2  islanding via pandapower.topology.unsupplied_buses, BEFORE
          any solve; de-energize unsupplied elements and count the shed

Multi-slack handling (dataset portability rule 5): the guard asks "does
ANY in-service ext_grid remain", never "is ext_grid 0 alive". Losing
one of several slacks is an islanding question, not a blackout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandapower as pp
import pandapower.topology as top

from src.engine.network import native_index


@dataclass
class Preflight:
    slack_lost: bool
    islanded_buses: list = field(default_factory=list)
    islanded_load_mw: float = 0.0


def slack_lost(net: "pp.pandapowerNet") -> bool:
    """STEP 1, the slack guard: true when no in-service ext_grid is
    left, or every remaining in-service ext_grid sits on an
    out-of-service bus. Pure topology, no solver call."""
    es = net.ext_grid
    live = es[es.in_service]
    if len(live) == 0:
        return True
    in_service_buses = set(net.bus.index[net.bus.in_service])
    return not any(b in in_service_buses for b in live.bus.values)


def unsupplied(net: "pp.pandapowerNet") -> set:
    """Buses with no path to an in-service ext_grid.

    respect_switches=True is passed explicitly: real TSO datasets carry
    active switch elements that change topology dynamically (dataset
    portability rule 9). case118 has no switches, so this is a no-op
    there.
    """
    return set(top.unsupplied_buses(net, respect_switches=True))


def deenergize_and_count(net: "pp.pandapowerNet", buses: set) -> float:
    """Set load, sgen, gen and storage at the given buses out of
    service IN PLACE (callers pass a working copy, never the live net)
    and return the de-energized load p_mw.

    Doc 03 names loads and sgens; gens and storages are de-energized
    too so a dead island never carries an in-service injection into the
    next solve. storage matters for the battery bolt-on: a virtual
    battery whose bus islands is disconnected with its island (an
    islanded battery cannot form a grid in this model; logged as D55).
    """
    if not buses:
        return 0.0
    shed = 0.0
    for etype in ("load", "sgen", "gen", "storage"):
        table = getattr(net, etype, None)
        if table is None or len(table) == 0:
            continue
        mask = table.bus.isin(buses) & table.in_service
        if not mask.any():
            continue
        if etype == "load":
            scaling = table.scaling if "scaling" in table.columns else 1.0
            shed += float((table.p_mw * scaling)[mask].sum())
        net[etype].loc[mask, "in_service"] = False
    return shed


def all_buses_unsupplied(net: "pp.pandapowerNet", unsupplied_set: set) -> bool:
    """True when every in-service bus is unsupplied (degenerate
    topology = FULL_BLACKOUT). Compared against the in-service bus set,
    not net.bus.index: TSO datasets may carry permanently
    out-of-service buses that must not mask a total blackout."""
    in_service_buses = set(net.bus.index[net.bus.in_service])
    return len(in_service_buses) > 0 and in_service_buses <= unsupplied_set


def preflight_islanding(net: "pp.pandapowerNet") -> tuple[Preflight, set]:
    """STEP 2 wrapper: find unsupplied buses, de-energize, build the
    Preflight record. Returns (preflight, unsupplied_set). Mutates the
    working copy (de-energization), never called on the live net."""
    unsup = unsupplied(net)
    shed = deenergize_and_count(net, unsup)
    pf = Preflight(
        slack_lost=False,
        islanded_buses=sorted((native_index(b) for b in unsup), key=str),
        islanded_load_mw=shed,
    )
    return pf, unsup
