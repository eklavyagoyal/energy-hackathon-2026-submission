"""Named demo scenarios for the battery feature.

A scenario is a deterministic transformation of the loaded net that
manufactures a known, narratable grid state. Parameters are discovered
at build time and stored here (same pattern as the Phase 1 S1/S2 scales
in docs/03-phase1-engine.md section 3.3: discovered by search, stored in
config, never asserted as physics facts).

The demo scenario uses LOCALIZED stress (inject_local_stress) rather
than the grid-wide top-10 stress (logged as D61), because a localized
congestion pocket is what lets a single battery relieve a corridor
cleanly. center=115 /
radius=2 / scale=4.0 was discovered for case118 with the Case118Loader
ratings (30 percent base-loading target) and voltage lift: it keeps the
base case secure while turning a handful of N-1 contingencies (notably
the loss of line 89) into cascades that a battery at the top-scored bus
demonstrably prevents.

These bus indices are case118-specific by definition. On a real TSO
dataset, scenarios are defined against that dataset; nothing else in the
codebase imports these constants.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import pandapower as pp
import pandapower.topology as top

from src.engine.network import inject_local_stress, working_copy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    description: str
    center_bus: int | str | None = None
    radius: int = 0
    load_scale: float = 1.0
    global_scale: float = 1.0  # system-wide load multiplier applied AFTER any localized stress


SCENARIOS: dict[str, ScenarioSpec] = {
    "calm": ScenarioSpec(
        scenario_id="calm",
        description="Unstressed base case; secure, no cascades.",
    ),
    "demo_congestion": ScenarioSpec(
        scenario_id="demo_congestion",
        description=(
            "Localized congestion pocket around bus 115; base secure, "
            "loss of line 89 cascades, battery at the top-scored bus "
            "prevents it. Discovered at build time for case118."
        ),
        center_bus=115,
        radius=2,
        load_scale=4.0,
    ),
    "demo_overload": ScenarioSpec(
        scenario_id="demo_overload",
        description=(
            "Evening-peak overload: the bus-115 congestion pocket PLUS a system-wide load "
            "peak pushes several lines over their thermal limits (base not secure). The agent "
            "redispatches generation back to a solver-verified secure base. Discovered at "
            "build time for case118."
        ),
        center_bus=115,
        radius=2,
        load_scale=4.0,
        global_scale=1.22,
    ),
}

DEFAULT_SCENARIO = "demo_congestion"


def scenario_is_compatible(net: "pp.pandapowerNet", scenario_id: str) -> bool:
    """True when the named scenario can be applied to this loaded net.

    The demo stress presets are intentionally case118-specific. Dataset
    loaders for PyPSA or TSO exports can still use `calm`; they should not
    inherit bus-115 stress just because that is the case118 demo default.
    """
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; known: {sorted(SCENARIOS)}"
        )
    spec = SCENARIOS[scenario_id]
    if spec.center_bus is None or spec.load_scale == 1.0:
        return True
    graph = top.create_nxgraph(net, respect_switches=True)
    return spec.center_bus in graph


def default_scenario_for_net(net: "pp.pandapowerNet") -> str:
    """Pick the startup scenario for the loaded dataset."""
    if scenario_is_compatible(net, DEFAULT_SCENARIO):
        return DEFAULT_SCENARIO
    logger.info(
        "default scenario %s is not compatible with dataset %s; using calm",
        DEFAULT_SCENARIO,
        net.get("grid_dataset", "unknown"),
    )
    return "calm"


def apply_scenario(net: "pp.pandapowerNet", scenario_id: str) -> "pp.pandapowerNet":
    """Return a fresh stressed net for the named scenario. The input net
    is never mutated."""
    if scenario_id not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; known: {sorted(SCENARIOS)}"
        )
    spec = SCENARIOS[scenario_id]
    if spec.center_bus is None or spec.load_scale == 1.0:
        out = working_copy(net)
    else:
        if not scenario_is_compatible(net, scenario_id):
            raise ValueError(
                f"scenario {scenario_id!r} is case118-specific: center bus "
                f"{spec.center_bus!r} is not present in dataset "
                f"{net.get('grid_dataset', 'unknown')!r}; use 'calm' or a "
                "dataset-specific scenario"
            )
        out = inject_local_stress(net, spec.center_bus, spec.radius, spec.load_scale)
    if spec.global_scale != 1.0 and len(out.load):
        out.load["p_mw"] = out.load["p_mw"] * spec.global_scale
        out.load["q_mvar"] = out.load["q_mvar"] * spec.global_scale
    return out
