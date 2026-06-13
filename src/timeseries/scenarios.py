"""Geographic PyPSA-Eur stress scenarios for the time-stepped simulator.

The scenarios in this module are deterministic named runs for geographically anchored PyPSA-Eur
networks. They select buses, lines, generators, and transformers from network metadata (coordinates,
carrier/type strings, ratings, and solved flows), never from hardcoded element IDs.

If the active net has no usable coordinates (case118, or a stripped export), the same scenario names
fall back to role-based choices and log a warning so the demo does not crash while making the loss of
geographic meaning explicit.
"""
from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandapower as pp

from src.engine.network import native_index, working_copy
from src.engine.scan import element_name
from src.timeseries.events import EventStream, LineOutage, TrafoOutage
from src.timeseries.profiles import apply_profile, capture_base

logger = logging.getLogger(__name__)

_WIND_CF = (0.05, 0.15)
_SOLAR_PEAK_CF = (0.85, 0.95)


def _value_at(values: list[float], t: int) -> float:
    if not values:
        return 1.0
    return float(values[min(t, len(values) - 1)])


@dataclass
class LoadShapeProfile:
    """Per-load multiplier profile derived from captured base load, not cumulative state."""

    name: str
    default: list[float]
    by_load: dict[Any, list[float]] = field(default_factory=dict)
    kind: str = "load"

    def at(self, t: int) -> float:
        return _value_at(self.default, t)

    def __len__(self) -> int:
        return len(self.default)

    def apply_to_net(self, net, base: dict, t: int) -> None:
        if base.get("load_p") is None:
            return
        for idx in net.load.index:
            mult = _value_at(self.by_load.get(idx, self.default), t)
            net.load.at[idx, "p_mw"] = float(base["load_p"].at[idx]) * mult
            if base.get("load_q") is not None and idx in base["load_q"].index:
                net.load.at[idx, "q_mvar"] = float(base["load_q"].at[idx]) * mult


@dataclass
class GenerationDispatchProfile:
    """Per-generator absolute or multiplier profile derived from captured base generation."""

    name: str
    default: list[float]
    absolute_mw_by_gen: dict[Any, list[float]] = field(default_factory=dict)
    multiplier_by_gen: dict[Any, list[float]] = field(default_factory=dict)
    kind: str = "gen"

    def at(self, t: int) -> float:
        return _value_at(self.default, t)

    def __len__(self) -> int:
        return len(self.default)

    def apply_to_net(self, net, base: dict, t: int) -> None:
        if base.get("gen_p") is None:
            return
        default_mult = self.at(t)
        for idx in net.gen.index:
            if idx in self.absolute_mw_by_gen:
                net.gen.at[idx, "p_mw"] = _value_at(self.absolute_mw_by_gen[idx], t)
            else:
                mult = _value_at(self.multiplier_by_gen.get(idx, self.default), t)
                if mult != 1.0 or default_mult != 1.0:
                    net.gen.at[idx, "p_mw"] = float(base["gen_p"].at[idx]) * mult


@dataclass(frozen=True)
class GeographicScenario:
    """A named deterministic stress sequence for a geographically anchored network."""

    name: str
    title: str
    description: str
    duration_hours: int
    target_region_hint: str
    load_profile: dict[str, Callable[[int], float]] = field(default_factory=dict)
    generation_profile: dict[str, Callable[[int], float]] = field(default_factory=dict)
    compatible_networks: tuple[str, ...] = ("pypsa_eur_*",)
    fallback_available: bool = True
    region_highlight: dict = field(default_factory=dict)

    def prepare(self, net, seed: int = 42):
        """Return a configured working copy with static preconditions applied."""
        work = working_copy(net)
        work["active_geographic_scenario"] = self.name
        if self.name == "heatwave":
            _apply_heatwave_derating(work)
        return work

    def events_at(self, _t: int) -> list:
        """Template scenarios need a prepared net before events can be selected."""
        return []


@dataclass
class PreparedGeographicScenario:
    scenario: GeographicScenario
    net: Any
    load_profile: LoadShapeProfile
    gen_profile: GenerationDispatchProfile
    event_stream: EventStream
    metadata: dict

    def events_at(self, t: int) -> list:
        return self.event_stream.events_at(t)


def _gaussian(hour: float, center: float, width: float) -> float:
    return math.exp(-0.5 * ((hour - center) / width) ** 2)


def _winter_load(hours: int) -> list[float]:
    out = []
    for t in range(hours):
        hour = t % 24
        out.append(1.12 + 0.20 * _gaussian(hour, 7.0, 1.7) + 0.24 * _gaussian(hour, 18.0, 2.1))
    return out


def _bavaria_load(hours: int) -> list[float]:
    out = []
    for t in range(hours):
        hour = t % 24
        midday = 0.28 if 11 <= hour <= 15 else 0.0
        shoulder = 0.10 * _gaussian(hour, 13.0, 3.5)
        out.append(1.02 + midday + shoulder)
    return out


def _heatwave_load(hours: int) -> list[float]:
    return [1.25 for _ in range(hours)]


def _solar_shape(hours: int, peak_cf: float) -> list[float]:
    values = []
    for t in range(hours):
        hour = t % 24
        if 11 <= hour <= 15:
            values.append(peak_cf)
        elif 8 <= hour < 11 or 15 < hour <= 18:
            values.append(max(0.05, peak_cf * 0.45))
        else:
            values.append(0.0)
    return values


def _near_zero_solar(hours: int) -> list[float]:
    return [0.02 if 9 <= (t % 24) <= 15 else 0.0 for t in range(hours)]


def _has_geography(net) -> bool:
    if "x" not in net.bus.columns or "y" not in net.bus.columns:
        return False
    coords = net.bus[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    return bool(coords.dropna().shape[0] >= 2)


def _warn(warnings: list[str], message: str) -> None:
    warnings.append(message)
    logger.warning(message)


def _bus_xy(net, bus) -> tuple[float, float] | None:
    if bus not in net.bus.index or "x" not in net.bus.columns or "y" not in net.bus.columns:
        return None
    x = pd.to_numeric(pd.Series([net.bus.at[bus, "x"]]), errors="coerce").iloc[0]
    y = pd.to_numeric(pd.Series([net.bus.at[bus, "y"]]), errors="coerce").iloc[0]
    if pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y)


def _line_voltage_kv(net, idx) -> float:
    row = net.line.loc[idx]
    if "v_nom" in net.line.columns and not pd.isna(row.get("v_nom")):
        return float(row.get("v_nom"))
    buses = [row.from_bus, row.to_bus]
    vals = [float(net.bus.at[b, "vn_kv"]) for b in buses if b in net.bus.index and "vn_kv" in net.bus.columns]
    return max(vals) if vals else 0.0


def _solved_line_flows(net) -> dict:
    work = working_copy(net)
    try:
        pp.runpp(work)
    except Exception:
        return {}
    flows: dict = {}
    if len(work.res_line):
        for idx in work.res_line.index:
            vals = []
            for col in ("p_from_mw", "p_to_mw"):
                if col in work.res_line.columns:
                    val = work.res_line.at[idx, col]
                    if pd.notna(val):
                        vals.append(abs(float(val)))
            flows[idx] = max(vals) if vals else 0.0
    return flows


def _pick_line(net, predicate: Callable[[Any, tuple[float, float] | None, tuple[float, float] | None], bool]):
    flows = _solved_line_flows(net)
    candidates = []
    for idx, row in net.line[net.line.in_service].iterrows():
        a = _bus_xy(net, row.from_bus)
        b = _bus_xy(net, row.to_bus)
        if predicate(idx, a, b):
            rating = float(row.get("max_i_ka", 0.0) or 0.0)
            candidates.append((flows.get(idx, rating), str(native_index(idx)), idx))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _pick_high_flow_line(net):
    flows = _solved_line_flows(net)
    candidates = []
    for idx, row in net.line[net.line.in_service].iterrows():
        rating = float(row.get("max_i_ka", 0.0) or 0.0)
        candidates.append((flows.get(idx, rating), str(native_index(idx)), idx))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _north_south_line(net):
    def pred(idx, a, b) -> bool:
        if a is None or b is None:
            return False
        crosses = (a[1] > 52.0 and b[1] < 49.0) or (b[1] > 52.0 and a[1] < 49.0)
        return crosses and _line_voltage_kv(net, idx) >= 350.0

    return _pick_line(net, pred)


def _southern_line(net):
    def pred(idx, a, b) -> bool:
        if a is None or b is None:
            return False
        southern = a[1] < 49.0 and b[1] < 49.0
        touches_bavaria = a[0] > 10.0 or b[0] > 10.0
        return southern and touches_bavaria and _line_voltage_kv(net, idx) >= 350.0

    return _pick_line(net, pred)


def _pick_trafo(net):
    if len(net.trafo) == 0:
        return None
    work = working_copy(net)
    try:
        pp.runpp(work)
        if len(work.res_trafo):
            loading = work.res_trafo.loading_percent.dropna()
            if len(loading):
                return loading.sort_values(ascending=False, kind="mergesort").index[0]
    except Exception:
        pass
    return net.trafo.sort_values("sn_mva", ascending=False, kind="mergesort").index[0]


def _tech_string(row) -> str:
    parts = []
    for col in ("carrier", "type", "name", "pypsa_name"):
        if col in row and not pd.isna(row[col]):
            parts.append(str(row[col]))
    return " ".join(parts).lower()


def _is_wind(row) -> bool:
    text = _tech_string(row)
    return "wind" in text or "onwind" in text or "offwind" in text


def _is_solar(row) -> bool:
    text = _tech_string(row)
    return "solar" in text or "pv" in text


def _max_p(row, fallback: float) -> float:
    for col in ("max_p_mw", "p_nom"):
        if col in row and not pd.isna(row[col]):
            return max(0.0, float(row[col]))
    return max(0.0, fallback)


def _bavaria_buses(net) -> set:
    out = set()
    for idx in net.bus.index:
        xy = _bus_xy(net, idx)
        if xy is not None and xy[1] < 49.0 and xy[0] > 10.0:
            out.add(idx)
    return out


def _top_load_buses(net, n: int = 3) -> set:
    if len(net.load) == 0:
        return set()
    top = net.load.sort_values("p_mw", ascending=False, kind="mergesort").head(max(1, min(n, len(net.load))))
    return set(top.bus.tolist())


def _loads_on_buses(net, buses: set) -> list:
    if not buses or len(net.load) == 0:
        return []
    return list(net.load.index[net.load.bus.isin(buses)])


def _apply_heatwave_derating(net) -> None:
    if len(net.line) == 0:
        return
    if "max_loading_percent" in net.line.columns:
        limits = pd.to_numeric(net.line["max_loading_percent"], errors="coerce")
        mask = limits.notna()
        net.line.loc[mask, "max_loading_percent"] = limits[mask] * 0.85
    if "max_i_ka" in net.line.columns:
        max_i = pd.to_numeric(net.line["max_i_ka"], errors="coerce")
        mask = max_i.notna()
        net.line.loc[mask, "max_i_ka"] = max_i[mask] * 0.85


def _build_profiles(scenario: GeographicScenario, net, seed: int, warnings: list[str]):
    rng = random.Random(seed)
    h = scenario.duration_hours

    if scenario.name == "dunkelflaute":
        load = LoadShapeProfile("dunkelflaute_winter_load", _winter_load(h))
        gen = GenerationDispatchProfile("dunkelflaute_low_renewables", [1.0] * h)
        wind_seen = solar_seen = False
        for idx, row in net.gen.iterrows():
            base = float(row.get("p_mw", 0.0) or 0.0)
            cap = _max_p(row, base)
            if _is_wind(row):
                wind_seen = True
                cf = rng.uniform(*_WIND_CF)
                gen.absolute_mw_by_gen[idx] = [round(cap * cf, 3)] * h
            elif _is_solar(row):
                solar_seen = True
                gen.absolute_mw_by_gen[idx] = [round(cap * cf, 3) for cf in _near_zero_solar(h)]
        if not wind_seen:
            _warn(warnings, "dunkelflaute: no wind generators found by carrier/type/name; wind curtailment skipped")
        if not solar_seen:
            _warn(warnings, "dunkelflaute: no solar generators found by carrier/type/name; solar curtailment skipped")
        return load, gen

    if scenario.name == "solar_peak_south":
        default = [1.0] * h
        by_load: dict[Any, list[float]] = {}
        buses = _bavaria_buses(net)
        if not buses:
            _warn(warnings, "solar_peak_south: no Bavaria coordinate match; falling back to highest-load buses")
            buses = _top_load_buses(net)
        for idx in _loads_on_buses(net, buses):
            by_load[idx] = _bavaria_load(h)
        load = LoadShapeProfile("solar_peak_south_industrial_load", default, by_load=by_load)
        gen = GenerationDispatchProfile("solar_peak_south_pv", [1.0] * h)
        solar_seen = False
        for idx, row in net.gen.iterrows():
            if _is_solar(row):
                solar_seen = True
                peak_cf = rng.uniform(*_SOLAR_PEAK_CF)
                cap = _max_p(row, float(row.get("p_mw", 0.0) or 0.0))
                gen.absolute_mw_by_gen[idx] = [round(cap * cf, 3) for cf in _solar_shape(h, peak_cf)]
        if not solar_seen:
            _warn(warnings, "solar_peak_south: no solar generators found by carrier/type/name; PV profile skipped")
        return load, gen

    if scenario.name == "heatwave":
        return (
            LoadShapeProfile("heatwave_125pct_load", _heatwave_load(h)),
            GenerationDispatchProfile("heatwave_gen_flat", [1.0] * h),
        )

    raise KeyError(f"unknown geographic scenario {scenario.name!r}")


def _event_probe_net(net, load_profile, gen_profile, t: int):
    probe = working_copy(net)
    base = capture_base(probe)
    apply_profile(probe, base, load_profile, gen_profile, t)
    return probe


def _build_events(scenario: GeographicScenario, net, load_profile, gen_profile, warnings: list[str]) -> tuple[EventStream, dict]:
    schedule: dict[int, list] = {}
    event_meta: dict = {}
    h = scenario.duration_hours

    if scenario.name == "dunkelflaute":
        hour = 14
        probe = _event_probe_net(net, load_profile, gen_profile, hour)
        line = _north_south_line(probe)
        if line is None:
            _warn(warnings, "dunkelflaute: no coordinate-valid north-south 380kV line; falling back to highest-flow line")
            line = _pick_high_flow_line(probe)
        if line is not None:
            schedule[hour] = [LineOutage(native_index(line), duration_steps=max(1, h - hour))]
            event_meta = _line_event_meta(probe, line, "north_south_380kv")

    elif scenario.name == "solar_peak_south":
        hour = 13
        probe = _event_probe_net(net, load_profile, gen_profile, hour)
        line = _southern_line(probe)
        if line is None:
            _warn(warnings, "solar_peak_south: no coordinate-valid southern 380kV line; falling back to highest-flow line")
            line = _pick_high_flow_line(probe)
        if line is not None:
            schedule[hour] = [LineOutage(native_index(line), duration_steps=max(1, h - hour))]
            event_meta = _line_event_meta(probe, line, "southern_380kv")

    elif scenario.name == "heatwave":
        hour = 6
        probe = _event_probe_net(net, load_profile, gen_profile, hour)
        trafo = _pick_trafo(probe)
        if trafo is not None:
            schedule[hour] = [TrafoOutage(native_index(trafo), duration_steps=max(1, h - hour))]
            event_meta = {"etype": "trafo", "index": native_index(trafo), "role": "major_substation",
                          "name": element_name(probe, "trafo", trafo)}
        else:
            _warn(warnings, "heatwave: no transformer available; falling back to highest-flow line outage")
            line = _pick_high_flow_line(probe)
            if line is not None:
                schedule[hour] = [LineOutage(native_index(line), duration_steps=max(1, h - hour))]
                event_meta = _line_event_meta(probe, line, "fallback_line")

    return EventStream(schedule), event_meta


def _line_event_meta(net, line, role: str) -> dict:
    row = net.line.loc[line]
    a = _bus_xy(net, row.from_bus)
    b = _bus_xy(net, row.to_bus)
    return {
        "etype": "line",
        "index": native_index(line),
        "role": role,
        "name": element_name(net, "line", line),
        "from_bus": native_index(row.from_bus),
        "to_bus": native_index(row.to_bus),
        "from_xy": a,
        "to_xy": b,
        "v_nom": _line_voltage_kv(net, line),
    }


GEOGRAPHIC_SCENARIOS: dict[str, GeographicScenario] = {
    "dunkelflaute": GeographicScenario(
        name="dunkelflaute",
        title="Winter Dunkelflaute",
        description="Cold winter day, wind drops to 5-15%, solar is near-zero, peak load, forced north-south outage.",
        duration_hours=24,
        target_region_hint="North Germany / North-South corridor",
        region_highlight={"kind": "north_south_corridor", "label": "North-South corridor",
                          "box": {"x": 0.34, "y": 0.08, "w": 0.30, "h": 0.72}},
    ),
    "solar_peak_south": GeographicScenario(
        name="solar_peak_south",
        title="Solar Peak South",
        description="Bavarian midday industrial peak with PV at 85-95% and a southern 380kV outage.",
        duration_hours=24,
        target_region_hint="Bavarian industrial cluster",
        region_highlight={"kind": "bavaria", "label": "Bavarian industrial cluster",
                          "box": {"x": 0.52, "y": 0.55, "w": 0.25, "h": 0.25}},
    ),
    "heatwave": GeographicScenario(
        name="heatwave",
        title="Heatwave Thermal Derating",
        description="Midday heatwave: load +25%, line thermal limits derated 15%, forced major-substation transformer outage.",
        duration_hours=12,
        target_region_hint="Germany-wide derated thermal corridor",
        region_highlight={"kind": "system_wide", "label": "System-wide heat derating",
                          "box": {"x": 0.22, "y": 0.20, "w": 0.56, "h": 0.56}},
    ),
}


def scenario_options() -> list[dict]:
    return [
        {
            "name": s.name,
            "title": s.title,
            "description": s.description,
            "duration_hours": s.duration_hours,
            "compatible_networks": list(s.compatible_networks),
            "fallback_available": s.fallback_available,
            "target_region_hint": s.target_region_hint,
            "region_highlight": s.region_highlight,
        }
        for s in GEOGRAPHIC_SCENARIOS.values()
    ]


def build_geographic_scenario_run(name: str, net, seed: int = 42) -> PreparedGeographicScenario:
    if name not in GEOGRAPHIC_SCENARIOS:
        raise KeyError(f"unknown geographic scenario {name!r}; known: {sorted(GEOGRAPHIC_SCENARIOS)}")

    scenario = GEOGRAPHIC_SCENARIOS[name]
    warnings: list[str] = []
    geographic = _has_geography(net)
    if not geographic:
        _warn(warnings, f"{name}: active network has no usable bus.x/bus.y coordinates; geographic context is lost")

    prepared_net = scenario.prepare(net, seed=seed)
    load_profile, gen_profile = _build_profiles(scenario, prepared_net, seed, warnings)
    event_stream, event_meta = _build_events(scenario, prepared_net, load_profile, gen_profile, warnings)

    metadata = {
        "name": scenario.name,
        "title": scenario.title,
        "description": scenario.description,
        "duration_hours": scenario.duration_hours,
        "target_region_hint": scenario.target_region_hint,
        "compatible_networks": list(scenario.compatible_networks),
        "fallback_available": scenario.fallback_available,
        "geographic_context": geographic,
        "warnings": warnings,
        "event_selection": event_meta,
        "region_highlight": scenario.region_highlight,
    }
    prepared_net["geographic_scenario"] = metadata
    return PreparedGeographicScenario(scenario, prepared_net, load_profile, gen_profile, event_stream, metadata)
