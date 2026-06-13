"""Deterministic grid events for the time-stepped simulator (U7).

An EventStream yields zero or more events per timestep. Events are deterministic given the scenario
(no RNG needed for the prepared scenarios). Temporary line outages auto-clear; maintenance windows
are known in advance to the agent; load spikes scale a bus's load for a window. ev.apply(net)
mutates the working net in place (the simulator owns the net); cleanup restores expired outages.

Element references are kept as native indices (int | str), never int-coerced, so the stream works
on string-bus datasets unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LineOutage:
    line_index: Any
    duration_steps: int
    kind: str = "line_outage"

    def apply(self, net) -> None:
        if self.line_index in net.line.index:
            net.line.loc[self.line_index, "in_service"] = False

    def restore(self, net) -> None:
        if self.line_index in net.line.index:
            net.line.loc[self.line_index, "in_service"] = True

    def to_dict(self) -> dict:
        return {"kind": self.kind, "line_index": self.line_index, "duration_steps": self.duration_steps}


@dataclass
class TrafoOutage:
    trafo_index: Any
    duration_steps: int
    kind: str = "trafo_outage"

    def apply(self, net) -> None:
        if self.trafo_index in net.trafo.index:
            net.trafo.loc[self.trafo_index, "in_service"] = False

    def restore(self, net) -> None:
        if self.trafo_index in net.trafo.index:
            net.trafo.loc[self.trafo_index, "in_service"] = True

    def to_dict(self) -> dict:
        return {"kind": self.kind, "trafo_index": self.trafo_index, "duration_steps": self.duration_steps}


@dataclass
class Maintenance:
    line_index: Any
    start: int
    end: int
    kind: str = "maintenance"

    def active_at(self, t: int) -> bool:
        return self.start <= t < self.end

    def apply(self, net) -> None:
        if self.line_index in net.line.index:
            net.line.loc[self.line_index, "in_service"] = False

    def restore(self, net) -> None:
        if self.line_index in net.line.index:
            net.line.loc[self.line_index, "in_service"] = True

    def to_dict(self) -> dict:
        return {"kind": self.kind, "line_index": self.line_index, "start": self.start, "end": self.end}


@dataclass
class LoadSpike:
    bus_index: Any
    factor: float
    duration_steps: int
    kind: str = "load_spike"
    _orig: dict = field(default_factory=dict, repr=False)

    def apply(self, net) -> None:
        mask = net.load.bus == self.bus_index
        for idx in net.load.index[mask]:
            self._orig[idx] = float(net.load.at[idx, "p_mw"])
            net.load.at[idx, "p_mw"] = self._orig[idx] * self.factor

    def restore(self, net) -> None:
        for idx, val in self._orig.items():
            if idx in net.load.index:
                net.load.at[idx, "p_mw"] = val
        self._orig = {}

    def to_dict(self) -> dict:
        return {"kind": self.kind, "bus_index": self.bus_index, "factor": self.factor,
                "duration_steps": self.duration_steps}


class EventStream:
    """Holds a schedule of events and produces those active/firing at each step.

    schedule: dict[int, list[event]] keyed by the step the event FIRES. Temporary events
    (LineOutage, LoadSpike) carry a duration and are auto-restored by cleanup() when expired.
    Maintenance is applied for its [start, end) window. All deterministic.
    """

    def __init__(self, schedule: dict | None = None):
        self.schedule: dict = schedule or {}
        self._active: list = []  # (event, expires_at_step)

    def events_at(self, t: int) -> list:
        """Fire this step's scheduled events; return the events firing now (for the trace)."""
        firing = list(self.schedule.get(t, []))
        for ev in firing:
            dur = getattr(ev, "duration_steps", None)
            if dur is not None:
                self._active.append((ev, t + dur))
        return firing

    def maintenance_windows(self) -> list:
        """All scheduled Maintenance events, so the agent can see them in advance."""
        out = []
        for evs in self.schedule.values():
            out.extend(ev for ev in evs if isinstance(ev, Maintenance))
        return out

    def cleanup(self, t: int, net) -> list:
        """Restore temporary events whose window has expired at the end of step t."""
        restored = []
        still = []
        for ev, expires in self._active:
            if t + 1 >= expires:
                ev.restore(net)
                restored.append(ev)
            else:
                still.append((ev, expires))
        self._active = still
        return restored


def build_event_scenario(name: str, net) -> EventStream:
    """Prepared, deterministic event scenarios. 'stress_demo' has a known cascade-risk window;
    'calm' has none. Bus/line indices are read from the net (no hardcoded assumptions on a dataset
    that lacks them)."""
    if name == "calm":
        return EventStream({})
    if name == "stress_demo":
        # A known sequence that opens a cascade-risk window mid-horizon. Indices chosen from the
        # net's own tables; if a chosen line is absent (other datasets), the event is a safe no-op.
        line_ids = list(net.line.index)
        load_buses = list(net.load.bus.unique())
        sched: dict = {}
        if len(line_ids) > 7:
            sched[4] = [LineOutage(line_ids[6], duration_steps=4)]            # corridor line out for 4h
        if load_buses:
            sched[8] = [LoadSpike(load_buses[0], factor=1.4, duration_steps=3)]  # evening demand jump
        if len(line_ids) > 90:
            sched[12] = [Maintenance(line_ids[89], start=12, end=16)]         # planned maintenance window
        return EventStream(sched)
    if name == "default":
        line_ids = list(net.line.index)
        sched = {6: [LineOutage(line_ids[6], duration_steps=3)]} if len(line_ids) > 7 else {}
        return EventStream(sched)
    raise KeyError(f"unknown event scenario {name!r}; known: calm, default, stress_demo")
