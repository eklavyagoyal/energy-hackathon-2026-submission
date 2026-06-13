"""SimulationTrace and TimeStepRecord (U9): the serializable record of a time-stepped run.

Each step carries everything the frontend timeline scrubber needs: the timestamp, the events that
fired, the solver base state, the worst contingency, the agent's action, the commit status, the
solver-verified post-action state, the narration, and per-edge loadings for the one-line render.
Serialization goes through the engine's to_jsonable (NaN/inf -> null, numpy -> native).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.engine.actions import to_jsonable


@dataclass
class TimeStepRecord:
    t: int
    timestamp: str
    events: list                 # [{kind, ...}] firing this step
    baseline_state: dict         # solver base-case summary before action
    worst_contingency: dict | None
    agent_action: dict           # the proposed Action (dict)
    commit_status: str           # applied | rejected_infeasible | applied_unverified | noop
    verification: dict           # the VerificationReport (verified flag, method, measured deltas)
    verified_state: dict         # solver state after the action (the verify-before-commit result)
    narration: list              # 3 sentences
    edge_loadings: dict = field(default_factory=dict)   # {line_<i>/trafo_<i>: loading_pct}
    islanded_buses: list = field(default_factory=list)
    n_insecure: int = 0                  # CRITICAL/HIGH line+trafo N-1 (operational, load-driven)
    standing_blackout_risk: bool = False  # single-slack loss is a constant blackout on case118


@dataclass
class SimulationTrace:
    profile_id: str
    event_scenario: str
    agent_mode: str
    horizon_steps: int
    step_minutes: int
    seed: int
    steps: list = field(default_factory=list)
    scenario: str | None = None
    scenario_title: str | None = None
    scenario_context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return to_jsonable(self)
