"""Shared action / verification / baseline types for remediation and the agent.

The Phase 1 engine produces ContingencyResult records (src/engine/scan.py). The remediation
layer and the agent (Phase 0 unification) need a small vocabulary of additional records: a
proposed Action, the VerificationReport that a solver re-run produces about it, and a
BaselineResult row for the comparison table. These are plain dataclasses (no pydantic in the
engine); the API mirrors their shapes. Ported from the warden build's app/engine/types.py and
adapted to the src naming and the int | str index contract.

Action.changes are plain dicts with literal keys "from"/"to" (from is a Python keyword), so they
stay JSON-native and survive the str | int index portability rule unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any


@dataclass
class Action:
    """A proposed remediation. type: redispatch | load_shed | noop. source: opf | agent | greedy | operator."""
    action_id: str
    type: str
    changes: list  # list[{"etype","index","field","from","to"}]
    source: str
    estimated_cost_delta: float
    rationale: str = ""


@dataclass
class VerificationReport:
    verified: bool
    method: str  # e.g. "ac_cascade_rescan"
    contingency_ids: list
    before: list  # list[{"contingency_id","status","score"}], "base" pseudo-entry first
    after: list
    deltas: dict  # {violations_resolved, load_shed_avoided_mw, worst_score_before, worst_score_after}
    committed: bool = False


@dataclass
class BaselineResult:
    policy: str  # greedy | opf | agent
    secure_after: bool
    violations_resolved: int
    load_shed_mw: float
    worst_cascade_depth_after: int
    redispatch_cost: float
    wall_ms: float
    explanation_available: bool
    trace: list = field(default_factory=list)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / enums / numpy scalars to JSON-native values.
    NaN and inf become null so the result is strict-JSON serializable."""
    import numpy as np

    if obj is None:
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bool) or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    return str(obj)
