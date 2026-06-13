"""Warden Phase 1 engine core (pure Python + pandapower, no LLM).

Provenance note (logged as D49 in docs/decisions.md): when the
feature/battery-recommendations branch started, main contained the
planning docs only; no Phase 1 code existed. This package is the
minimal doc-canonical core of docs/03-phase1-engine.md needed by the
battery bolt-on: constants, network helpers, preflight (slack guard +
islanding), the cascade loop, CSS severity, the contingency set builder
and the full N-1 sweep. The Screener slot, tighten-and-verify
remediation and the agent loop belong to the main Phase 1 build and are
deliberately NOT part of this branch.

Boundary rule (docs/02-architecture.md section 8): nothing in this
package may import from src.battery; the battery feature consumes this
package's outputs only.
"""

from src.engine.cascade import CascadeOutcome, TraceStep, run_cascade
from src.engine.scan import (
    ContingencyResult,
    Outage,
    analyze_contingency,
    build_contingency_set,
    run_contingency_sweep,
)

__all__ = [
    "CascadeOutcome",
    "TraceStep",
    "run_cascade",
    "ContingencyResult",
    "Outage",
    "analyze_contingency",
    "build_contingency_set",
    "run_contingency_sweep",
]
