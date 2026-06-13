"""Battery Storage Location Recommendation (bolt-on).

Thesis, same rule as Phase 1: the LLM never computes physics. The
engine aggregates structured contingency results into a score, then
VERIFIES the top candidates by re-running the same N-1 cascade pipeline
with a virtual battery at the candidate bus.

Pitch sentence: "Every recommended battery location is verified by
re-running the cascade with the battery in place."

Boundary rules:
- this package consumes src.engine OUTPUTS (ContingencyResult records)
  and calls its public pipeline (run_contingency_sweep); it never
  duplicates physics and never reaches into engine internals
- src.engine never imports this package (bolt-on, not a dependency)
- the LLM (src/battery/narration.py) only narrates numbers the solver
  already produced
"""
