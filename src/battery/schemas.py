"""Pydantic models for the battery recommendation API.

Field names of the request/response bodies follow the battery-feature
contract (docs/battery-feature.md); additive fields beyond the minimum
contract are allowed per the doc 07 additive-only rule.

Bus references are typed int | str everywhere (dataset portability
rule 2). pydantic v2 smart unions keep ints as ints and strings as
strings, so a net with string identifiers round-trips them untouched.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, model_validator

BusRef = Union[int, str]

Verdict = Literal["RECOMMENDED", "MIXED", "NO_IMPACT", "NOT_RECOMMENDED"]


class ScoreWeights(BaseModel):
    """Scoring weights. Defaults are the documented starting point and
    deliberately exposed on the API so they can be stress-tested in
    Q&A; they are normalized to sum to 1 before use."""

    congestion: float = Field(default=0.35, ge=0.0)
    voltage: float = Field(default=0.25, ge=0.0)
    cascade: float = Field(default=0.25, ge=0.0)
    severity: float = Field(default=0.15, ge=0.0)

    @model_validator(mode="after")
    def _at_least_one_positive(self) -> "ScoreWeights":
        if (self.congestion + self.voltage + self.cascade + self.severity) <= 0.0:
            raise ValueError("at least one scoring weight must be positive")
        return self

    def normalized(self) -> "ScoreWeights":
        total = self.congestion + self.voltage + self.cascade + self.severity
        return ScoreWeights(
            congestion=self.congestion / total,
            voltage=self.voltage / total,
            cascade=self.cascade / total,
            severity=self.severity / total,
        )


class ScoreBreakdown(BaseModel):
    """The four normalized components, each in [0, 1]."""

    congestion: float
    voltage: float
    cascade: float
    severity: float


class BusScoreContext(BaseModel):
    """Solver-produced facts behind a bus score; the narration layer
    may only quote from here (and from VerificationResult)."""

    total_scenarios: int
    congestion_count: int
    voltage_count: int
    cascade_count: int
    worst_line: BusRef | None = None
    worst_line_loading_pct: float | None = None
    min_voltage_pu: float | None = None
    max_voltage_pu: float | None = None
    # The out-of-band voltage extreme actually seen at this bus (the
    # value furthest outside 0.95 to 1.05), with its direction, so
    # narration reports overvoltage as a rise and undervoltage as a drop.
    worst_voltage_pu: float | None = None
    worst_voltage_dir: Literal["low", "high", "none"] = "none"
    severity_shed_sum_mw: float = 0.0


class BusScore(BaseModel):
    bus_idx: BusRef
    score: float
    score_breakdown: ScoreBreakdown
    context: BusScoreContext


class PerContingencyDelta(BaseModel):
    """One changed contingency in the with-battery counterfactual.
    Drives the split-screen 'with vs without battery' demo view."""

    contingency_id: str
    status_before: str
    status_after: str
    score_before: float
    score_after: float
    load_shed_before_mw: float
    load_shed_after_mw: float


class VerificationResult(BaseModel):
    """Solver-verified counterfactual impact of one battery placement.

    Produced exclusively by re-running the full N-1 cascade sweep
    (same pipeline as the baseline) with a virtual storage element at
    the bus. Honesty fields are first-class: scenarios_worsened and the
    NO_IMPACT / NOT_RECOMMENDED verdicts are reported, never dropped.
    """

    bus_idx: BusRef
    battery_p_mw: float
    battery_max_e_mwh: float
    n_scenarios: int
    scenarios_improved: int
    scenarios_unchanged: int
    scenarios_worsened: int
    cascades_prevented: int
    mw_load_shed_avoided: float
    avg_voltage_improvement: float
    max_loading_reduction: float
    verdict: Verdict
    per_contingency: list[PerContingencyDelta] = Field(default_factory=list)
    computation_time_ms: float = 0.0


class BatteryCandidate(BaseModel):
    bus_idx: BusRef
    bus_name: str | None = None
    score: float
    score_breakdown: ScoreBreakdown
    context: BusScoreContext
    verification: VerificationResult | None = None
    narration: str | None = None
    narration_source: Literal["llm", "template", "none"] = "none"


class RecommendationRequest(BaseModel):
    top_k: int = Field(default=3, ge=1)
    weights: ScoreWeights = Field(default_factory=ScoreWeights)
    battery_capacity_mw: float = Field(default=10.0, gt=0.0)
    battery_energy_mwh: float = Field(default=40.0, gt=0.0)
    verify: bool = True
    # Time-aware siting (U16). horizon_steps=1 (default) is the single-snapshot recommendation,
    # unchanged. horizon_steps>1 sites the battery for the PEAK load of a synthetic horizon (when
    # storage value is highest), scoring and verifying against that peak-stressed net.
    horizon_steps: int = Field(default=1, ge=1, le=96)


class RecommendationResponse(BaseModel):
    candidates: list[BatteryCandidate]
    computation_time_ms: float
    baseline_cached: bool
    # Honesty and reproducibility surface (additive fields):
    n_scenarios: int
    n_buses_scored: int
    excluded_slack_buses: list[BusRef]
    weights_used: ScoreWeights
    opf_available: bool
    dispatch_model: str
    # Time-aware siting context (U16); defaults describe the single-snapshot case so existing
    # responses are unchanged.
    horizon_steps: int = 1
    peak_load_factor: float = 1.0


class VerifyRequest(BaseModel):
    bus_idx: BusRef
    battery_capacity_mw: float = Field(default=10.0, gt=0.0)
    battery_energy_mwh: float = Field(default=40.0, gt=0.0)
