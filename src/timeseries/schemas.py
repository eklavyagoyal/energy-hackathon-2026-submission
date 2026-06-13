"""Request/response models for the /timeseries API (U10).

Kept deliberately small: the run response is the SimulationTrace dict (already JSON-safe via
to_jsonable), so we validate only the request and the lightweight discovery payload.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

AGENT_MODES = ("opf", "llm", "greedy")
PROFILES = ("synthetic_24h", "calm", "stress_demo")
EVENT_SCENARIOS = ("calm", "default", "stress_demo")
GEOGRAPHIC_SCENARIO_NAMES = ("dunkelflaute", "solar_peak_south", "heatwave", "custom")


class TimeseriesRunRequest(BaseModel):
    """One time-stepped run. agent_mode picks the decision policy; replay (the default) forces the
    deterministic template narrator so a run is reproducible and free. narrate=True swaps in real LLM
    narration per insecure step for a recorded demo (slower, costs tokens)."""

    profile_id: str = "synthetic_24h"
    event_scenario: str = "stress_demo"
    agent_mode: str = "opf"
    horizon_steps: int = Field(default=24, ge=1, le=96)
    step_minutes: int = Field(default=60, ge=5, le=180)
    seed: int = 42
    start_scenario: str | None = None  # base scenario to seed the net; default derived from events
    narrate: bool = False
    scenario: str | None = None  # geographic scenario name; overrides profile_id/event_scenario when set


class TimeseriesProfilesResponse(BaseModel):
    profiles: list[str]
    event_scenarios: list[str]
    agent_modes: list[str]
