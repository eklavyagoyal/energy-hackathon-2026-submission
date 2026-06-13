"""Runtime settings for the Warden backend.

Environment-variable driven, no extra dependency. Every battery and
loader knob lives here so swapping case118 for a real TSO dataset is a
config change, not a code change (see docs/battery-feature.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """All values resolvable from the environment with safe defaults."""

    # Dataset selection. "case118" is the development default; "tso_real"
    # switches to TSORealLoader and "pypsa_eur" to PyPSAEurLoader, both
    # reading GRID_DATA_PATH. No module other than src/grid/loader.py
    # interprets these values.
    grid_dataset: str = "case118"
    grid_data_path: str | None = None

    # Battery recommendation knobs.
    battery_verification_parallel: bool = True
    battery_default_topk: int = 3
    battery_max_topk: int = 10
    battery_default_p_mw: float = 10.0
    battery_default_e_mwh: float = 40.0

    # LLM narration. The narrator is optional: without an API key the
    # deterministic template narrator is used (same structured inputs,
    # zero invented numbers either way). Battery narration is explicitly
    # opt-in so endpoint latency never depends on a live model by accident.
    anthropic_api_key: str | None = None
    narration_model: str = "claude-sonnet-4-6"
    battery_narration_mode: str = "template"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            grid_dataset=os.getenv("GRID_DATASET", "case118"),
            grid_data_path=os.getenv("GRID_DATA_PATH"),
            battery_verification_parallel=_env_bool(
                "BATTERY_VERIFICATION_PARALLEL", True
            ),
            battery_default_topk=_env_int("BATTERY_DEFAULT_TOPK", 3),
            battery_max_topk=_env_int("BATTERY_MAX_TOPK", 10),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            narration_model=os.getenv("BATTERY_NARRATION_MODEL", "claude-sonnet-4-6"),
            battery_narration_mode=os.getenv("BATTERY_NARRATION_MODE", "template"),
        )


def get_settings() -> Settings:
    return Settings.from_env()
