"""Load and generation profiles for the time-stepped simulator (U6).

A profile is a per-step multiplier sequence applied to the BASE element values, which are captured
once at the start of a run and never multiplied cumulatively (cumulative scaling is a classic drift
bug). synthetic() is a deterministic BIMODAL daily curve (a morning peak and a higher evening peak, an
overnight trough and a midday dip) with seeded noise; from_csv reads
a timestamp/scaling_factor table; from_entsoe reads data/entsoe/ if present, else warns and falls
back to synthetic. Resolution defaults to 24 one-hour steps (configurable via STEP_MINUTES).
"""
from __future__ import annotations

import csv
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_HOURS = 24


def _step_minutes() -> int:
    try:
        return int(os.getenv("STEP_MINUTES", "60"))
    except ValueError:
        return 60


@dataclass
class Profile:
    """A per-step multiplier sequence. multipliers[t] scales the base element values at step t."""
    multipliers: list
    name: str = "profile"
    kind: str = "load"  # "load" | "gen"

    def at(self, t: int) -> float:
        if not self.multipliers:
            return 1.0
        return float(self.multipliers[min(t, len(self.multipliers) - 1)])

    def __len__(self) -> int:
        return len(self.multipliers)

    @classmethod
    def flat(cls, hours: int = DEFAULT_HOURS, value: float = 1.0, name: str = "flat", kind: str = "gen") -> "Profile":
        return cls([float(value)] * hours, name=name, kind=kind)

    @classmethod
    def synthetic(cls, hours: int = DEFAULT_HOURS, seed: int = 42, amplitude: float = 0.26,
                  noise: float = 0.02, peak_hour: int = 19, name: str = "synthetic", kind: str = "load") -> "Profile":
        """Realistic BIMODAL daily load curve: a morning peak (~08:00) and a higher evening peak
        (around peak_hour, default 19:00), an overnight trough (~03:00-04:00) and a midday dip between
        the two peaks. Real demand is bimodal; a single afternoon hump (the old cos() shape) is the
        classic synthetic-load tell and put the heaviest load at the wrong hours. Mean multiplier is
        held near 1.0 so a scenario's base load level is preserved, and the curve is clamped so the
        evening peak never blows past the grid's convergence headroom. Deterministic for a fixed seed.
        """
        rng = random.Random(seed)

        def _circ(hour: float, center: float) -> float:
            d = abs(hour - center)
            return min(d, 24.0 - d)  # circular distance so the evening tail wraps past midnight

        mults = []
        for t in range(hours):
            hour = (t * _step_minutes() / 60.0) % 24
            morning = math.exp(-0.5 * (_circ(hour, 8.0) / 2.0) ** 2)
            evening = math.exp(-0.5 * (_circ(hour, float(peak_hour)) / 2.6) ** 2)
            # floor (overnight base) + two demand humps, evening higher than morning
            shape = 0.80 + amplitude * (0.80 * morning + 1.0 * evening)
            mults.append(max(0.05, shape + rng.gauss(0.0, noise)))
        return cls(mults, name=name, kind=kind)

    @classmethod
    def from_csv(cls, path: str, name: str | None = None, kind: str = "load") -> "Profile":
        """Read columns timestamp, scaling_factor (one row per step)."""
        mults = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if "scaling_factor" in row:
                    mults.append(float(row["scaling_factor"]))
        return cls(mults, name=name or Path(path).stem, kind=kind)

    @classmethod
    def from_entsoe(cls, country_code: str, date: str, hours: int = DEFAULT_HOURS,
                    seed: int = 42, kind: str = "load") -> "Profile":
        """Read data/entsoe/<country>_<date>.csv if present; otherwise warn and use synthetic.
        ENTSO-E actual-load series are hourly MW per bidding zone; we normalize to a multiplier."""
        path = Path("data/entsoe") / f"{country_code}_{date}.csv"
        if path.is_file():
            logger.info("loading ENTSO-E profile from %s", path)
            return cls.from_csv(str(path), name=f"entsoe_{country_code}_{date}", kind=kind)
        logger.warning("ENTSO-E data %s not found; falling back to synthetic profile", path)
        return cls.synthetic(hours=hours, seed=seed, name=f"entsoe_{country_code}_{date}_synthetic", kind=kind)


# Convenience aliases so call sites read intentionally.
def LoadProfile(*a, **k) -> Profile:  # noqa: N802 (factory-style alias)
    return Profile.synthetic(*a, **k, kind="load")


def GenerationProfile(hours: int = DEFAULT_HOURS) -> Profile:  # noqa: N802
    return Profile.flat(hours=hours, value=1.0, name="gen_flat", kind="gen")


def capture_base(net) -> dict:
    """Snapshot the base element values the profiles scale, once, before stepping."""
    base = {
        "load_p": net.load["p_mw"].copy() if len(net.load) else None,
        "load_q": net.load["q_mvar"].copy() if len(net.load) else None,
        "gen_p": net.gen["p_mw"].copy() if len(net.gen) else None,
    }
    return base


def apply_profile(net, base: dict, load_profile: Profile, gen_profile: Profile, t: int) -> None:
    """Set this step's load/gen to BASE * profile(t), in place. Never cumulative: always derived
    from the captured base, so reapplying a step is idempotent."""
    if hasattr(load_profile, "apply_to_net"):
        load_profile.apply_to_net(net, base, t)
    else:
        lm = load_profile.at(t)
        if base.get("load_p") is not None:
            net.load["p_mw"] = base["load_p"] * lm
        if base.get("load_q") is not None:
            net.load["q_mvar"] = base["load_q"] * lm

    if hasattr(gen_profile, "apply_to_net"):
        gen_profile.apply_to_net(net, base, t)
    else:
        gm = gen_profile.at(t)
        if base.get("gen_p") is not None and gm != 1.0:
            net.gen["p_mw"] = base["gen_p"] * gm
