"""Freeze the real-time time-stepped demo into a fixture and print a report.

Run: python scripts/freeze_realtime_demo.py   (from an activated .venv; see README)

Produces fixtures/realtime_trace.json: a full 24-step SimulationTrace over the congestion pocket,
so the frontend timeline can replay an instant, deterministic demo without paying the ~20 s
simulation cost live. Every number is solver-produced (the simulator reuses the Phase 1 engine);
nothing here is illustrative. Deterministic for the fixed seed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from src.engine.scenarios import apply_scenario  # noqa: E402
from src.grid.loader import Case118Loader  # noqa: E402
from src.timeseries.events import build_event_scenario  # noqa: E402
from src.timeseries.profiles import GenerationProfile, Profile  # noqa: E402
from src.timeseries.simulator import run_simulation  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
HORIZON = 24
SEED = 42


def main() -> None:
    FIXTURES.mkdir(exist_ok=True)
    print("loading case118 and seeding the congestion pocket ...")
    base = Case118Loader().load()
    net = apply_scenario(base, "demo_congestion")

    load_profile = Profile.synthetic(hours=HORIZON, seed=SEED)
    gen_profile = GenerationProfile(hours=HORIZON)
    events = build_event_scenario("stress_demo", net)

    t0 = time.perf_counter()
    trace = run_simulation(
        net, load_profile, gen_profile, events,
        agent_mode="opf", horizon_steps=HORIZON, seed=SEED,
        profile_id="synthetic_24h", event_scenario="stress_demo",
    )
    wall = time.perf_counter() - t0

    out = FIXTURES / "realtime_trace.json"
    out.write_text(json.dumps(trace.to_dict(), indent=2))

    counts: dict = {}
    for s in trace.steps:
        counts[s.commit_status] = counts.get(s.commit_status, 0) + 1

    print(f"  {HORIZON}-step run in {wall:.1f} s")
    print("\n=== REAL-TIME DEMO REPORT ===")
    print(f"profile: {trace.profile_id} | events: {trace.event_scenario} | agent: {trace.agent_mode}")
    print(f"commit-status mix: {counts}")
    print("per-step (t | load mult | base overloads | worst N-1 | decision):")
    for s in trace.steps:
        worst = (s.worst_contingency or {}).get("outage_name", "none")
        sev = (s.worst_contingency or {}).get("severity", {})
        band = sev.get("band", "-") if isinstance(sev, dict) else "-"
        ov = s.baseline_state.get("n_overloads", 0)
        print(f"  t={s.t:2d} {s.timestamp} | ov={ov:2d} | worst={band:8s} {worst[:34]:34s} | {s.commit_status}")
    print(f"\nfixture written to {out}")


if __name__ == "__main__":
    main()
