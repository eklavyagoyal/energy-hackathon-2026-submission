"""Live application state: one net, one cached baseline sweep.

Centralizes the single mutable grid state (docs/02-architecture.md
section 6: exactly one live net, single uvicorn worker, no locking) plus
the baseline N-1 sweep cache that makes /battery/recommendations fast.

The baseline cache is the missing piece flagged in
src/battery/recommender.py (D52): Phase 1 ships no shared result cache on
this branch, so the API owns it. It is recomputed exactly when the live
net changes (scenario set), and reused by every recommendation and
verify call until then. This is why a recommendation returns
baseline_cached=true and meets the <10 s budget: the expensive full
sweep already ran at scenario-set time.
"""

from __future__ import annotations

import logging
import time

import pandapower as pp

from src.config import Settings
from src.engine.network import base_case_summary, opf_available
from src.engine.scan import (
    ContingencyResult,
    build_contingency_set,
    rank,
    run_contingency_sweep,
)
from src.engine.scenarios import apply_scenario, default_scenario_for_net
from src.grid.loader import get_loader

logger = logging.getLogger(__name__)


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_net: "pp.pandapowerNet" = get_loader(settings).load()
        self.scenario_id: str = default_scenario_for_net(self.base_net)
        self.net: "pp.pandapowerNet" = apply_scenario(self.base_net, self.scenario_id)
        self.contingencies: list = []
        self.baseline: list[ContingencyResult] = []
        self.baseline_ms: float = 0.0
        self.refresh_baseline()

    def set_scenario(self, scenario_id: str) -> None:
        """Replace the live net with a fresh scenario and recompute the
        baseline cache. This is the only place (besides init) the live
        net changes."""
        self.net = apply_scenario(self.base_net, scenario_id)
        self.scenario_id = scenario_id
        self.refresh_baseline()

    def refresh_baseline(self) -> None:
        t0 = time.perf_counter()
        self.contingencies = build_contingency_set(self.net)
        self.baseline = run_contingency_sweep(self.net, self.contingencies)
        self.baseline_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "baseline cached: scenario=%s, %d contingencies, %.0f ms",
            self.scenario_id, len(self.baseline), self.baseline_ms,
        )

    def ranked_baseline(self) -> list[ContingencyResult]:
        return rank(self.baseline)

    def state_summary(self) -> dict:
        bc = base_case_summary(self.net)
        worst = self.ranked_baseline()[:5]
        n_insecure = sum(
            1 for r in self.baseline if r.severity.band in ("CRITICAL", "HIGH")
        )
        return {
            "scenario_id": self.scenario_id,
            "base_case": bc,
            "security": {
                "n_contingencies_scanned": len(self.baseline),
                "n_insecure": n_insecure,
                "baseline_ms": round(self.baseline_ms, 1),
                "worst": [
                    {
                        "contingency_id": r.contingency_id,
                        "outage_name": r.outage["name"],
                        "status": r.status,
                        "severity": {
                            "score": r.severity.score,
                            "band": r.severity.band,
                            "cascade_depth": r.severity.cascade_depth,
                            "load_shed_mw": r.severity.load_shed_mw,
                        },
                    }
                    for r in worst
                ],
            },
            "opf_available": opf_available(self.net),
        }
