"""Cascade Severity Score (CSS), status and band derivation.

Doc reference: docs/03-phase1-engine.md section 6. The formula block is
the law; this module implements it and nothing else:

  CSS = 1000 * blackout
      +  500 * diverged
      +   10 * load_shed_pct          (0 to 100, percent of system load)
      +   20 * min(cascade_depth, 20)
      +    1 * residual_violations

Pinning rule (D9): FULL_BLACKOUT fixes load_shed_pct = 100,
cascade_depth = 0, residual = 0, so CSS = CSS_MAX = 2000 exactly.
Bands derive from flags, never from score thresholds (D10).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.engine.cascade import CascadeOutcome
from src.engine.constants import CSS_MAX


@dataclass
class Severity:
    score: float
    band: str  # CRITICAL | HIGH | MEDIUM | LOW
    blackout: bool
    diverged: bool
    cascade_depth: int
    load_shed_mw: float
    load_shed_pct: float
    residual_violations: int


def css(
    blackout: bool,
    diverged: bool,
    load_shed_pct: float,
    cascade_depth: int,
    residual_violations: int,
) -> float:
    return (
        1000.0 * bool(blackout)
        + 500.0 * bool(diverged)
        + 10.0 * float(load_shed_pct)
        + 20.0 * min(int(cascade_depth), 20)
        + 1.0 * int(residual_violations)
    )


def band_from_flags(
    blackout: bool, diverged: bool, cascade_depth: int, load_shed_mw: float,
    residual_violations: int,
) -> str:
    if blackout or diverged:
        return "CRITICAL"
    if cascade_depth >= 1 or load_shed_mw > 0:
        return "HIGH"
    if residual_violations > 0:
        return "MEDIUM"
    return "LOW"


def status_from_flags(
    blackout: bool, diverged: bool, cascade_depth: int, load_shed_mw: float,
    residual_violations: int,
) -> str:
    if blackout:
        return "FULL_BLACKOUT"
    if diverged:
        return "DIVERGED"
    if cascade_depth >= 1:
        return "CASCADE"
    if residual_violations > 0 or load_shed_mw > 0:
        # Doc 03 defines VIOLATIONS as "depth 0, no shed, residual > 0"
        # and leaves topology-only islanding (depth 0, shed > 0,
        # residual 0) without a status. SECURE would contradict band
        # HIGH, and CASCADE would break the depth >= 1 invariant, so
        # such cases report VIOLATIONS; the record's islanded_buses and
        # load_shed_mw carry the substance. Doc gap logged as D57.
        return "VIOLATIONS"
    return "SECURE"


def blackout_severity() -> Severity:
    return Severity(
        score=CSS_MAX,
        band="CRITICAL",
        blackout=True,
        diverged=False,
        cascade_depth=0,
        load_shed_mw=0.0,  # overwritten by the caller with system load
        load_shed_pct=100.0,
        residual_violations=0,
    )


def score_severity(cascade: CascadeOutcome, total_load_mw: float) -> Severity:
    shed_pct = (
        100.0 * cascade.shed_mw / total_load_mw if total_load_mw > 0 else 0.0
    )
    shed_pct = min(shed_pct, 100.0)
    residual = len(cascade.final_state.get("violations", []))
    return Severity(
        score=css(False, cascade.diverged, shed_pct, cascade.depth, residual),
        band=band_from_flags(
            False, cascade.diverged, cascade.depth, cascade.shed_mw, residual
        ),
        blackout=False,
        diverged=cascade.diverged,
        cascade_depth=cascade.depth,
        load_shed_mw=cascade.shed_mw,
        load_shed_pct=shed_pct,
        residual_violations=residual,
    )
