"""Operator narration for battery candidates.

Same hard constraint as the Phase 1 agent layer (docs/04-agent.md):
the LLM never computes physics. It receives structured findings whose
every number came out of the solver pipeline and returns exactly 3
sentences. Live LLM narration is opt-in via BATTERY_NARRATION_MODE=llm;
otherwise the deterministic template narrator answers from the same
findings. If the LLM is unavailable, violates the 3-sentence contract,
or states a number that does not appear in its input, the template is
used instead. Either way: zero invented numbers.
"""

from __future__ import annotations

import logging
import re

from src.config import Settings

logger = logging.getLogger(__name__)

# Literal system prompt template from the battery-feature contract.
# Placeholder semantics: the verification sentence claims prevented
# cascades, so {improved_count} is filled with the measured
# cascades_prevented count (not scenarios_improved) to keep the
# narrated sentence true. Both counts are present in the API response.
NARRATION_PROMPT_TEMPLATE = """You are a TSO grid operations advisor. A battery storage recommendation
engine has identified bus {bus_idx} as a high-priority location with
solver-verified impact.

Structured findings:
- Score: {score} (top {rank} of {total_buses})
- Congestion: line {worst_line} on this bus reached {max_loading}%
  in {congestion_count} of {total_scenarios} N-1 scenarios
- Voltage: bus voltage dropped to {min_voltage} p.u. in {voltage_count} scenarios
- Verification result: deploying a {capacity}MW / {energy}MWh battery here
  prevents {improved_count} cascades and saves {mw_saved} MW of load shedding

Write exactly 3 sentences for the operator:
1. What problem this bus has and why
2. Why this location helps specifically (P vs Q vs topology argument)
3. What the verified impact is in concrete numbers

Use TSO terminology. Be specific, not generic. Do not invent numbers.
"""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def build_findings(rank: int, total_buses: int, bus_score, verification) -> dict:
    """Flatten one candidate into the narration placeholder dict.
    Every value is copied from solver-produced records; rounding here
    defines the exact tokens the narration is allowed to use."""
    ctx = bus_score.context
    return {
        "bus_idx": bus_score.bus_idx,
        "score": round(bus_score.score, 2),
        "rank": rank,
        "total_buses": total_buses,
        "worst_line": ctx.worst_line if ctx.worst_line is not None else "none",
        "max_loading": (
            round(ctx.worst_line_loading_pct, 1)
            if ctx.worst_line_loading_pct is not None
            else "n/a"
        ),
        "congestion_count": ctx.congestion_count,
        "total_scenarios": ctx.total_scenarios,
        "min_voltage": (
            round(ctx.worst_voltage_pu, 3)
            if ctx.worst_voltage_pu is not None
            else (round(ctx.min_voltage_pu, 3) if ctx.min_voltage_pu is not None else "n/a")
        ),
        "voltage_dir": ctx.worst_voltage_dir,
        "voltage_count": ctx.voltage_count,
        "cascade_count": ctx.cascade_count,
        "capacity": verification.battery_p_mw if verification else None,
        "energy": verification.battery_max_e_mwh if verification else None,
        "improved_count": verification.cascades_prevented if verification else None,
        "scenarios_improved": verification.scenarios_improved if verification else None,
        "scenarios_worsened": verification.scenarios_worsened if verification else None,
        "mw_saved": (
            round(verification.mw_load_shed_avoided, 1) if verification else None
        ),
        "verdict": verification.verdict if verification else None,
    }


def template_narration(findings: dict) -> str:
    """Deterministic 3-sentence fallback, filled from the same
    structured findings the LLM would see."""
    problems = []
    if findings["congestion_count"] > 0:
        problems.append(
            f"line {findings['worst_line']} connected to it reached "
            f"{findings['max_loading']} percent loading in "
            f"{findings['congestion_count']} of {findings['total_scenarios']} "
            "N-1 scenarios"
        )
    if findings["voltage_count"] > 0:
        verb = "rose to" if findings.get("voltage_dir") == "high" else "dropped to"
        problems.append(
            f"its voltage left the 0.95 to 1.05 p.u. band in "
            f"{findings['voltage_count']} scenarios, {verb} "
            f"{findings['min_voltage']} p.u."
        )
    if findings["cascade_count"] > 0:
        problems.append(
            f"it sat in the cascade chain or an islanded component in "
            f"{findings['cascade_count']} scenarios"
        )
    if not problems:
        problems.append("it shows no standing violation but ranks highest overall")
    s1 = (
        f"Bus {findings['bus_idx']} ranks {findings['rank']} of "
        f"{findings['total_buses']} candidate buses (score {findings['score']}): "
        + "; ".join(problems)
        + "."
    )
    s2 = (
        "A battery at this bus injects active power at the stressed node "
        "itself, so the relief lands on the exact corridor and voltage "
        "zone the contingencies hit instead of depending on remote "
        "redispatch across the same congested paths."
    )
    if findings["capacity"] is not None:
        s3 = (
            f"The solver-verified counterfactual with a {findings['capacity']} MW / "
            f"{findings['energy']} MWh unit improves {findings['scenarios_improved']} "
            f"of {findings['total_scenarios']} N-1 scenarios, prevents "
            f"{findings['improved_count']} cascades and avoids {findings['mw_saved']} MW "
            f"of load shedding (verdict {findings['verdict']})."
        )
    else:
        s3 = (
            "Verification was not run for this candidate, so this ranking is "
            "a screening signal only and carries no solver-verified impact "
            "claim."
        )
    return " ".join([s1, s2, s3])


def _narration_valid(text: str, prompt: str) -> bool:
    """Contract check (docs/04-agent.md section 6): exactly 3
    sentences, and every numeric token appears verbatim in the input
    the model saw."""
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
    if len(sentences) != 3:
        return False
    for token in _NUMBER_TOKEN.findall(text):
        if token not in prompt:
            return False
    return True


def llm_narration(findings: dict, settings: Settings) -> str | None:
    """One Anthropic call, temperature 0; returns None on any failure
    or contract violation so the caller falls back to the template."""
    if settings.battery_narration_mode.strip().lower() != "llm":
        return None
    if not settings.anthropic_api_key:
        return None
    prompt = NARRATION_PROMPT_TEMPLATE.format(**{
        k: findings.get(k) for k in (
            "bus_idx", "score", "rank", "total_buses", "worst_line",
            "max_loading", "congestion_count", "total_scenarios",
            "min_voltage", "voltage_count", "capacity", "energy",
            "improved_count", "mw_saved",
        )
    })
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.narration_model,
            max_tokens=400,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
    except Exception as exc:
        logger.warning("LLM narration failed (%s); using template", exc)
        return None
    if not _narration_valid(text, prompt):
        logger.warning(
            "LLM narration violated the 3-sentence / verbatim-number "
            "contract; using template"
        )
        return None
    return text


def narrate(findings: dict, settings: Settings) -> tuple[str, str]:
    """Returns (narration, source) where source is "llm" or
    "template". Narration is only attempted for verified candidates;
    unverified ones get the honest template variant."""
    if findings.get("capacity") is not None:
        text = llm_narration(findings, settings)
        if text:
            return text, "llm"
    return template_narration(findings), "template"
