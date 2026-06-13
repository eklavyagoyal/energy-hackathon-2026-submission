"""build_grid_state_summary: assemble the GridStateSummary dict from a live net + sweep output.

Shape matches docs/04-agent.md / docs/07-api-contracts.md. Every number originates from an engine
result; nothing is invented here. Adapted to the src engine: ContingencyResult.outage is a dict and
severity is a Severity dataclass; element/bus references go through native_index.
"""
from __future__ import annotations

from src.engine.actions import to_jsonable
from src.engine.network import native_index, opf_available, total_load_mw, base_case_summary
from src.engine.scan import ContingencyResult

_INSECURE_BANDS = ("CRITICAL", "HIGH")


def _contingency_summary(r: ContingencyResult) -> dict:
    """Trim a full ContingencyResult to the ContingencySummary shape."""
    return {
        "contingency_id": r.contingency_id,
        "outage_name": r.outage.get("name", r.contingency_id),
        "status": r.status,
        "severity": to_jsonable(r.severity),
        "first_overloads": to_jsonable(r.first_overloads),
    }


def _redispatchable_gens(net) -> list[dict]:
    """Controllable generators + their marginal cost from poly_cost. native_index keeps the
    portability contract (int | str element/bus references)."""
    cost_by_gen: dict = {}
    if "poly_cost" in net and len(net.poly_cost):
        for _, row in net.poly_cost.iterrows():
            if str(row["et"]) == "gen":
                cost_by_gen[native_index(row["element"])] = float(row["cp1_eur_per_mw"])

    gens: list[dict] = []
    if len(net.gen) == 0:
        return gens
    in_service = net.gen.in_service if "in_service" in net.gen.columns else net.gen.index == net.gen.index
    for idx in net.gen.index[in_service]:
        row = net.gen.loc[idx]
        gens.append({
            "index": native_index(idx),
            "bus": native_index(row["bus"]),
            "p_mw": round(float(row["p_mw"]), 2),
            "min_p_mw": round(float(row.get("min_p_mw", 0.0) or 0.0), 2),
            "max_p_mw": round(float(row.get("max_p_mw", 0.0) or 0.0), 2),
            "cost_per_mw": round(cost_by_gen.get(native_index(idx), 0.0), 4),
        })
    return gens


def build_grid_state_summary(net, scan_out: dict, scenario_id: str, scenario_context: dict | None = None) -> dict:
    """Return the GridStateSummary dict. scan_out is {"results": [ContingencyResult]}; the live net
    is read-only (base_case_summary runs on a working copy)."""
    base = base_case_summary(net)

    results: list[ContingencyResult] = scan_out.get("results", []) or []
    insecure = [r for r in results if r.severity.band in _INSECURE_BANDS]
    worst_entries = [_contingency_summary(r) for r in results[:5]]

    security = {
        "n_contingencies_scanned": len(results),
        "n_insecure": len(insecure),
        "worst": worst_entries,
    }

    options = {
        "redispatchable_gens": _redispatchable_gens(net),
        "opf_available": bool(opf_available(net)),
        "sheddable_load_mw": round(total_load_mw(net), 1),
    }

    if scenario_context is None:
        scenario_context = net.get("geographic_scenario", {}) if isinstance(net, dict) else {}

    return to_jsonable({
        "scenario_id": scenario_id,
        "base_case": base,
        "security": security,
        "options": options,
        "geographic_context": {
            "available": bool(scenario_context.get("geographic_context")),
            "scenario": scenario_context.get("name"),
            "title": scenario_context.get("title"),
            "target_region_hint": scenario_context.get("target_region_hint"),
            "event_selection": scenario_context.get("event_selection", {}),
        } if scenario_context else {},
    })
