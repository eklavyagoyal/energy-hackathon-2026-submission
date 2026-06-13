"""Acceptance coverage for geographic PyPSA-Eur stress scenarios."""
from __future__ import annotations

import logging
from pathlib import Path

from src.grid.loader import Case118Loader
from src.grid.pypsa_eur_loader import PyPSAEurLoader
from src.timeseries.scenarios import build_geographic_scenario_run
from src.timeseries.simulator import run_simulation

ROOT = Path(__file__).resolve().parents[2]


def _pypsa_sample():
    return PyPSAEurLoader(ROOT / "data" / "pypsa_sample").load()


def _run_prepared(prepared, horizon: int | None = None):
    return run_simulation(
        prepared.net,
        prepared.load_profile,
        prepared.gen_profile,
        prepared.event_stream,
        horizon_steps=horizon or prepared.scenario.duration_hours,
        seed=42,
        profile_id=prepared.scenario.name,
        event_scenario=prepared.scenario.name,
        scenario_context=prepared.metadata,
    )


def test_geographic_event_selection():
    """The Dunkelflaute line outage is selected by coordinates: one end north, one end south."""
    prepared = build_geographic_scenario_run("dunkelflaute", _pypsa_sample(), seed=42)
    event = prepared.metadata["event_selection"]
    assert event["etype"] == "line"
    assert event["v_nom"] >= 350.0

    net = prepared.net
    row = net.line.loc[event["index"]]
    y0 = float(net.bus.at[row.from_bus, "y"])
    y1 = float(net.bus.at[row.to_bus, "y"])
    assert (y0 > 52.0 and y1 < 49.0) or (y1 > 52.0 and y0 < 49.0)


def test_dunkelflaute_on_pypsa_eur_de():
    """Scenario loads, runs 24 steps, and includes a verified action that avoids cascade shedding."""
    prepared = build_geographic_scenario_run("dunkelflaute", _pypsa_sample(), seed=42)
    trace = _run_prepared(prepared)

    assert len(trace.steps) == 24
    assert any(s.t == 14 and any(e["kind"] == "line_outage" for e in s.events) for s in trace.steps)

    preventing = [
        s for s in trace.steps
        if s.commit_status == "applied"
        and s.verification.get("verified") is True
        and s.verification.get("deltas", {}).get("load_shed_avoided_mw", 0.0) > 0
    ]
    assert preventing, "expected at least one verified action to avoid cascade shedding"


def test_scenario_fallback_to_case118(caplog):
    """Geographic scenarios fall back on case118 with a warning and no crash."""
    caplog.set_level(logging.WARNING)
    prepared = build_geographic_scenario_run("heatwave", Case118Loader().load(), seed=7)
    trace = _run_prepared(prepared, horizon=2)

    assert len(trace.steps) == 2
    assert prepared.metadata["geographic_context"] is False
    assert "geographic context is lost" in caplog.text


def test_scenarios_are_deterministic():
    """Same scenario + same seed produces the same trace."""
    first = build_geographic_scenario_run("dunkelflaute", _pypsa_sample(), seed=42)
    second = build_geographic_scenario_run("dunkelflaute", _pypsa_sample(), seed=42)

    a = _run_prepared(first, horizon=8).to_dict()
    b = _run_prepared(second, horizon=8).to_dict()
    assert a == b
