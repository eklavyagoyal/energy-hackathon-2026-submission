"""Dataset portability: the code is built dataset-agnostic from the
start, so it survives the eventual swap of case118 for a real TSO net.

Covered:
- 4-bus net with TWO ext_grids: slack guard finds both, neither is a
  battery candidate, a full sweep runs and verification works
- non-sequential integer bus indices [0, 5, 23, 99]: scoring and a full
  N-1 sweep iterate by net.bus.index, never range(len(...))
- string bus identifiers: every dtype-agnostic code path preserves them
  (native_index, candidate set, scoring, schema serialization). The
  pandapower SOLVER requires numeric indices, so a real string-ID TSO
  export is mapped to ints by the loader; the rest of the stack stays
  string-clean, which is what these assertions lock in
- missing poly_cost / no controllable gens: recommendation runs, logs a
  warning, falls back to the discharge heuristic, does not crash
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pandapower as pp
import pytest

from src.battery.recommender import recommend_battery_locations
from src.battery.schemas import RecommendationRequest, VerificationResult
from src.battery.scoring import candidate_buses, score_buses
from src.battery.verification import verify_battery_candidate
from src.config import Settings
from src.engine.network import native_index, opf_available, slack_bus_set
from src.engine.scan import build_contingency_set, run_contingency_sweep
from tests.battery.factories import line_net, make_result


# --------------------------------------------------------------------------
# Two ext_grids
# --------------------------------------------------------------------------
def two_slack_net():
    net = pp.create_empty_network()
    for i in range(4):
        pp.create_bus(net, vn_kv=110.0, index=i)
    pp.create_ext_grid(net, bus=0, vm_pu=1.02)
    pp.create_ext_grid(net, bus=3, vm_pu=1.02)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        pp.create_line_from_parameters(
            net, a, b, length_km=10, r_ohm_per_km=0.06, x_ohm_per_km=0.2,
            c_nf_per_km=10, max_i_ka=0.4,
        )
    pp.create_load(net, bus=1, p_mw=30, q_mvar=10)
    pp.create_load(net, bus=2, p_mw=30, q_mvar=10)
    return net


def test_two_ext_grids_recognized_and_excluded():
    net = two_slack_net()
    assert slack_bus_set(net) == {0, 3}
    candidates, excluded = candidate_buses(net)
    assert set(candidates) == {1, 2}
    assert excluded == [0, 3]


def test_two_ext_grids_full_sweep_and_verify_run():
    net = two_slack_net()
    cont = build_contingency_set(net)
    # contingency set includes both ext_grids
    assert {"ext_grid_0", "ext_grid_1"} <= {o.contingency_id for o in cont}
    baseline = run_contingency_sweep(net, cont)
    assert len(baseline) == len(cont)
    # verify a non-slack candidate; runs the same pipeline end to end
    vr = verify_battery_candidate(net, 1, contingencies=cont, baseline=baseline,
                                  battery_p_mw=10.0)
    assert isinstance(vr, VerificationResult)
    assert vr.n_scenarios == len(cont)
    assert vr.bus_idx == 1


# --------------------------------------------------------------------------
# Non-sequential integer indices
# --------------------------------------------------------------------------
def nonseq_net():
    net = pp.create_empty_network()
    for i in [0, 5, 23, 99]:
        pp.create_bus(net, vn_kv=110.0, index=i)
    pp.create_ext_grid(net, bus=0, vm_pu=1.02)
    for a, b in [(0, 5), (5, 23), (23, 99)]:
        pp.create_line_from_parameters(
            net, a, b, length_km=10, r_ohm_per_km=0.06, x_ohm_per_km=0.2,
            c_nf_per_km=10, max_i_ka=0.4,
        )
    pp.create_load(net, bus=23, p_mw=20, q_mvar=5)
    pp.create_load(net, bus=99, p_mw=20, q_mvar=5)
    return net


def test_non_sequential_indices_candidates():
    net = nonseq_net()
    candidates, excluded = candidate_buses(net)
    assert set(candidates) == {5, 23, 99}  # bus 0 is slack
    assert excluded == [0]


def test_non_sequential_indices_full_pipeline():
    net = nonseq_net()
    cont = build_contingency_set(net)
    baseline = run_contingency_sweep(net, cont)
    scores = score_buses(net, baseline)
    scored_buses = {bs.bus_idx for bs in scores}
    assert scored_buses == {5, 23, 99}
    # every scored bus index is one of the real, non-sequential indices
    assert all(bs.bus_idx in [5, 23, 99] for bs in scores)
    # verification re-solves against the same non-sequential indices
    vr = verify_battery_candidate(net, 99, contingencies=cont, baseline=baseline,
                                  battery_p_mw=10.0)
    assert vr.bus_idx == 99


# --------------------------------------------------------------------------
# String bus identifiers (dtype-agnostic code paths; solver-free)
# --------------------------------------------------------------------------
def string_id_net():
    """Bus indices are strings.

    pandapower CANNOT hold string bus indices: its create_* helpers store
    bus references in numeric columns and the solver builds its ppc from
    numeric ids, so a string-id net cannot be constructed or solved
    (verified: create_line raises "could not convert string to float").
    A real string-id TSO export is therefore mapped label->int by the
    loader (TSORealLoader TODO), keeping a reverse map for display.

    What MUST stay string-clean is everything ABOVE the solver: the
    aggregation, scoring, slack exclusion, and API schemas. We assert
    that here with a minimal pandas-backed net stub exposing the same
    .bus/.line/.ext_grid/.trafo frames those code paths read, with string
    bus ids throughout. If any path coerced ids to int, these fail.
    """
    bus = pd.DataFrame(
        {"in_service": [True, True, True, True]},
        index=["SLACK", "NORTH", "SOUTH", "EAST"],
    )
    line = pd.DataFrame(
        {
            "from_bus": ["SLACK", "NORTH", "SOUTH"],
            "to_bus": ["NORTH", "SOUTH", "EAST"],
            "in_service": [True, True, True],
        }
    )
    ext_grid = pd.DataFrame({"bus": ["SLACK"], "in_service": [True]})
    trafo = pd.DataFrame()
    return SimpleNamespace(bus=bus, line=line, ext_grid=ext_grid, trafo=trafo)


def test_string_ids_preserved_by_native_index():
    assert native_index("NORTH") == "NORTH"
    assert isinstance(native_index("NORTH"), str)


def test_string_ids_slack_set_and_candidates():
    net = string_id_net()
    assert slack_bus_set(net) == {"SLACK"}
    candidates, excluded = candidate_buses(net)
    assert set(candidates) == {"NORTH", "SOUTH", "EAST"}
    assert excluded == ["SLACK"]


def test_string_ids_scoring_emits_string_buses():
    net = string_id_net()
    # synthetic results referencing string buses and int line indices
    results = [
        make_result("line_0", line_loading={0: 95.0, 1: 50.0, 2: 10.0},
                    bus_vm={"NORTH": 0.92, "SOUTH": 0.99, "EAST": 1.0}),
        make_result("ext_grid_0", status="FULL_BLACKOUT", blackout=True,
                    load_shed_mw=40.0),
    ]
    scores = score_buses(net, results)
    for bs in scores:
        assert isinstance(bs.bus_idx, str)
    assert {bs.bus_idx for bs in scores} == {"NORTH", "SOUTH", "EAST"}


def test_string_ids_survive_schema_serialization():
    net = string_id_net()
    results = [
        make_result("line_0", line_loading={0: 95.0}, bus_vm={"NORTH": 0.92}),
    ]
    scores = score_buses(net, results)
    north = next(bs for bs in scores if bs.bus_idx == "NORTH")
    dumped = north.model_dump()
    assert dumped["bus_idx"] == "NORTH"
    # round-trips as JSON with the string id intact
    as_json = north.model_dump_json()
    assert '"NORTH"' in as_json


def test_string_id_verify_rejects_slack_without_solver():
    net = string_id_net()
    from src.battery.verification import assert_not_slack, resolve_bus, SlackBusError

    assert resolve_bus(net, "NORTH") == "NORTH"
    with pytest.raises(SlackBusError):
        assert_not_slack(net, "SLACK")


# --------------------------------------------------------------------------
# Missing poly_cost / no controllable gens
# --------------------------------------------------------------------------
def no_cost_net():
    """A solvable net with NO poly_cost and a non-controllable gen."""
    net = pp.create_empty_network()
    for i in range(4):
        pp.create_bus(net, vn_kv=110.0, index=i)
    pp.create_ext_grid(net, bus=0, vm_pu=1.02)
    pp.create_gen(net, bus=2, p_mw=10.0, vm_pu=1.0, controllable=False)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        pp.create_line_from_parameters(
            net, a, b, length_km=10, r_ohm_per_km=0.06, x_ohm_per_km=0.2,
            c_nf_per_km=10, max_i_ka=0.4,
        )
    pp.create_load(net, bus=1, p_mw=20, q_mvar=5)
    pp.create_load(net, bus=3, p_mw=20, q_mvar=5)
    return net


def test_opf_available_false_without_cost_data():
    net = no_cost_net()
    assert len(net.poly_cost) == 0
    assert opf_available(net) is False


def test_recommendation_runs_without_cost_data_and_warns(caplog):
    net = no_cost_net()
    req = RecommendationRequest(top_k=2, battery_capacity_mw=10.0, verify=True)
    with caplog.at_level(logging.WARNING):
        resp = recommend_battery_locations(net, req, Settings())
    # did not crash; produced candidates
    assert len(resp.candidates) >= 1
    assert resp.opf_available is False
    assert resp.dispatch_model == "fixed_full_discharge_heuristic"
    # a warning about the OPF fallback was logged
    assert any("poly_cost" in r.message or "OPF" in r.message or "heuristic" in r.message
               for r in caplog.records)
