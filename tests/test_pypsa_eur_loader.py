"""PyPSAEurLoader (U12): the str<->int bus map, the hand-map (KTD6), and a real solve.

Builds a tiny PyPSA CSV-folder export in tmp_path (string bus names, two voltage levels) so the
loader is tested with no pypsa dependency, the way the CSV intake path runs.
"""
from __future__ import annotations

import math

import pandapower as pp
import pytest

from src.grid.pypsa_eur_loader import PyPSAEurLoader

SQRT3 = math.sqrt(3.0)


@pytest.fixture
def pypsa_csv(tmp_path):
    """A 3-bus AC export: a 380 kV corridor DE<->FR, a 380/220 kV transformer, a slack gen, loads.
    String bus names are the whole point (KTD3)."""
    (tmp_path / "buses.csv").write_text(
        "name,v_nom\nDE_380,380\nFR_380,380\nDE_220,220\n"
    )
    # one 380 kV line DE<->FR; s_nom 2000 MVA
    (tmp_path / "lines.csv").write_text(
        "name,bus0,bus1,r,x,s_nom,v_nom\n"
        "DE_FR_1,DE_380,FR_380,2.0,20.0,2000,380\n"
    )
    # 380/220 transformer, per-unit r/x on its own base
    (tmp_path / "transformers.csv").write_text(
        "name,bus0,bus1,r,x,s_nom\nT_DE,DE_380,DE_220,0.01,0.12,1000\n"
    )
    (tmp_path / "generators.csv").write_text(
        "name,bus,control,p_set,p_nom\n"
        "slack_DE,DE_380,Slack,500,3000\n"
        "wind_FR,FR_380,PV,300,800\n"
    )
    (tmp_path / "loads.csv").write_text(
        "name,bus,p_set,q_set\nload_FR,FR_380,400,50\nload_DE220,DE_220,350,40\n"
    )
    return tmp_path


def test_loads_with_int_buses_and_name_map(pypsa_csv):
    loader = PyPSAEurLoader(pypsa_csv)
    net = loader.load()
    # pandapower 3.4 cannot index buses by string, so buses are int-indexed (KTD3)
    assert list(net.bus.index) == [0, 1, 2]
    assert all(isinstance(i, int) for i in net.bus.index)
    # the str<->int map round-trips both ways
    assert loader.bus_name_to_idx["DE_380"] == 0
    assert loader.idx_to_bus_name[0] == "DE_380"
    assert loader.native_bus(2) == "DE_220"
    # and it is attached to the net for boundary translation
    assert net["bus_name_map"][1] == "FR_380"


def test_handmap_thermal_and_reactance(pypsa_csv):
    """KTD6 hand-map: max_i_ka = s_nom/(sqrt(3)*v_nom); trafo vk_percent = sqrt(r^2+x^2)*100."""
    net = PyPSAEurLoader(pypsa_csv).load()
    line = net.line.iloc[0]
    assert line["max_i_ka"] == pytest.approx(2000.0 / (SQRT3 * 380.0), rel=1e-6)
    tr = net.trafo.iloc[0]
    assert tr["vk_percent"] == pytest.approx(math.sqrt(0.01**2 + 0.12**2) * 100.0, rel=1e-6)
    assert tr["vkr_percent"] == pytest.approx(0.01 * 100.0, rel=1e-6)
    assert tr["sn_mva"] == pytest.approx(1000.0)


def test_slack_generator_becomes_ext_grid(pypsa_csv):
    net = PyPSAEurLoader(pypsa_csv).load()
    assert len(net.ext_grid) == 1
    assert int(net.ext_grid.iloc[0]["bus"]) == 0  # DE_380, the slack-tagged generator
    assert len(net.gen) == 1  # the PV wind generator remains a gen


def test_network_solves(pypsa_csv):
    """The hand-mapped net must converge under a real AC power flow (the whole point of mapping)."""
    net = PyPSAEurLoader(pypsa_csv).load()
    pp.runpp(net)
    assert net.converged
    assert len(net.res_line) == 1
    assert net.res_line.loading_percent.iloc[0] >= 0


def test_disconnected_component_gets_its_own_slack(tmp_path):
    """KTD6: an island with no slack-tagged gen still gets an ext_grid (or the power flow has no
    angle reference). Two buses, no line between them, no slack gen."""
    (tmp_path / "buses.csv").write_text("name,v_nom\nA,220\nB,220\n")
    (tmp_path / "lines.csv").write_text("name,bus0,bus1,r,x,s_nom,v_nom\n")  # no lines: 2 islands
    (tmp_path / "generators.csv").write_text(
        "name,bus,control,p_set,p_nom\ngA,A,PV,10,50\ngB,B,PV,10,50\n"
    )
    (tmp_path / "loads.csv").write_text("name,bus,p_set,q_set\n")
    net = PyPSAEurLoader(tmp_path).load()
    assert len(net.ext_grid) == 2  # one slack promoted per component


def test_engine_runs_on_pypsa_dataset(pypsa_csv):
    """The portability cornerstone: the unchanged Phase 1 N-1 engine runs on a string-named PyPSA
    network (via the int bus map), no case118 assumptions."""
    from src.engine.network import base_case_summary
    from src.engine.scan import build_contingency_set, rank, run_contingency_sweep

    net = PyPSAEurLoader(pypsa_csv).load()
    assert base_case_summary(net)["converged"] is True
    cont = build_contingency_set(net)
    assert len(cont) >= 1
    sweep = rank(run_contingency_sweep(net))
    assert len(sweep) == len(cont)
    # losing the single slack is the topological blackout, exactly as on case118
    assert any(r.contingency_id.startswith("ext_grid") and r.status == "FULL_BLACKOUT" for r in sweep)


def test_missing_buses_csv_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="buses.csv"):
        PyPSAEurLoader(tmp_path).load()


def test_case118_demo_scenario_is_incompatible_with_pypsa(pypsa_csv):
    from src.engine.scenarios import (
        apply_scenario,
        default_scenario_for_net,
        scenario_is_compatible,
    )

    net = PyPSAEurLoader(pypsa_csv).load()
    assert not scenario_is_compatible(net, "demo_congestion")
    assert default_scenario_for_net(net) == "calm"
    with pytest.raises(ValueError, match="case118-specific"):
        apply_scenario(net, "demo_congestion")


def test_app_state_starts_pypsa_dataset_on_calm(pypsa_csv):
    from src.api.state import AppState
    from src.config import Settings

    state = AppState(Settings(grid_dataset="pypsa_eur", grid_data_path=str(pypsa_csv)))
    assert state.scenario_id == "calm"
    assert state.net["grid_dataset"] == "pypsa_eur"
    assert len(state.baseline) == len(state.contingencies)


def test_dynamic_topology_uses_pypsa_loaded_net(pypsa_csv):
    from src.api.main import _topology_from_net

    net = PyPSAEurLoader(pypsa_csv).load()
    topo = _topology_from_net(net)
    assert len(topo["buses"]) == 3
    assert len(topo["edges"]) == 2
    assert {b["label"] for b in topo["buses"]} == {"DE_380", "FR_380", "DE_220"}
