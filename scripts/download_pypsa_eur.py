"""Acquire a small PyPSA sample network and export it as a CSV folder for PyPSAEurLoader (U13).

Run: python scripts/download_pypsa_eur.py [dest_dir]   (from an activated .venv; see README)

PyPSA-Eur's prebuilt networks are large and gated behind a Snakemake build, which is too heavy for
a hackathon checkout. This script instead materializes a small, real PyPSA network and writes it in
the CSV-folder layout the loader reads. Order of preference:

  1. pypsa.examples.ac_dc_meshed() if pypsa is installed (a genuine AC+DC test network).
  2. otherwise, write a hand-built 6-bus two-voltage-level CSV export so the loader path is
     exercisable with zero extra dependencies.

For the full PyPSA-Eur dataset, follow https://pypsa-eur.readthedocs.io and point
GRID_DATASET=pypsa_eur GRID_DATA_PATH=<that network> (a .nc file or a CSV-folder export).
"""

from __future__ import annotations

import sys
from pathlib import Path

DEST_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "pypsa_sample"


def _via_pypsa(dest: Path) -> bool:
    try:
        import pypsa  # type: ignore
    except ImportError:
        return False
    print("pypsa is installed; exporting pypsa.examples.ac_dc_meshed() ...")
    n = pypsa.examples.ac_dc_meshed()
    dest.mkdir(parents=True, exist_ok=True)
    n.export_to_csv_folder(str(dest))
    print(f"  wrote a real PyPSA AC/DC network to {dest}")
    return True


def _handbuilt(dest: Path) -> None:
    """A 6-bus, two-voltage-level network with string bus names, written as a PyPSA CSV export."""
    print("pypsa not installed; writing a hand-built PyPSA CSV sample (no extra deps) ...")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "buses.csv").write_text(
        "name,v_nom,x,y,country\n"
        "DE_N_380,380,9.8,53.5,DE\n"
        "DE_S_380,380,11.4,48.4,DE\n"
        "FR_380,380,7.2,48.8,FR\n"
        "DE_N_220,220,9.6,53.6,DE\n"
        "DE_S_220,220,11.6,48.2,DE\n"
        "FR_220,220,7.0,48.7,FR\n"
    )
    (dest / "lines.csv").write_text(
        "name,bus0,bus1,r,x,s_nom,v_nom\n"
        "L_DEN_DES,DE_N_380,DE_S_380,1.8,18.0,2500,380\n"
        "L_DES_FR,DE_S_380,FR_380,2.4,24.0,2000,380\n"
        "L_DEN_FR,DE_N_380,FR_380,3.0,30.0,1500,380\n"
        "L_220_DE,DE_N_220,DE_S_220,1.2,9.0,180,220\n"
    )
    (dest / "transformers.csv").write_text(
        "name,bus0,bus1,r,x,s_nom\n"
        "T_DEN,DE_N_380,DE_N_220,0.008,0.11,1200\n"
        "T_DES,DE_S_380,DE_S_220,0.008,0.11,1200\n"
        "T_FR,FR_380,FR_220,0.009,0.12,1000\n"
    )
    (dest / "generators.csv").write_text(
        "name,bus,control,p_set,p_nom,carrier,type\n"
        "slack_DE_N,DE_N_380,Slack,800,4000,gas,CCGT\n"
        "ccgt_DE_S,DE_S_380,PV,600,1500,gas,CCGT\n"
        "nuclear_FR,FR_380,PV,900,1600,nuclear,nuclear\n"
        "wind_DE_N220,DE_N_220,PV,200,600,onwind,onshore wind\n"
        "solar_DE_S220,DE_S_220,PV,150,700,solar,solar PV\n"
    )
    (dest / "loads.csv").write_text(
        "name,bus,p_set,q_set\n"
        "ld_DE_S220,DE_S_220,700,90\n"
        "ld_FR_220,FR_220,650,80\n"
        "ld_DE_N220,DE_N_220,300,40\n"
    )
    print(f"  wrote a 6-bus hand-built sample to {dest}")


def main() -> None:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else DEST_DEFAULT
    if not _via_pypsa(dest):
        _handbuilt(dest)

    # Verify the loader can read what we just wrote.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import warnings

    warnings.filterwarnings("ignore")
    import pandapower as pp

    from src.grid.pypsa_eur_loader import PyPSAEurLoader

    loader = PyPSAEurLoader(dest)
    net = loader.load()
    pp.runpp(net)
    print(f"\nloaded {len(net.bus)} buses, {len(net.line)} lines, {len(net.trafo)} trafos, "
          f"{len(net.ext_grid)} ext_grids | power flow converged: {net.converged}")
    print(f"set GRID_DATASET=pypsa_eur GRID_DATA_PATH={dest} to run the engine on it.")


if __name__ == "__main__":
    main()
