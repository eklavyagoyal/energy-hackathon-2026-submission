"""Grid2Op adapter (U14): a thin, fully isolated bridge between Grid2Op and Warden's view.

KTD7 (isolation, deliberate): Grid2Op pins pandapower < 3 historically, which CONFLICTS with this
repo's pandapower 3.4. It therefore cannot live in the same environment; running L2RPN means a
SEPARATE venv with grid2op + lightsim2grid. So everything here is import-guarded: the module loads
fine without grid2op, grid2op_available() reports the truth, and the mapping helpers are duck-typed
so they are unit-testable against a fake observation with no grid2op installed.

Mapping conventions:
  - Grid2Op observation.rho is line loading as a ratio (1.0 == at thermal limit), so
    loading_percent = rho * 100 (KTD7).
  - element ids are integers (KTD7); Warden's native_index is int|str and accepts them unchanged.
  - a Warden redispatch Action maps to a Grid2Op {"redispatch": [(gen_id, delta_mw), ...]} action,
    delta = to - from on each gen p_mw move.
"""

from __future__ import annotations

import importlib.util
import logging

logger = logging.getLogger(__name__)


def grid2op_available() -> bool:
    """True only if grid2op can be imported in THIS environment. Expected False in the main repo
    venv (pandapower 3.4); True only in a dedicated grid2op venv."""
    return importlib.util.find_spec("grid2op") is not None


def observation_to_grid_state(obs) -> dict:
    """Map a Grid2Op observation to a Warden-flavored grid-state dict. Duck-typed: any object with
    rho / line_status / load_p / gen_p works, so this is testable without grid2op. rho is a ratio,
    so loading_percent = rho * 100. Line and gen ids are kept as integers."""
    rho = list(getattr(obs, "rho", []) or [])
    status = list(getattr(obs, "line_status", []) or [])
    load_p = list(getattr(obs, "load_p", []) or [])
    gen_p = list(getattr(obs, "gen_p", []) or [])

    line_loadings: dict = {}
    lines_out: list = []
    n_overloads = 0
    for i, r in enumerate(rho):
        in_service = bool(status[i]) if i < len(status) else True
        if not in_service:
            lines_out.append(i)
            continue
        pct = float(r) * 100.0
        line_loadings[f"line_{i}"] = round(pct, 1)
        if pct > 100.0:
            n_overloads += 1

    return {
        "source": "grid2op",
        "time_step": int(getattr(obs, "current_step", getattr(obs, "time_step", 0)) or 0),
        "n_line": len(rho),
        "line_loadings": line_loadings,
        "lines_out_of_service": lines_out,
        "n_overloads": n_overloads,
        "max_loading_pct": round(max((float(r) * 100.0 for r in rho), default=0.0), 1),
        "total_load_mw": round(float(sum(load_p)), 1),
        "total_gen_mw": round(float(sum(gen_p)), 1),
    }


def agent_action_to_grid2op(action: dict, action_space):
    """Convert a Warden redispatch Action (dict) into a Grid2Op action via the env's action_space.
    Only p_mw generator moves map (Grid2Op redispatch is in MW deltas); voltage setpoints and load
    shed have no direct Grid2Op redispatch equivalent and are dropped (logged). gen ids are ints."""
    deltas: list = []
    for c in action.get("changes", []):
        if c.get("etype") == "gen" and c.get("field") == "p_mw":
            try:
                gen_id = int(c["index"])
                delta = float(c["to"]) - float(c["from"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(delta) > 1e-6:
                deltas.append((gen_id, delta))
    if not deltas:
        return action_space({})  # do-nothing action
    return action_space({"redispatch": deltas})
