"""L2RPN episode runner (U15): drive a Grid2Op episode with the Warden decision view.

OPTIONAL and ISOLATED (KTD7). This needs a SEPARATE environment with grid2op + lightsim2grid (they
pin pandapower < 3 and cannot coexist with this repo's pandapower 3.4). In the main venv,
run_l2rpn_episode raises a clear, actionable RuntimeError instead of failing obscurely. The loop
uses Grid2Op's native 4-tuple env.step contract (obs, reward, done, info), not the gymnasium 5-tuple.

What it demonstrates: the same observe -> decide -> act shape as the time-stepped simulator, but on
the L2RPN benchmark, so Warden's framing transfers to the standard competition environment. The
decision here is intentionally light (a topology-agnostic safe action); the redispatch verification
that is Warden's core lives on the pandapower engine, not in this thin runner.
"""

from __future__ import annotations

import logging

from src.integrations.grid2op_adapter import grid2op_available, observation_to_grid_state

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "grid2op is not importable in this environment. L2RPN needs a SEPARATE venv:\n"
    "  python -m venv .venv-grid2op && . .venv-grid2op/bin/activate\n"
    "  pip install grid2op lightsim2grid\n"
    "grid2op pins pandapower < 3, which conflicts with this repo's pandapower 3.4, so it is "
    "deliberately kept out of the main environment (KTD7)."
)


def run_l2rpn_episode(env_name: str = "l2rpn_case14_sandbox", max_steps: int = 100,
                      decide=None) -> dict:
    """Run one L2RPN episode and return a compact trace. Requires grid2op in this environment.

    decide(grid_state, obs, action_space) -> grid2op action; defaults to do-nothing (a safe,
    benchmark-valid baseline). Uses the native 4-tuple env.step.
    """
    if not grid2op_available():
        raise RuntimeError(_INSTALL_HINT)

    import grid2op  # type: ignore
    from grid2op.Agent import DoNothingAgent  # type: ignore

    try:
        from lightsim2grid import LightSimBackend  # type: ignore
        backend = LightSimBackend()
        env = grid2op.make(env_name, backend=backend)
    except Exception:  # pragma: no cover - depends on optional backend
        logger.warning("lightsim2grid backend unavailable; falling back to the default backend")
        env = grid2op.make(env_name)

    do_nothing = DoNothingAgent(env.action_space)
    obs = env.reset()
    steps: list = []
    total_reward = 0.0
    for t in range(max_steps):
        state = observation_to_grid_state(obs)
        if decide is not None:
            act = decide(state, obs, env.action_space)
        else:
            act = do_nothing.act(obs, 0.0, False)
        obs, reward, done, info = env.step(act)  # native 4-tuple (KTD7)
        total_reward += float(reward)
        steps.append({
            "t": t,
            "n_overloads": state["n_overloads"],
            "max_loading_pct": state["max_loading_pct"],
            "reward": round(float(reward), 3),
            "done": bool(done),
        })
        if done:
            break

    return {
        "env_name": env_name,
        "steps_survived": len(steps),
        "max_steps": max_steps,
        "total_reward": round(total_reward, 3),
        "blackout": bool(steps and steps[-1]["done"] and len(steps) < max_steps),
        "trace": steps,
    }
