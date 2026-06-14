<div align="center">

# ⚡ Warden

### A grid operations agent that never gets the physics wrong.

**🏆 Winner · E.ON track "Grid Operation Agents" · [Energy Hack Munich 2026](https://luma.com/o6r06acc) · €1,000 prize**

![Winner](https://img.shields.io/badge/Energy_Hack_Munich_2026-Winner_·_%E2%82%AC1000-FFD700?style=for-the-badge&labelColor=1a1a1a)
![Track](https://img.shields.io/badge/E.ON-Grid_Operation_Agents-E2001A?style=for-the-badge&labelColor=1a1a1a)

![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Engine](https://img.shields.io/badge/engine-pandapower-1f8b4c?style=flat-square)
![Tests](https://img.shields.io/badge/tests-130_passing-success?style=flat-square)
![Solver verified](https://img.shields.io/badge/every_action-solver_verified-blue?style=flat-square)

<br/>

*Most grid automation makes you choose: a raw optimizer that is correct but opaque,*
*or an LLM that is explainable but unverified. Warden refuses the trade.*
*The LLM orchestrates a real pandapower power-flow solver, and every action it proposes*
*is re-verified on the full AC cascade before an operator ever sees it.*
*The model narrates. The physics decides.*

</div>

---

## At a glance

| | |
|---|---|
| 🏆 **Result** | Winner of the E.ON track, Energy Hack Munich 2026, €1,000 prize |
| 🧠 **The one idea** | The LLM **never computes physics**. Every number it speaks is read verbatim from solver JSON. |
| ✅ **Safety guarantee** | **Verify before commit**: no action is `applied` unless a full AC cascade rescan confirms the base case is N-1 secure |
| 🔬 **Physics per diagnosis** | Sweeps **all 187 N-1 contingencies**, runs OPF, re-runs the cascade on every offending contingency |
| ⏱️ **Real-time loop** | A full **24-hour** step-through verifies each hour and still finishes in **~18s**, under the 30s budget |
| 💪 **Hardest case cleared** | Redispatched **1,794 MW across 50 generators** to clear a **168% line overload** and hold N-1 |
| 📊 **Honest baselines** | Head-to-head against rule-based greedy and AC-OPF on the *same* cascade scoring. We state plainly where OPF beats us. |
| 🔌 **Portable** | One dataset seam: IEEE 118-bus to PyPSA-Eur to a real TSO export, with **zero hardcoded bus indices** |
| 🧱 **Built** | ~6,400 lines of Python, **130 tests**, every demo number solver-produced and regenerable |

---

## The core idea

Two ways to automate a control room, and each one fails on its own:

- A **raw AC-OPF** is mathematically optimal but it hands an operator a vector of setpoints with no reason attached, and pandapower has no SCOPF, so its optimum can still leave an N-1 contingency that cascades.
- An **LLM** can explain itself in plain language, but if it is the thing computing the megawatts, it can hallucinate a number that looks plausible and is wrong. On a power grid, that is not a typo. That is a blackout.

Warden runs both in sequence and lets neither do the other's job. The agent **orchestrates** the solver: it calls `runpp`, sweeps contingencies, calls `runopp`, and then sends every proposed action back through the full AC cascade rescan before it is committed. The model only ever sees JSON tool results. It never touches a pandapower net, and it never does arithmetic on physics.

The payoff is the one thing neither a raw solver nor a rule-based system can give you at the same time: **a solver-verified secure guarantee AND a plain-language explanation of why.**

This maps straight onto the E.ON challenge brief (compare an LLM-driven approach against a rule-based or optimization baseline) and onto the X-GridAgent blueprint the brief points to (arXiv:2512.20789), built end to end.

---

## Why this won

1. **It cannot lie about the grid.** The `commit_status` an operator sees (`applied`, `noop`, `rejected_infeasible`) is the solver's verdict, never the model's claim. When an action cannot reach secure, Warden says so out loud instead of pretending. Judges trust a system that admits failure.
2. **It is verified, not just optimized.** The agent approximates SCOPF with a tighten-and-verify outer loop: tighten `max_loading_percent` on the lines implicated in offending cascades, re-solve OPF on a copy, re-run the full cascade, repeat. Every claim ships with the solver run that proves it.
3. **It is honest about its baselines.** We do not hide behind a strawman. The AC-OPF baseline is the serious one, and the README and the demo both say plainly where it wins (base-case cost, speed). We win on the axes a real operator cares about: N-1 cascade security, explainability, and edge cases like slack loss and islanding.
4. **It is portable to a real network.** Swap `GRID_DATASET` and the whole stack moves from the IEEE 118-bus case to a PyPSA-Eur sample to a real TSO export. No bus index is hardcoded anywhere, so the geographic stress scenarios (Dunkelflaute, Solar Peak South, Heatwave Derating) light up automatically on real data.
5. **Every number is real.** Each figure in this README and in the demo is solver-produced and regenerable from `make freeze-*`. Nothing is illustrative unless it says so.

---

## What it does

### Real-time decision loop

A 24-hour, time-stepped simulation where, at every hour, the agent observes the live solver state, proposes a remediation, and the action is verified on the full AC cascade before it would be committed.

- **Secure hours** produce a no-op with narration of the standing N-1 risk
- **Peak hours** (up to **19 lines** over their thermal limit at the evening peak) run OPF redispatch, verify on the cascade, and commit only if the base case reaches secure
- **Infeasible hours** are reported honestly as `rejected_infeasible`

Every step carries a `commit_status`. The grid never lies about what the solver found. The timeline shows a bimodal daily load curve (overnight trough ~02:00, morning peak ~08:00, midday dip, evening peak ~19:00) matching real European demand, on top of the IEEE 118-bus case.

### Agent plan

Single-snapshot diagnosis: run power flow, sweep **all 187 N-1 contingencies**, run OPF, verify on the full cascade. Returns a tool trace (every solver call timed), a structured proof (violations resolved, MW of cascade load shedding avoided), and a 3-sentence control-room explanation told as a before/after/how story.

Toggle "LLM narration" to swap the deterministic template for a live model call. Either way, every number in the narration is solver-produced.

### Baselines comparison

Three policies on the same scenario, scored by the *same* full cascade rescan. No policy gets a static-scan pass while another gets a cascade check. Apples to apples, always.

| Policy | Secures the base | N-1 cascade verified | Explains itself |
|---|:---:|:---:|:---:|
| Greedy (rule-based) | sometimes | never | no |
| AC-OPF (optimization) | yes | no | no |
| **Warden agent** | **yes** | **yes** | **yes** |

The agent uses the same OPF the optimization baseline does, then adds the two things a real operator needs: a solver-verified secure guarantee (base plus top N-1) and a plain-language explanation.

### Battery siting

Scores every candidate bus, drops a virtual battery into the top candidates, and re-runs the full N-1 cascade to verify the headroom improvement is real, not modeled. Returns cascades prevented and MW of load shedding avoided per candidate.

---

## Architecture

```
src/
  engine/        physics core - slack guard, islanding, iterative cascade,
                 contingency severity scoring, OPF remediation, load curtailment
  agent/         LLM tool-driving loop + deterministic narration fallback
  timeseries/    real-time simulator - load/event profiles, per-step verify-before-commit
  grid/          dataset seam - Case118Loader, PyPSAEurLoader, TSORealLoader
  battery/       candidate scoring, solver-verified counterfactual, narration
  api/           FastAPI app + routes (timeseries, agent, compare, battery) + console UI
  integrations/  Grid2Op / L2RPN adapter (import-guarded)

tests/           130 tests across engine, remediation, agent, real-time, datasets
fixtures/        frozen demo data (solver-produced, regenerable)
scripts/         freeze_realtime_demo.py, freeze_topology.py, download_pypsa_eur.py
```

The physics engine and the LLM are cleanly separated. `src/agent/loop.py` drives `src/agent/tools.py`, which wraps `src/engine/`. The model sees only JSON tool results. It never touches a pandapower net directly.

---

## Quick start

Python 3.11+.

```bash
make install   # creates .venv, installs all deps from requirements.txt
make test      # 130 tests, ~2 min
make serve     # starts on :8000, loads case118, caches the N-1 baseline at startup
```

Open `http://127.0.0.1:8000/` then **Enter the console**.

The console loads the frozen 24-step replay instantly. The Agent plan, Baselines, and Battery tabs run live against the engine. All three stay locked to the scenario selected on the Timeline tab.

Regenerate frozen artifacts (all numbers are solver-produced):

```bash
make freeze-topology    # 118-bus one-line layout (Kamada-Kawai)
make freeze-realtime    # 24-step timeline trace
make freeze-battery     # battery recommendation fixtures
```

---

## Deploying it

Warden is **one FastAPI application that also serves its own frontend**. The console (`src/api/static/`) is plain HTML and JavaScript with no build step, served by the same app that runs the physics. So:

- It is **not** a standalone static site. The Agent, Baselines, and Battery tabs call live endpoints (`/agent/plan`, `/timeseries/run`, `/compare`, `/battery/recommendations`) that run pandapower in Python. You cannot drop the `static/` folder on a CDN and have a working app. (The frozen 24-step Timeline replay *does* render from committed JSON, so a static host would show that one tab and nothing else.)
- The **full app deploys as a single container** anywhere that runs Python 3.11 with a long-lived process (Render, Railway, Fly.io, a VM, Kubernetes). One process, one port:

  ```bash
  pip install -r requirements.txt
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
  ```

  Set `ANTHROPIC_API_KEY` to enable live LLM narration; without it the deterministic narrator runs and the verify-before-commit guarantee is identical. The baseline N-1 sweep is cached once at startup, so the first boot does real work before the app is ready. Serverless or scale-to-zero platforms will pay that cold-start cost on every wake, so a single always-on instance is the right shape.

---

## Scenarios

Three scenarios on the IEEE 118-bus case:

| Scenario | Base state | What the agent does |
|---|---|---|
| **Congestion (N-1 pocket)** | Base secure, 15 insecure N-1s | Precautionary redispatch to reduce N-1 cascade depth |
| **Peak overload** | 7 lines over thermal limit (worst at 168%) | Redispatch 1,794 MW across 50 generators, clear all violations, hold N-1 |
| **Calm (secure base)** | Secure, 13 low-severity N-1s | No-op |

---

## Dataset swap

The active grid is selected by `GRID_DATASET`, resolved only in `src/grid/loader.py`:

```bash
GRID_DATASET=case118                                         # default (IEEE 118-bus)
GRID_DATASET=pypsa_eur  GRID_DATA_PATH=data/pypsa_sample     # PyPSA-Eur sample
GRID_DATASET=tso_real   GRID_DATA_PATH=/path/to/grid.json    # real TSO export
```

`scripts/download_pypsa_eur.py` downloads a PyPSA-Eur sample to `data/pypsa_sample/`. On a real dataset, the Timeline switches to geographic stress scenarios: Winter Dunkelflaute, Solar Peak South, Heatwave Thermal Derating. No hardcoded bus indices anywhere.

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GRID_DATASET` | `case118` | active dataset |
| `GRID_DATA_PATH` | - | dataset path for pypsa_eur / tso_real |
| `ANTHROPIC_API_KEY` | - | enables live LLM narration (optional) |
| `WARDEN_AGENT_MODE` | - | set `llm` to let the model drive the tool loop |
| `BATTERY_NARRATION_MODE` | `template` | set `llm` for live battery narration |
| `STEP_MINUTES` | `60` | timeline step resolution in minutes |

Without an API key the deterministic template narrator runs. The verify-before-commit guarantee is identical either way.

---

## Verification protocol

Every proposed action goes through `src/engine/remediation.py::verify_action`:

1. Apply the proposed redispatch on a working copy (the live net is never mutated)
2. Re-run `pandapower.runpp` on the copy
3. Re-run the full AC cascade for each offending contingency
4. Compare base-case violations before and after, and compare per-contingency cascade depth and load shed
5. Return `verified=True` only if the base case is secure AND no screened contingency worsened

The `commit_status` in the UI is the solver's verdict, never the model's claim.

---

## Integrity properties

- **LLM never computes physics** - every number in the narration is extracted verbatim from solver JSON
- **Verify before commit** - no action is marked `applied` unless the cascade rescan confirms the base is secure
- **Honest rejection** - when the action cannot reach secure, `rejected_infeasible` is returned and shown
- **No cumulative drift** - load profiles apply `base * multiplier(t)` at each step, never cumulatively
- **Reproducible** - every demo figure is regenerable from `make freeze-*`, nothing is illustrative unless labeled

---

<div align="center">

Built by **Eklavya Goyal** and **Laurentin Harter**, winners of the E.ON track at [Energy Hack Munich 2026](https://luma.com/o6r06acc).

</div>
