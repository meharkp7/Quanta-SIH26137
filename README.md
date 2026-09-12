# Quanta · SIH26137

**Adaptive Quantum-Inspired Vehicle Routing under Dynamic Traffic**

Official problem: [SIH26137](https://sih2026.vuce.in/ps/SIH26137) · [SIH 2026 PS list](https://www.sih.gov.in/sih2026PS)

This repository is a software platform that plans delivery routes on a weighted road graph with **QPSO** (quantum-inspired, classical hardware), checks every plan with an independent validator, compares classical and exact solvers, and executes the tiny city fixture in SUMO.

## Show the demo

```powershell
cd Quanta-SIH26137
python -m pip install -r requirements.txt
python -m app.server
```

Open **http://127.0.0.1:8765**

1. **Solve routes** — QPSO colors two vehicle paths and draws a convergence curve  
2. **Show illegal plan** — overloaded V1 turns constraint chips red  
3. **Close a road** + **Run loop** — persistence forecast + rule scope + re-solve  
4. **Benchmark** — constructive / QPSO / PSO / ALNS / tiny MILP  
5. **Run SUMO** — launches `sumo-gui` if SUMO is installed  

Headless SUMO (no GUI):

```powershell
python -m src.sim.fixture
```

## What SIH wants vs what this repo is

| SIH ask | In this repo |
|---|---|
| Weighted transport graph | `src/contracts/`, `src/data/`, map in the UI |
| Math + constraints | `docs/routing_snapshot_model.md`, `src/routing/validator.py` |
| QPSO engine | `src/optim/qpso.py` |
| Classical + exact benchmarks | PSO, ALNS, tiny MILP, Dijkstra |
| Convergence + quality | UI chart + `src/optim` traces |
| Executable platform | `app/` dispatcher |
| Simulated traffic | SUMO fixture + close-road control |

Long architecture plan: [`plan.md`](plan.md)  
Step list: [`SIH26137_step_by_step_implementation_plan (1).md`](SIH26137_step_by_step_implementation_plan%20(1).md)  
Plain-language overview: [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)  
Code map: [`src/README.md`](src/README.md)

## Project shape

```
app/            dispatcher UI + API
src/platform/   the only module the UI should import
src/runtime/    demo operating loop
src/routing/    paths, evaluator, encoding, repair
src/optim/      QPSO, PSO, ALNS, MILP
src/sim/        SUMO
src/data/       graph / CVRP generation
src/contracts/  shared schemas
fixtures/       5-job hand-checked city
```

GNN, Transformer, and PPO are designed but **not trained**. The UI labels persistence forecasts and rule policies as baselines.

## Tests

```powershell
python -m src.data.generate_dataset causal-pilot fixtures/step3/base/scenario.json artifacts/step11_pilot
python -m src.data.generate_dataset forecast-windows artifacts/step11_pilot artifacts/step12_windows fixtures/step3/base/scenario.json
python -m pytest tests/step11 tests/step12 tests/test_platform_demo.py -q
```
