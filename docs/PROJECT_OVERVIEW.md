# SIH26137 — what we are building

Official problem: [SIH26137](https://sih2026.vuce.in/ps/SIH26137)  
Listing: [sih.gov.in/sih2026PS](https://www.sih.gov.in/sih2026PS)  
Organisation: Egreen Quanta · Category: Transportation & Logistics · Deadline listed as 30 September 2026

## What the problem statement wants

Urban routing is a large Vehicle Routing Problem. Exact methods do not scale. Real quantum computers cannot run this at city size yet. SIH asks for a **software platform** that:

1. Models the city as a **weighted graph**
2. Uses a **quantum-inspired** method (QPSO) on a normal computer
3. Builds **near-optimal vehicle routes** under simulated or live traffic
4. Minimizes **time, distance, and congestion**
5. Handles **constraints**
6. Shows **convergence**
7. **Benchmarks** QPSO against classical metaheuristics and exact methods
8. Works for **VRP and shortest-path**, and can scale toward smart-city logistics

Expected solution: graph model + math formulation + constraint handling + QPSO + convergence analysis + systematic benchmarks + a usable platform/UI.

This is **not** a request for quantum hardware.

## What we are building

**Adaptive Quantum-Inspired Vehicle Routing under Dynamic Traffic.**

```
roads + jobs
    -> independent validator
    -> QPSO (and PSO / ALNS / tiny MILP)
    -> legal RoutePlan
    -> SUMO execution
    -> dispatcher UI
```

The full research architecture also has GNN–Transformer forecasts and PPO scope control. Those are designed and contracted. They are **not** required to show a legal SIH demo. Until trained, the loop uses a persistence forecast and a rule policy, labelled as baselines.

## What is already done

| Piece | Evidence |
|---|---|
| Contracts, units, IDs | `src/contracts/` |
| 5-job / 2-vehicle fixture | `fixtures/step3/` |
| Road / CVRP dataset pipeline | `src/data/` |
| Math + independent evaluator | `docs/routing_snapshot_model.md`, `src/routing/` |
| SUMO fixture + closure | `src/sim/`, `python -m src.sim.fixture --gui` |
| QPSO + matched PSO | `src/optim/qpso.py`, `src/optim/pso.py` |
| ALNS + snapshot MILP | `src/optim/references.py` |
| Demo facade + loop | `src/platform/`, `src/runtime/` |
| Dispatcher UI | `app/` |
| Causal 6-episode labels (Step 11) | `src/data/causal_episodes.py` |
| Forecast windows + baselines (Step 12) | `src/learning/` |

## How to show it

From the repo root:

```powershell
python -m pip install -r requirements.txt
python -m app.server
```

Open http://127.0.0.1:8765

Live demo path: load fixture → Solve QPSO → watch routes and convergence → Close a road → Run loop → Show illegal plan → Benchmark → Run SUMO.

## Honest limits

- QPSO is quantum-**inspired**. It runs on CPU.
- “Near-optimal” is only claimed against the tiny MILP when that solve is certified.
- GNN, Transformer, and PPO are not trained. The UI says so.
- The live map is the 5-job fixture. Larger cases are offline / generator outputs.
