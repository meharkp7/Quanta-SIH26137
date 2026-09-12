# Source map

Read this folder first. The UI and demo scripts should import `src.platform`, not reach into optimizer internals.

| Folder | What it is | Who uses it |
|---|---|---|
| `contracts/` | Shared JSON schemas: scenario, routes, forecasts, decisions | Everyone |
| `data/` | CVRP parser, synthetic roads, traffic datasets | Dataset generation |
| `routing/` | Shortest paths, evaluator, encoding, repair, validator | All solvers |
| `optim/` | QPSO, PSO, ALNS, MILP, traces | Search and benchmarks |
| `sim/` | SUMO export, TraCI, closures, fixture runner | Execution |
| `platform/` | **Front door for the demo UI** | `app/`, `runtime/` |
| `runtime/` | Observe → forecast baseline → scope rule → solve → validate | Dispatcher loop |
| `learning/` | Reserved for GNN / Transformer / PPO | Not trained yet |

Do not add a second `Scenario` class or a second feasibility checker. `RouteEvaluator` / `evaluate_scenario` are the source of truth.
