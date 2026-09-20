"""
Step 9 — matched-budget QPSO-vs-ALNS comparison on the 5-job fixture.

Matched controls (declared in the evidence JSON):
  - identical fixture (fixtures/step3/base/scenario.json);
  - identical snapshot costs: a fresh RouteEvaluator with the same
    distance + travel_time + waiting weights used by the MILP reference;
  - identical independent validator: src.routing.validator.evaluate_scenario
    (standalone Dijkstra implementation, independent of RouteEvaluator)
    plus a RouteEvaluator re-check of every best-feasible route;
  - identical evaluation budget (max_evaluations) and wall-clock limit;
  - identical optimizer seeds.

QPSO uses the shared Step7 2n continuous encoding/decoder plus bounded
Step7 repair through RouteFitnessOracle.  ALNS operates directly on discrete
fixed-fleet routes with its own destroy/repair operators, but every ALNS
candidate is scored by the same RouteEvaluator semantics, so feasibility and
objective comparisons are apples-to-apples.  The move operators are the
subject of the comparison, not a hidden variable.

Run from the repository root:
    python -m src.optim.qpso_vs_alns
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.contracts.scenario import Scenario
from src.optim.common import FitnessResult
from src.optim.qpso import (
    AdaptiveQPSO,
    AdaptiveQPSOConfig,
    RouteFitnessOracle,
)
from src.optim.references import (
    ALNSConfig,
    ALNSReference,
    ReferenceRoute,
)
from src.routing.route_encoding import Step7RouteEngine
from src.routing.route_evaluator import (
    RouteEvaluationConfig,
    RouteEvaluator,
)
from src.routing.route_plan import RoutePlan, VehicleRoute
from src.routing.validator import evaluate_scenario


DEFAULT_FIXTURE = Path("fixtures/step3/base/scenario.json")
DEFAULT_ARTIFACT = Path(
    "artifacts/step9_research/qpso_vs_alns_5job.json"
)
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_MAX_EVALUATIONS = 2000
DEFAULT_TIME_LIMIT_S = 5.0
DEFAULT_POPULATION = 10


def snapshot_config() -> RouteEvaluationConfig:
    """Snapshot costs shared by MILP, ALNS and QPSO on this fixture."""
    return RouteEvaluationConfig(
        distance_weight=1.0,
        travel_time_weight=1.0,
        waiting_time_weight=1.0,
        service_time_weight=0.0,
        default_start_time_s=0.0,
    )


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate_json(path.read_text())


@dataclass
class EvalTrace:
    """Per-evaluation recording wrapper state (shared by both methods)."""

    started: float = 0.0
    calls: int = 0
    first_feasible_eval: int | None = None
    first_feasible_time_s: float | None = None
    best_feasible: float | None = None
    # (eval_index, elapsed_s, best_feasible_so_far); appended on improvement.
    best_trace: list[tuple[int, float, float]] = field(
        default_factory=list
    )


class RecordingQPSOOracle:
    """Callable oracle wrapper recording first-feasible and trace data."""

    def __init__(
        self,
        inner: RouteFitnessOracle,
    ) -> None:
        self._inner = inner
        self.trace = EvalTrace(started=time.perf_counter())

    def __call__(
        self,
        position: Any,
    ) -> FitnessResult:
        result = self._inner(position)
        self.trace.calls += 1
        elapsed = time.perf_counter() - self.trace.started
        if result.feasible:
            if self.trace.first_feasible_eval is None:
                self.trace.first_feasible_eval = self.trace.calls
                self.trace.first_feasible_time_s = elapsed
            if (
                self.trace.best_feasible is None
                or float(result.fitness)
                < self.trace.best_feasible - 1e-12
            ):
                self.trace.best_feasible = float(result.fitness)
                self.trace.best_trace.append(
                    (
                        self.trace.calls,
                        elapsed,
                        float(result.fitness),
                    )
                )
        return result


def make_alns_evaluator(
    evaluator: RouteEvaluator,
    index_to_customer: dict[int, str],
    index_to_vehicle: dict[int, str],
    trace: EvalTrace,
) -> Callable[[ReferenceRoute], FitnessResult]:
    """Evaluator-backed ALNS oracle over the same snapshot costs."""

    def fn(route: ReferenceRoute) -> FitnessResult:
        mapping = route.as_dict()
        plan = RoutePlan.from_routes(
            [
                VehicleRoute.from_sequence(
                    index_to_vehicle[vehicle_id],
                    [
                        index_to_customer[c]
                        for c in mapping.get(vehicle_id, ())
                    ],
                )
                for vehicle_id in sorted(mapping)
            ]
        )
        evaluation = evaluator.evaluate(plan)
        result = FitnessResult(
            fitness=float(evaluation.objective_value),
            feasible=bool(evaluation.feasible),
        )
        trace.calls += 1
        elapsed = time.perf_counter() - trace.started
        if result.feasible:
            if trace.first_feasible_eval is None:
                trace.first_feasible_eval = trace.calls
                trace.first_feasible_time_s = elapsed
            if (
                trace.best_feasible is None
                or float(result.fitness)
                < trace.best_feasible - 1e-12
            ):
                trace.best_feasible = float(result.fitness)
                trace.best_trace.append(
                    (
                        trace.calls,
                        elapsed,
                        float(result.fitness),
                    )
                )
        return result

    return fn


def validate_route(
    scenario: Scenario,
    evaluator: RouteEvaluator,
    route: dict[str, list[str]],
) -> dict[str, Any]:
    """Independent validator verdict plus evaluator agreement check."""
    independent = evaluate_scenario(scenario, route)
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(vehicle_id, customers)
            for vehicle_id, customers in route.items()
        ]
    )
    checking = evaluator.evaluate(plan)
    return {
        "independent_validator_feasible": bool(independent.feasible),
        "independent_validator_all_served": bool(
            independent.all_requests_served
        ),
        "independent_validator_capacity": bool(
            independent.capacity_feasible
        ),
        "independent_validator_connectivity": bool(
            independent.connectivity_feasible
        ),
        "independent_validator_time_window": bool(
            independent.time_window_feasible
        ),
        "evaluator_feasible": bool(checking.feasible),
        "evaluator_objective": float(checking.objective_value),
        "verdict_agree": bool(independent.feasible)
        == bool(checking.feasible),
    }


def run_qpso_seed(
    *,
    scenario: Scenario,
    seed: int,
    max_evaluations: int,
    population_size: int,
) -> dict[str, Any]:
    evaluator = RouteEvaluator(scenario, config=snapshot_config())
    engine = Step7RouteEngine(scenario, evaluator)
    oracle = RecordingQPSOOracle(
        RouteFitnessOracle(
            engine,
            planning_time_s=0.0,
            repair=True,
            repair_penalty=0.0,
        )
    )
    config = AdaptiveQPSOConfig(
        dimensions=2 * len(scenario.requests),
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=population_size,
        max_evaluations=max_evaluations,
        seed=seed,
    )
    optimizer = AdaptiveQPSO(config, oracle)
    started = time.perf_counter()
    result = optimizer.optimize()
    wall_s = time.perf_counter() - started

    # Decode the reported best position through the shared engine/repair
    # path (one extra verification evaluation, outside the budget).
    candidate = engine.evaluate_keys(
        result.best_position,
        planning_time_s=0.0,
        repair=True,
    )
    route = {
        str(vr.vehicle_id): [str(c) for c in vr.customer_ids]
        for vr in candidate.repaired_plan.vehicle_routes
    }
    verdict = validate_route(scenario, evaluator, route)
    best_feasible = (
        float(result.best_fitness)
        if bool(result.best_result.feasible)
        else None
    )
    return {
        "method": "qpso",
        "seed": seed,
        "best_fitness": float(result.best_fitness),
        "best_feasible": bool(result.best_result.feasible),
        "best_feasible_objective": best_feasible,
        "evaluations": int(result.evaluations),
        "oracle_calls": int(oracle.trace.calls),
        "iterations": int(result.iterations),
        "wall_s": wall_s,
        "first_feasible_eval": oracle.trace.first_feasible_eval,
        "first_feasible_time_s": (
            oracle.trace.first_feasible_time_s
        ),
        "route": route,
        "history_best": [float(v) for v in result.history_best],
        "history_mean": [float(v) for v in result.history_mean],
        "best_feasible_trace": [
            {"eval": e, "time_s": t, "objective": o}
            for e, t, o in oracle.trace.best_trace
        ],
        "validator_verdict": verdict,
    }


def run_alns_seed(
    *,
    scenario: Scenario,
    seed: int,
    max_evaluations: int,
    time_limit_s: float,
) -> dict[str, Any]:
    evaluator = RouteEvaluator(scenario, config=snapshot_config())
    customers = [str(r.request_id) for r in scenario.requests]
    vehicles = [str(v.vehicle_id) for v in scenario.fleet]
    index_to_customer = dict(enumerate(customers))
    index_to_vehicle = dict(enumerate(vehicles))
    trace = EvalTrace(started=time.perf_counter())
    reference = ALNSReference(
        evaluator=make_alns_evaluator(
            evaluator,
            index_to_customer,
            index_to_vehicle,
            trace,
        ),
        customer_ids=list(range(len(customers))),
        vehicle_ids=list(range(len(vehicles))),
        config=ALNSConfig(
            max_evaluations=max_evaluations,
            time_limit_s=time_limit_s,
        ),
        seed=seed,
    )
    started = time.perf_counter()
    result = reference.run()
    wall_s = time.perf_counter() - started

    route: dict[str, list[str]] | None = None
    verdict: dict[str, Any] | None = None
    if result.route is not None:
        mapping = result.route.as_dict()
        route = {
            index_to_vehicle[vid]: [
                index_to_customer[c]
                for c in mapping.get(vid, ())
            ]
            for vid in sorted(mapping)
        }
        verdict = validate_route(scenario, evaluator, route)
    return {
        "method": "alns",
        "seed": seed,
        "best_fitness": (
            float(result.objective)
            if result.objective is not None
            else None
        ),
        "best_feasible": bool(result.feasible),
        "best_feasible_objective": (
            float(result.objective)
            if result.feasible and result.objective is not None
            else None
        ),
        "evaluations": int(result.evaluations),
        "oracle_calls": int(trace.calls),
        "iterations": int(result.iterations),
        "wall_s": wall_s,
        "first_feasible_eval": trace.first_feasible_eval,
        "first_feasible_time_s": trace.first_feasible_time_s,
        "route": route,
        "removal_statistics": dict(result.removal_statistics),
        "repair_statistics": dict(result.repair_statistics),
        "operator_weights": dict(result.operator_weights),
        "best_feasible_trace": [
            {"eval": e, "time_s": t, "objective": o}
            for e, t, o in trace.best_trace
        ],
        "validator_verdict": verdict,
    }


def run(
    *,
    fixture: Path = DEFAULT_FIXTURE,
    artifact: Path = DEFAULT_ARTIFACT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
    time_limit_s: float = DEFAULT_TIME_LIMIT_S,
    population_size: int = DEFAULT_POPULATION,
) -> dict[str, Any]:
    scenario = load_scenario(fixture)
    try:
        exact_objective: float | None = float(
            json.loads(
                Path(
                    "artifacts/step9_research"
                    "/five_job_milp_reference.json"
                ).read_text()
            )["objective"]
        )
    except (OSError, KeyError, ValueError):
        exact_objective = None

    qpso_runs = [
        run_qpso_seed(
            scenario=scenario,
            seed=seed,
            max_evaluations=max_evaluations,
            population_size=population_size,
        )
        for seed in seeds
    ]
    alns_runs = [
        run_alns_seed(
            scenario=scenario,
            seed=seed,
            max_evaluations=max_evaluations,
            time_limit_s=time_limit_s,
        )
        for seed in seeds
    ]

    def gap(value: float | None) -> float | None:
        if value is None or not exact_objective:
            return None
        return (value - exact_objective) / abs(exact_objective)

    table = []
    for qpso, alns in zip(qpso_runs, alns_runs):
        table.append(
            {
                "seed": qpso["seed"],
                "qpso_best_feasible": qpso[
                    "best_feasible_objective"
                ],
                "qpso_gap": gap(qpso["best_feasible_objective"]),
                "qpso_evals": qpso["evaluations"],
                "qpso_wall_s": qpso["wall_s"],
                "alns_best_feasible": alns[
                    "best_feasible_objective"
                ],
                "alns_gap": gap(alns["best_feasible_objective"]),
                "alns_evals": alns["evaluations"],
                "alns_wall_s": alns["wall_s"],
            }
        )

    evidence: dict[str, Any] = {
        "fixture": str(fixture),
        "controls": {
            "encoding": "QPSO: shared Step7 2n continuous "
            "encoding/decoder; ALNS: discrete fixed-fleet "
            "routes (move operators are the comparison "
            "subject)",
            "decoder": "shared Step7 decoder for QPSO; "
            "direct route construction for ALNS",
            "repair": "shared bounded Step7 repair for QPSO; "
            "ALNS greedy/regret insertion over the same "
            "evaluator",
            "evaluator": "fresh RouteEvaluator per run with "
            "distance + travel_time + waiting snapshot "
            "costs (identical to the MILP reference)",
            "validator": "src.routing.validator.evaluate_scenario "
            "(independent Dijkstra) plus RouteEvaluator "
            "re-check",
            "eval_budget_per_run": max_evaluations,
            "wall_clock_limit_s": time_limit_s,
            "qpso_population_size": population_size,
            "seeds": [int(s) for s in seeds],
        },
        "exact_reference_objective": exact_objective,
        "comparison_table": table,
        "qpso_runs": qpso_runs,
        "alns_runs": alns_runs,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched-budget QPSO-vs-ALNS on the 5-job fixture."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS)
    parser.add_argument("--time-limit-s", type=float, default=DEFAULT_TIME_LIMIT_S)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    args = parser.parse_args()
    evidence = run(
        fixture=args.fixture,
        artifact=args.artifact,
        seeds=tuple(int(s) for s in args.seeds),
        max_evaluations=args.max_evaluations,
        time_limit_s=args.time_limit_s,
        population_size=args.population,
    )
    print("Step 9 — matched-budget QPSO vs ALNS (5-job)")
    print("=" * 50)
    for row in evidence["comparison_table"]:
        print(
            f"seed={row['seed']}: "
            f"QPSO={row['qpso_best_feasible']} "
            f"(gap={row['qpso_gap']}) "
            f"[{row['qpso_evals']} evals, {row['qpso_wall_s']:.2f}s] | "
            f"ALNS={row['alns_best_feasible']} "
            f"(gap={row['alns_gap']}) "
            f"[{row['alns_evals']} evals, {row['alns_wall_s']:.2f}s]"
        )
    print(f"Artifact: {args.artifact}")


if __name__ == "__main__":
    main()
