"""
Step 9 — ALNS reference experiment on the five-job fixture.

The existing ALNS implementation in src.optim.references is used unchanged.
This runner supplies an adapter from its integer-only ReferenceRoute contract
to the project's actual string-ID RoutePlan contract and evaluates every
candidate through RouteEvaluator.

The experiment records:
  - best feasible objective
  - feasibility
  - evaluation/iteration counts
  - runtime
  - route
  - gap to the certified five-job optimum (1575.0)
  - removal/repair operator usage
  - final adaptive operator weights

Run:
    python run_step9_alns.py --evaluations 2000 --time-limit 5 --seeds 1 2 3 4 5 6 7 8 9 10

Smoke:
    python run_step9_alns.py --evaluations 100 --time-limit 2 --seeds 1
"""

from __future__ import annotations

# When this research driver is executed as a file from experiments/step9,
# Python puts that directory on sys.path rather than the repository root.
# Add the repo root explicitly so imports such as `src.contracts...` resolve.
import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from src.contracts.scenario import Scenario
from src.optim.common import FitnessResult
from src.optim.references import (
    ALNSConfig,
    ALNSReference,
    ReferenceMethod,
    ReferenceRoute,
)
from src.routing.route_evaluator import (
    RouteEvaluationConfig,
    RouteEvaluator,
)
from src.routing.route_plan import RoutePlan, VehicleRoute


DEFAULT_FIXTURE = Path("fixtures/step3/base/scenario.json")
DEFAULT_OUTPUT = Path("artifacts/step9_research/alns_five_job")
EXACT_OBJECTIVE = 1575.0
EPS = 1e-8


def load_scenario(path: Path) -> Scenario:
    if not path.exists():
        raise FileNotFoundError(f"fixture not found: {path}")
    return Scenario.model_validate_json(path.read_text(encoding="utf-8"))


def make_evaluator(scenario: Scenario) -> RouteEvaluator:
    return RouteEvaluator(
        scenario,
        config=RouteEvaluationConfig(
            distance_weight=1.0,
            travel_time_weight=1.0,
            waiting_time_weight=1.0,
            service_time_weight=0.0,
            default_start_time_s=0.0,
        ),
    )


class ReferenceRouteAdapter:
    """
    Bridges ALNS's integer IDs to the repository's actual Scenario IDs.

    Integer IDs are local to this experiment only:
        vehicle 0..K-1 -> actual vehicle IDs
        customer 0..N-1 -> actual request IDs

    No route semantics are changed by this mapping.
    """

    def __init__(
        self,
        scenario: Scenario,
        evaluator: RouteEvaluator,
    ) -> None:
        self.scenario = scenario
        self.evaluator = evaluator

        self.vehicle_ids = tuple(
            v.vehicle_id for v in scenario.fleet
        )
        self.customer_ids = tuple(
            r.request_id for r in scenario.requests
        )

        self.vehicle_to_int = {
            vehicle_id: index
            for index, vehicle_id
            in enumerate(self.vehicle_ids)
        }
        self.customer_to_int = {
            customer_id: index
            for index, customer_id
            in enumerate(self.customer_ids)
        }

        self.int_to_vehicle = {
            index: vehicle_id
            for vehicle_id, index
            in self.vehicle_to_int.items()
        }
        self.int_to_customer = {
            index: customer_id
            for customer_id, index
            in self.customer_to_int.items()
        }

    def to_route_plan(
        self,
        route: ReferenceRoute,
    ) -> RoutePlan:
        routes: list[VehicleRoute] = []

        route_map = route.as_dict()

        for vehicle_index, vehicle_id in enumerate(
            self.vehicle_ids
        ):
            customer_indices = route_map.get(
                vehicle_index,
                (),
            )

            customers = [
                self.int_to_customer[int(customer_index)]
                for customer_index in customer_indices
            ]

            routes.append(
                VehicleRoute.from_sequence(
                    vehicle_id,
                    customers,
                )
            )

        return RoutePlan.from_routes(routes)

    def to_reference_route(
        self,
        route_plan: RoutePlan,
    ) -> ReferenceRoute:
        routes: dict[int, tuple[int, ...]] = {}

        for vehicle_route in route_plan.vehicle_routes:
            vehicle_index = self.vehicle_to_int[
                vehicle_route.vehicle_id
            ]

            routes[vehicle_index] = tuple(
                self.customer_to_int[customer_id]
                for customer_id in vehicle_route.customer_ids
            )

        return ReferenceRoute.from_routes(routes)

    def fitness(
        self,
        route: ReferenceRoute,
    ) -> FitnessResult:
        plan = self.to_route_plan(route)
        evaluation = self.evaluator.evaluate(plan)

        signature = tuple(
            (
                self.vehicle_to_int[vehicle_route.vehicle_id],
                tuple(
                    self.customer_to_int[customer_id]
                    for customer_id in vehicle_route.customer_ids
                ),
            )
            for vehicle_route in plan.vehicle_routes
        )

        return FitnessResult(
            fitness=float(evaluation.objective_value),
            feasible=bool(evaluation.feasible),
            route_signature=signature,
            repair_distance=0.0,
        )

    def independently_verify(
        self,
        route: ReferenceRoute,
    ) -> dict[str, Any]:
        plan = self.to_route_plan(route)
        evaluation = self.evaluator.evaluate(plan)

        expected_customers = set(self.customer_ids)
        observed_customers = set(
            plan.all_customer_ids()
        )

        complete = (
            len(plan.all_customer_ids())
            == len(expected_customers)
            and observed_customers == expected_customers
            and len(plan.all_customer_ids())
            == len(observed_customers)
        )

        return {
            "feasible": bool(evaluation.feasible),
            "complete": bool(complete),
            "objective": float(evaluation.objective_value),
            "objective_components": {
                key: float(value)
                for key, value
                in evaluation.objective_components.items()
            },
            "route": {
                str(vehicle_route.vehicle_id): [
                    str(customer_id)
                    for customer_id
                    in vehicle_route.customer_ids
                ]
                for vehicle_route in plan.vehicle_routes
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 100 evaluations and 2 seconds unless explicitly overridden.",
    )

    return parser.parse_args()


def run_one(
    *,
    scenario: Scenario,
    evaluator: RouteEvaluator,
    adapter: ReferenceRouteAdapter,
    seed: int,
    max_evaluations: int,
    time_limit_s: float,
) -> dict[str, Any]:
    config = ALNSConfig(
        max_evaluations=max_evaluations,
        time_limit_s=time_limit_s,
        removal_fraction_min=0.10,
        removal_fraction_max=0.40,
        reaction_factor=0.20,
        segment_length=25,
        initial_temperature=1.0,
        cooling_rate=0.995,
        destroy_operators=(
            "random_removal",
            "worst_removal",
            "related_removal",
        ),
        repair_operators=(
            "greedy_insertion",
            "regret_insertion",
        ),
    )

    # Deliberately do not inject the certified optimal route as ALNS's start.
    # ALNS gets the reference class's deterministic fixed-fleet construction.
    solver = ALNSReference(
        evaluator=adapter.fitness,
        customer_ids=tuple(range(len(adapter.customer_ids))),
        vehicle_ids=tuple(range(len(adapter.vehicle_ids))),
        config=config,
        seed=seed,
    )

    result = solver.run()

    verified = None
    absolute_gap = None
    relative_gap = None

    if result.route is not None:
        verified = adapter.independently_verify(
            result.route
        )

        if (
            result.objective is not None
            and verified["feasible"]
        ):
            # The result returned by ALNS must agree with a fresh independent
            # evaluator call on the actual decoded RoutePlan.
            if abs(
                float(result.objective)
                - float(verified["objective"])
            ) > EPS:
                raise RuntimeError(
                    "ALNS result/evaluator objective mismatch: "
                    f"{result.objective} vs "
                    f"{verified['objective']}"
                )

            absolute_gap = (
                float(verified["objective"])
                - EXACT_OBJECTIVE
            )
            relative_gap = (
                absolute_gap / EXACT_OBJECTIVE
            )

    return {
        "seed": int(seed),
        "method": ReferenceMethod.ALNS.value,
        "best_feasible_objective": (
            float(result.objective)
            if result.objective is not None
            and result.feasible
            else None
        ),
        "feasible": bool(result.feasible),
        "evaluations": int(result.evaluations),
        "iterations": int(result.iterations),
        "elapsed_s": float(result.elapsed_s),
        "exact_reference_objective": EXACT_OBJECTIVE,
        "absolute_gap": absolute_gap,
        "relative_gap": relative_gap,
        "verified": verified,
        "removal_statistics": dict(
            result.removal_statistics
        ),
        "repair_statistics": dict(
            result.repair_statistics
        ),
        "operator_weights": dict(
            result.operator_weights
        ),
        "message": result.message,
    }


def aggregate(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    feasible = [
        run["best_feasible_objective"]
        for run in runs
        if run["best_feasible_objective"] is not None
    ]

    gaps = [
        run["relative_gap"]
        for run in runs
        if run["relative_gap"] is not None
    ]

    exact_hits = sum(
        1
        for run in runs
        if (
            run["best_feasible_objective"] is not None
            and abs(
                run["best_feasible_objective"]
                - EXACT_OBJECTIVE
            ) <= EPS
        )
    )

    return {
        "method": "alns",
        "runs": len(runs),
        "exact_reference_objective": EXACT_OBJECTIVE,
        "feasible_rate": (
            sum(run["feasible"] for run in runs)
            / len(runs)
            if runs
            else 0.0
        ),
        "exact_hit_rate": (
            exact_hits / len(runs)
            if runs
            else 0.0
        ),
        "best_feasible_objective_mean": (
            statistics.mean(feasible)
            if feasible
            else None
        ),
        "best_feasible_objective_std": (
            statistics.pstdev(feasible)
            if len(feasible) > 1
            else 0.0
            if feasible
            else None
        ),
        "relative_gap_mean": (
            statistics.mean(gaps)
            if gaps
            else None
        ),
        "relative_gap_std": (
            statistics.pstdev(gaps)
            if len(gaps) > 1
            else 0.0
            if gaps
            else None
        ),
        "evaluations_mean": (
            statistics.mean(
                [run["evaluations"] for run in runs]
            )
            if runs
            else None
        ),
        "elapsed_s_mean": (
            statistics.mean(
                [run["elapsed_s"] for run in runs]
            )
            if runs
            else None
        ),
        "operator_usage": {
            "removal": {
                name: sum(
                    run["removal_statistics"].get(name, 0)
                    for run in runs
                )
                for name in (
                    "random_removal",
                    "worst_removal",
                    "related_removal",
                )
            },
            "repair": {
                name: sum(
                    run["repair_statistics"].get(name, 0)
                    for run in runs
                )
                for name in (
                    "greedy_insertion",
                    "regret_insertion",
                )
            },
        },
    }


def write_artifacts(
    output: Path,
    runs: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    fixture: Path,
    max_evaluations: int,
    time_limit_s: float,
    seeds: list[int],
) -> None:
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "protocol": {
            "fixture": str(fixture),
            "customer_count": 5,
            "vehicle_count": 2,
            "evaluation_budget": max_evaluations,
            "time_limit_s": time_limit_s,
            "seeds": seeds,
            "destroy_operators": [
                "random_removal",
                "worst_removal",
                "related_removal",
            ],
            "repair_operators": [
                "greedy_insertion",
                "regret_insertion",
            ],
            "initialization": (
                "ALNSReference deterministic fixed-fleet "
                "round-robin construction"
            ),
            "evaluator": (
                "RouteEvaluator with distance + travel_time + waiting"
            ),
            "exact_reference_objective": EXACT_OBJECTIVE,
        },
        "summary": summary,
    }

    (output / "protocol.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (output / "runs.json").write_text(
        json.dumps(
            runs,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with (output / "runs.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "seed",
            "method",
            "best_feasible_objective",
            "feasible",
            "evaluations",
            "iterations",
            "elapsed_s",
            "exact_reference_objective",
            "absolute_gap",
            "relative_gap",
            "verified",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for run in runs:
            writer.writerow(
                {
                    key: run.get(key)
                    for key in fieldnames
                }
            )

    (output / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.evaluations <= 0:
        raise ValueError("--evaluations must be positive")

    if args.time_limit <= 0.0:
        raise ValueError("--time-limit must be positive")

    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))

    scenario = load_scenario(args.fixture)

    if len(scenario.requests) != 5:
        raise RuntimeError(
            f"Expected five-job fixture, got {len(scenario.requests)} requests"
        )

    if len(scenario.fleet) != 2:
        raise RuntimeError(
            f"Expected two vehicles, got {len(scenario.fleet)}"
        )

    evaluator = make_evaluator(scenario)
    adapter = ReferenceRouteAdapter(
        scenario,
        evaluator,
    )

    max_evaluations = (
        100 if args.smoke and args.evaluations == 2000
        else args.evaluations
    )
    time_limit_s = (
        2.0 if args.smoke and args.time_limit == 5.0
        else args.time_limit
    )

    print("Step 9 — five-job ALNS reference")
    print("=" * 52)
    print(
        f"Scenario:            "
        f"{len(scenario.requests)} customers | "
        f"{len(scenario.fleet)} vehicles"
    )
    print(f"Evaluation budget:    {max_evaluations}")
    print(f"Time limit / run:     {time_limit_s:.3f} s")
    print(f"Seeds:                {seeds}")
    print("Destroy operators:    random, worst, related")
    print("Repair operators:     greedy, regret")
    print(
        "Exact reference:      "
        f"{EXACT_OBJECTIVE:.6f}"
    )
    print()

    runs: list[dict[str, Any]] = []

    for seed in seeds:
        print(
            f"Running ALNS seed={seed}...",
            flush=True,
        )

        run = run_one(
            scenario=scenario,
            evaluator=evaluator,
            adapter=adapter,
            seed=seed,
            max_evaluations=max_evaluations,
            time_limit_s=time_limit_s,
        )

        runs.append(run)

        objective = run["best_feasible_objective"]

        print(
            f"  best={objective} | "
            f"feasible={run['feasible']} | "
            f"evals={run['evaluations']} | "
            f"iterations={run['iterations']} | "
            f"time={run['elapsed_s']:.3f}s"
        )

    summary = aggregate(runs)

    write_artifacts(
        args.output,
        runs,
        summary,
        fixture=args.fixture,
        max_evaluations=max_evaluations,
        time_limit_s=time_limit_s,
        seeds=seeds,
    )

    print()
    print("Summary")
    print("-" * 52)
    print(
        f"Feasible rate:        "
        f"{summary['feasible_rate']:.3f}"
    )
    print(
        f"Exact-hit rate:       "
        f"{summary['exact_hit_rate']:.3f}"
    )
    print(
        f"Best objective mean:  "
        f"{summary['best_feasible_objective_mean']}"
    )
    print(
        f"Relative gap mean:    "
        f"{summary['relative_gap_mean']}"
    )
    print(
        f"Evaluations mean:     "
        f"{summary['evaluations_mean']}"
    )
    print(
        f"Runtime mean:         "
        f"{summary['elapsed_s_mean']}"
    )
    print()
    print("Operator usage")
    print(
        "  removal:",
        summary["operator_usage"]["removal"],
    )
    print(
        "  repair :",
        summary["operator_usage"]["repair"],
    )
    print()
    print(f"Artifacts:            {args.output}")


if __name__ == "__main__":
    main()
