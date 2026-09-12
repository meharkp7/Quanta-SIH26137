"""
Step 9 — five-job exact correctness reference.

This module provides a deliberately small, evaluator-grounded exact reference
for the hand-checkable Step 3 fixture.

The reference enumerates every assignment/order combination for the fixed
fleet, evaluates each RoutePlan with the project's independent RouteEvaluator,
and records the globally best feasible solution.  Because the search space is
exhaustively enumerated, the result is an exact optimum for the supplied
fixture under the evaluator's actual objective/feasibility semantics.

This is a correctness anchor for the later snapshot MILP; it is NOT a
scaling/performance benchmark.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.contracts.scenario import Scenario
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_plan import RoutePlan, VehicleRoute


@dataclass(frozen=True)
class ExhaustiveReferenceResult:
    scenario_id: str
    customer_count: int
    vehicle_count: int
    total_candidates: int
    feasible_candidates: int
    objective: float | None
    vehicle_routes: dict[str, list[str]] | None
    total_distance_m: float | None
    total_travel_time_s: float | None
    total_waiting_time_s: float | None
    total_service_time_s: float | None
    elapsed_s: float
    exact: bool
    evaluator_semantics: str


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate_json(path.read_text(encoding="utf-8"))


def enumerate_exact(
    scenario: Scenario,
    *,
    evaluator: RouteEvaluator | None = None,
) -> ExhaustiveReferenceResult:
    """Exhaustively enumerate all fixed-fleet logical route plans."""

    if len(scenario.requests) != 5:
        raise ValueError(
            "This correctness harness is intentionally restricted to the "
            "five-job fixture."
        )

    if len(scenario.fleet) != 2:
        raise ValueError(
            "This correctness harness expects the fixture's two-vehicle fleet."
        )

    evaluator = evaluator or RouteEvaluator(scenario)

    customer_ids = tuple(str(r.request_id) for r in scenario.requests)
    vehicle_ids = tuple(str(v.vehicle_id) for v in scenario.fleet)

    # A permutation gives a total visit ordering.  The bit mask assigns each
    # position to one of the two fixed vehicles.  Every assignment/order is
    # therefore represented exactly once.
    started = time.perf_counter()

    total = 0
    feasible = 0
    best_eval: Any | None = None
    best_plan: RoutePlan | None = None

    for permutation in itertools.permutations(customer_ids):
        for mask in range(1 << len(customer_ids)):
            routes = [[], []]

            for position, customer_id in enumerate(permutation):
                vehicle_index = (mask >> position) & 1
                routes[vehicle_index].append(customer_id)

            plan = RoutePlan.from_routes(
                [
                    VehicleRoute.from_sequence(vehicle_ids[0], routes[0]),
                    VehicleRoute.from_sequence(vehicle_ids[1], routes[1]),
                ]
            )

            result = evaluator.evaluate(plan)
            total += 1

            if not result.feasible:
                continue

            feasible += 1

            if (
                best_eval is None
                or float(result.objective_value)
                < float(best_eval.objective_value) - 1e-9
            ):
                best_eval = result
                best_plan = plan

    elapsed = time.perf_counter() - started

    if best_eval is None or best_plan is None:
        return ExhaustiveReferenceResult(
            scenario_id=str(scenario.scenario_id),
            customer_count=len(customer_ids),
            vehicle_count=len(vehicle_ids),
            total_candidates=total,
            feasible_candidates=feasible,
            objective=None,
            vehicle_routes=None,
            total_distance_m=None,
            total_travel_time_s=None,
            total_waiting_time_s=None,
            total_service_time_s=None,
            elapsed_s=elapsed,
            exact=True,
            evaluator_semantics="RouteEvaluator",
        )

    return ExhaustiveReferenceResult(
        scenario_id=str(scenario.scenario_id),
        customer_count=len(customer_ids),
        vehicle_count=len(vehicle_ids),
        total_candidates=total,
        feasible_candidates=feasible,
        objective=float(best_eval.objective_value),
        vehicle_routes={
            str(route.vehicle_id): [str(cid) for cid in route.customer_ids]
            for route in best_plan.vehicle_routes
        },
        total_distance_m=float(best_eval.total_distance_m),
        total_travel_time_s=float(best_eval.total_travel_time_s),
        total_waiting_time_s=float(best_eval.total_waiting_time_s),
        total_service_time_s=float(best_eval.total_service_time_s),
        elapsed_s=elapsed,
        exact=True,
        evaluator_semantics="RouteEvaluator",
    )


def write_result(result: ExhaustiveReferenceResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exhaustive exact reference for the five-job fixture."
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("fixtures/step3/base/scenario.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/step9_research/five_job_exhaustive_reference.json"
        ),
    )
    args = parser.parse_args()

    # Make execution from the repository root robust when invoked directly.
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    scenario = load_scenario(args.scenario)
    result = enumerate_exact(scenario)
    write_result(result, args.output)

    print("\nStep 9 — five-job exhaustive exact reference")
    print("=" * 52)
    print(f"Scenario:          {result.scenario_id}")
    print(f"Customers:         {result.customer_count}")
    print(f"Vehicles:          {result.vehicle_count}")
    print(f"Candidates:        {result.total_candidates}")
    print(f"Feasible:          {result.feasible_candidates}")
    print(f"Exact:             {result.exact}")
    print(f"Elapsed:           {result.elapsed_s:.3f} s")

    if result.objective is None:
        print("Best objective:    INFEASIBLE")
    else:
        print(f"Best objective:    {result.objective:.6f}")
        print(f"Distance:          {result.total_distance_m:.6f} m")
        print(f"Travel time:       {result.total_travel_time_s:.6f} s")
        print(f"Waiting time:      {result.total_waiting_time_s:.6f} s")
        print(f"Service time:      {result.total_service_time_s:.6f} s")
        print("Route:")
        for vehicle_id, customers in result.vehicle_routes.items():
            print(f"  {vehicle_id}: {' -> '.join(customers) if customers else '(empty)'}")

    print(f"Artifact:          {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
