#!/usr/bin/env python3
"""
Step 9 — 20-customer ALNS reference experiment.

Purpose
-------
Run the independent ALNS reference on the same 20-customer dynamic road
scenario used for Step 8 R2:

    scenario seed = 26157
    customers    = 20
    vehicles     = 4

The experiment keeps the routing semantics identical to the rest of the
project by evaluating every ALNS candidate through the real RouteEvaluator.

Important:
    There is NO certified optimum for this 20-customer instance.
    Therefore this runner reports best-feasible objectives and does not
    manufacture an "optimality gap".

The 5-job exact ALNS runner remains untouched.

Run from repository root:
    python experiments/step9/run_alns_20_customer.py --smoke

Full research run:
    python experiments/step9/run_alns_20_customer.py \
        --evaluations 2000 \
        --time-limit 5 \
        --seeds 1 2 3 4 5 6 7 8 9 10
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Make execution independent of the current working directory.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.scenario import Scenario
from src.data.dataset_generator import (
    DynamicDatasetConfig,
    generate_dynamic_scenario,
)
from src.optim.common import FitnessResult
from src.optim.references import ALNSConfig, ALNSReference, ReferenceRoute
from src.routing.initial_solution import (
    InitialSolutionBuilder,
    InitialSolutionConfig,
)
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_plan import RoutePlan, VehicleRoute


OUTPUT = (
    ROOT
    / "artifacts"
    / "step9_research"
    / "alns_20_customer"
)

SCENARIO_SEED = 26157
CUSTOMERS = 20
VEHICLES = 4
ROAD_JUNCTIONS = 196

DEFAULT_EVALUATIONS = 2000
DEFAULT_TIME_LIMIT = 5.0
DEFAULT_SEEDS = tuple(range(1, 11))


def make_config() -> DynamicDatasetConfig:
    """Exactly reproduce the Step 8 R2 scenario-generation family."""
    return DynamicDatasetConfig(
        customer_count=CUSTOMERS,
        road_junction_count=ROAD_JUNCTIONS,
        extent_m=5000.0,
        topology_family="grid",
        grid_rows=14,
        grid_cols=14,
        jitter_fraction=0.08,
        one_way_fraction=0.18,
        max_customer_snap_m=350.0,
        window_profile="medium",
        window_slack_s=900.0,
        service_duration_s=120.0,
        vehicle_capacity=30.0,
        fleet_size=VEHICLES,
        customer_spread=0.85,
        cluster_strength=0.20,
        minimum_road_segment_m=80.0,
        demand_min=1,
        demand_max=8,
        seed=SCENARIO_SEED,
        dataset_split="test",
    )


def build_scenario() -> Scenario:
    scenario = generate_dynamic_scenario(make_config())
    return Scenario.model_validate(
        scenario.model_dump(mode="json")
    )


def initial_solution_config() -> InitialSolutionConfig:
    return InitialSolutionConfig(
        heuristic="earliest_deadline",
        allow_unassigned=False,
        prioritize_time_windows=True,
        prioritize_distance=True,
        prioritize_travel_time=True,
        prioritize_waiting_time=True,
        require_capacity_feasibility=True,
        require_connectivity=True,
        require_time_window_feasibility=True,
        max_candidates_per_insertion=64,
        planning_time_s=0.0,
        include_future_released_requests=True,
    )


def build_feasible_initial_route(
    scenario: Scenario,
) -> tuple[RoutePlan, dict[str, int], dict[str, int]]:
    """
    Build one genuine feasible route using the project's constructive solver.

    ALNS uses integer IDs internally, so actual string IDs are mapped through
    explicit stable bijections. The mapping is only an adapter; RouteEvaluator
    still evaluates the reconstructed project RoutePlan with the original IDs.
    """
    evaluator = RouteEvaluator(scenario)

    builder = InitialSolutionBuilder(
        scenario,
        evaluator,
        config=initial_solution_config(),
    )
    initial = builder.build()

    if not initial.feasible or not initial.complete:
        raise RuntimeError(
            "20-customer constructive seed is not complete and feasible; "
            "do not run ALNS."
        )

    route_plan = initial.route_plan

    actual_customer_ids = [
        str(request.request_id)
        for request in scenario.requests
    ]
    actual_vehicle_ids = [
        str(vehicle.vehicle_id)
        for vehicle in scenario.fleet
    ]

    customer_to_int = {
        customer_id: index + 1
        for index, customer_id in enumerate(actual_customer_ids)
    }
    vehicle_to_int = {
        vehicle_id: index + 1
        for index, vehicle_id in enumerate(actual_vehicle_ids)
    }

    return route_plan, customer_to_int, vehicle_to_int


def route_plan_to_reference(
    route_plan: RoutePlan,
    customer_to_int: dict[str, int],
    vehicle_to_int: dict[str, int],
) -> ReferenceRoute:
    routes: dict[int, tuple[int, ...]] = {}

    for vehicle_route in route_plan.vehicle_routes:
        vehicle_key = str(vehicle_route.vehicle_id)
        vehicle_id = vehicle_to_int[vehicle_key]
        routes[vehicle_id] = tuple(
            customer_to_int[str(customer_id)]
            for customer_id in vehicle_route.customer_ids
        )

    # Keep every fleet vehicle represented, including unused vehicles.
    for vehicle_id in vehicle_to_int.values():
        routes.setdefault(vehicle_id, ())

    return ReferenceRoute.from_routes(routes)


def reference_to_route_plan(
    reference_route: ReferenceRoute,
    *,
    scenario: Scenario,
    int_to_customer: dict[int, str],
    int_to_vehicle: dict[int, str],
) -> RoutePlan:
    routes = []

    for vehicle_id, customers in reference_route.vehicles:
        actual_vehicle = int_to_vehicle[vehicle_id]
        actual_customers = tuple(
            int_to_customer[customer_id]
            for customer_id in customers
        )
        routes.append(
            VehicleRoute.from_sequence(
                actual_vehicle,
                actual_customers,
            )
        )

    return RoutePlan.from_routes(tuple(routes))


def make_evaluator(
    scenario: Scenario,
    customer_to_int: dict[str, int],
    vehicle_to_int: dict[str, int],
):
    """
    Adapter required by ALNSReference.

    ALNS supplies an integer-only ReferenceRoute. We reconstruct the actual
    RoutePlan and evaluate it with the project's independent RouteEvaluator.
    """
    evaluator = RouteEvaluator(scenario)

    int_to_customer = {
        value: key
        for key, value in customer_to_int.items()
    }
    int_to_vehicle = {
        value: key
        for key, value in vehicle_to_int.items()
    }

    def evaluate(reference_route: ReferenceRoute) -> FitnessResult:
        route_plan = reference_to_route_plan(
            reference_route,
            scenario=scenario,
            int_to_customer=int_to_customer,
            int_to_vehicle=int_to_vehicle,
        )

        result = evaluator.evaluate(route_plan)

        return FitnessResult(
            fitness=float(result.objective_value),
            feasible=bool(result.feasible),
            route_signature=tuple(
                (
                    vehicle_route.vehicle_id,
                    tuple(vehicle_route.customer_ids),
                )
                for vehicle_route in route_plan.vehicle_routes
            ),
            repair_distance=0.0,
        )

    return evaluate, evaluator, int_to_customer, int_to_vehicle


def validate_final_route(
    route: ReferenceRoute,
    *,
    evaluator: RouteEvaluator,
    scenario: Scenario,
    int_to_customer: dict[int, str],
    int_to_vehicle: dict[int, str],
) -> dict[str, Any]:
    route_plan = reference_to_route_plan(
        route,
        scenario=scenario,
        int_to_customer=int_to_customer,
        int_to_vehicle=int_to_vehicle,
    )

    evaluation = evaluator.evaluate(route_plan)

    expected_customers = {
        str(request.request_id)
        for request in scenario.requests
    }
    served_customers = {
        str(customer_id)
        for vehicle_route in route_plan.vehicle_routes
        for customer_id in vehicle_route.customer_ids
    }

    complete = served_customers == expected_customers
    unique = len(served_customers) == sum(
        len(vehicle_route.customer_ids)
        for vehicle_route in route_plan.vehicle_routes
    )

    return {
        "feasible": bool(evaluation.feasible),
        "complete": bool(complete),
        "unique_customers": bool(unique),
        "objective": float(evaluation.objective_value),
        "distance_m": float(evaluation.total_distance_m),
        "travel_time_s": float(evaluation.total_travel_time_s),
        "waiting_time_s": float(evaluation.total_waiting_time_s),
        "service_time_s": float(evaluation.total_service_time_s),
        "route": {
            str(vehicle_route.vehicle_id): [
                str(customer_id)
                for customer_id in vehicle_route.customer_ids
            ]
            for vehicle_route in route_plan.vehicle_routes
        },
    }


def run_one(
    scenario: Scenario,
    *,
    seed: int,
    evaluations: int,
    time_limit: float,
) -> dict[str, Any]:
    initial_plan, customer_to_int, vehicle_to_int = (
        build_feasible_initial_route(scenario)
    )

    initial_reference = route_plan_to_reference(
        initial_plan,
        customer_to_int,
        vehicle_to_int,
    )

    evaluate, evaluator, int_to_customer, int_to_vehicle = make_evaluator(
        scenario,
        customer_to_int,
        vehicle_to_int,
    )

    # Independent solver state per seed.
    config = ALNSConfig(
        max_evaluations=evaluations,
        time_limit_s=time_limit,
        removal_fraction_min=0.10,
        removal_fraction_max=0.40,
        reaction_factor=0.20,
        segment_length=25,
        initial_temperature=1.0,
        cooling_rate=0.995,
    )

    started = time.perf_counter()

    solver = ALNSReference(
        evaluator=evaluate,
        customer_ids=tuple(
            customer_to_int[str(request.request_id)]
            for request in scenario.requests
        ),
        vehicle_ids=tuple(
            vehicle_to_int[str(vehicle.vehicle_id)]
            for vehicle in scenario.fleet
        ),
        initial_route=initial_reference,
        config=config,
        seed=seed,
    )

    result = solver.run()
    wall_time = time.perf_counter() - started

    if not result.feasible or result.route is None:
        return {
            "seed": seed,
            "feasible": False,
            "complete": False,
            "unique_customers": False,
            "best_objective": None,
            "verified_objective": None,
            "distance_m": None,
            "travel_time_s": None,
            "waiting_time_s": None,
            "service_time_s": None,
            "evaluations": int(result.evaluations),
            "iterations": int(result.iterations),
            "elapsed_s": float(result.elapsed_s),
            "wall_time_s": float(wall_time),
            "removal_statistics": dict(result.removal_statistics),
            "repair_statistics": dict(result.repair_statistics),
            "operator_weights": dict(result.operator_weights),
            "route": None,
            "message": result.message,
        }

    verification = validate_final_route(
        result.route,
        evaluator=evaluator,
        scenario=scenario,
        int_to_customer=int_to_customer,
        int_to_vehicle=int_to_vehicle,
    )

    if (
        not verification["feasible"]
        or not verification["complete"]
        or not verification["unique_customers"]
    ):
        raise RuntimeError(
            f"ALNS seed={seed} returned a route that failed independent "
            f"verification: {verification}"
        )

    return {
        "seed": seed,
        "feasible": True,
        "complete": bool(verification["complete"]),
        "unique_customers": bool(verification["unique_customers"]),
        "best_objective": float(result.objective),
        "verified_objective": float(verification["objective"]),
        "distance_m": float(verification["distance_m"]),
        "travel_time_s": float(verification["travel_time_s"]),
        "waiting_time_s": float(verification["waiting_time_s"]),
        "service_time_s": float(verification["service_time_s"]),
        "evaluations": int(result.evaluations),
        "iterations": int(result.iterations),
        "elapsed_s": float(result.elapsed_s),
        "wall_time_s": float(wall_time),
        "removal_statistics": dict(result.removal_statistics),
        "repair_statistics": dict(result.repair_statistics),
        "operator_weights": dict(result.operator_weights),
        "route": verification["route"],
        "message": result.message,
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def write_artifacts(
    *,
    scenario: Scenario,
    args: argparse.Namespace,
    runs: list[dict[str, Any]],
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    feasible_runs = [
        run for run in runs
        if run["feasible"] and run["best_objective"] is not None
    ]
    objectives = [float(run["best_objective"]) for run in feasible_runs]
    runtimes = [float(run["wall_time_s"]) for run in runs]
    evaluations = [float(run["evaluations"]) for run in runs]
    iterations = [float(run["iterations"]) for run in runs]

    summary = {
        "experiment": "step9_alns_20_customer",
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "seed": SCENARIO_SEED,
            "customers": len(scenario.requests),
            "vehicles": len(scenario.fleet),
            "nodes": len(scenario.nodes),
            "edges": len(scenario.edges),
        },
        "protocol": {
            "evaluations": args.evaluations,
            "time_limit_s": args.time_limit,
            "seeds": list(args.seeds),
            "destroy_operators": [
                "random_removal",
                "worst_removal",
                "related_removal",
            ],
            "repair_operators": [
                "greedy_insertion",
                "regret_insertion",
            ],
            "objective_reference": (
                "No certified 20-customer optimum is available. "
                "Best-feasible objectives are reported; no optimality "
                "gap is claimed."
            ),
        },
        "results": {
            "runs": len(runs),
            "feasible_rate": (
                sum(bool(run["feasible"]) for run in runs) / len(runs)
                if runs else 0.0
            ),
            "complete_rate": (
                sum(bool(run["complete"]) for run in runs) / len(runs)
                if runs else 0.0
            ),
            "best_objective_mean": mean(objectives),
            "best_objective_std": stdev(objectives),
            "best_objective_min": min(objectives) if objectives else None,
            "best_objective_max": max(objectives) if objectives else None,
            "runtime_mean_s": mean(runtimes),
            "runtime_std_s": stdev(runtimes),
            "evaluations_mean": mean(evaluations),
            "iterations_mean": mean(iterations),
        },
    }

    protocol = {
        "experiment": "step9_alns_20_customer",
        "scenario_generation": {
            "seed": SCENARIO_SEED,
            "customer_count": CUSTOMERS,
            "vehicle_count": VEHICLES,
            "road_junction_count": ROAD_JUNCTIONS,
            "config": {
                key: value
                for key, value in vars(make_config()).items()
                if not key.startswith("_")
            },
        },
        "solver": {
            "evaluations": args.evaluations,
            "time_limit_s": args.time_limit,
            "seeds": list(args.seeds),
            "initialization": (
                "same project InitialSolutionBuilder heuristic "
                "'earliest_deadline', requiring complete feasibility"
            ),
            "evaluation": (
                "actual RouteEvaluator; ALNS integer IDs are only an "
                "adapter representation"
            ),
            "certification": False,
            "optimality_gap": None,
        },
    }

    (OUTPUT / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True)
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (OUTPUT / "runs.json").write_text(
        json.dumps(runs, indent=2, sort_keys=True)
    )

    with (OUTPUT / "runs.csv").open("w", newline="") as handle:
        fieldnames = [
            "seed",
            "feasible",
            "complete",
            "unique_customers",
            "best_objective",
            "verified_objective",
            "distance_m",
            "travel_time_s",
            "waiting_time_s",
            "service_time_s",
            "evaluations",
            "iterations",
            "elapsed_s",
            "wall_time_s",
            "removal_statistics",
            "repair_statistics",
            "operator_weights",
            "route",
            "message",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for run in runs:
            row = dict(run)
            row["removal_statistics"] = json.dumps(
                row["removal_statistics"], sort_keys=True
            )
            row["repair_statistics"] = json.dumps(
                row["repair_statistics"], sort_keys=True
            )
            row["operator_weights"] = json.dumps(
                row["operator_weights"], sort_keys=True
            )
            row["route"] = json.dumps(
                row["route"], sort_keys=True
            )
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluations",
        type=int,
        default=DEFAULT_EVALUATIONS,
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="100 evaluations, 2 seconds, seed 1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.evaluations = 100
        args.time_limit = 2.0
        args.seeds = [1]

    if args.evaluations <= 0:
        raise ValueError("--evaluations must be positive")
    if args.time_limit <= 0.0:
        raise ValueError("--time-limit must be positive")
    if not args.seeds:
        raise ValueError("--seeds requires at least one seed")

    scenario = build_scenario()

    print("Step 9 — 20-customer ALNS reference")
    print("=" * 58)
    print(
        f"Scenario:            {scenario.scenario_id} | "
        f"{len(scenario.requests)} customers | "
        f"{len(scenario.fleet)} vehicles | "
        f"{len(scenario.nodes)} nodes | "
        f"{len(scenario.edges)} edges"
    )
    print(f"Evaluation budget:   {args.evaluations}")
    print(f"Time limit / run:     {args.time_limit:.3f} s")
    print(f"Seeds:                {args.seeds}")
    print("Exact optimum:        NOT AVAILABLE")
    print("Reference status:     best-known heuristic, not certified")
    print()

    # Validate the constructive seed once before spending ALNS budget.
    seed_plan, customer_to_int, vehicle_to_int = build_feasible_initial_route(
        scenario
    )
    seed_evaluator = RouteEvaluator(scenario)
    seed_eval = seed_evaluator.evaluate(seed_plan)

    print("Constructive initialization")
    print("-" * 58)
    print(f"Feasible:             {seed_eval.feasible}")
    print(f"Complete customers:   {len(seed_eval.served_customer_ids)}/20")
    print(f"Objective:            {seed_eval.objective_value:.6f}")
    print()

    runs: list[dict[str, Any]] = []

    for seed in args.seeds:
        print(f"Running ALNS seed={seed}...")
        run = run_one(
            scenario,
            seed=seed,
            evaluations=args.evaluations,
            time_limit=args.time_limit,
        )
        runs.append(run)

        if run["feasible"]:
            print(
                f"  best={run['best_objective']:.6f} | "
                f"verified={run['verified_objective']:.6f} | "
                f"feasible={run['feasible']} | "
                f"evals={run['evaluations']} | "
                f"iterations={run['iterations']} | "
                f"time={run['wall_time_s']:.3f}s"
            )
        else:
            print(
                f"  best=None | feasible=False | "
                f"evals={run['evaluations']} | "
                f"iterations={run['iterations']} | "
                f"time={run['wall_time_s']:.3f}s"
            )

    write_artifacts(
        scenario=scenario,
        args=args,
        runs=runs,
    )

    feasible = [
        run for run in runs
        if run["feasible"] and run["best_objective"] is not None
    ]
    objectives = [float(run["best_objective"]) for run in feasible]

    print()
    print("Summary")
    print("-" * 58)
    print(
        f"Feasible rate:        "
        f"{sum(bool(run['feasible']) for run in runs) / len(runs):.3f}"
    )
    print(
        f"Complete rate:        "
        f"{sum(bool(run['complete']) for run in runs) / len(runs):.3f}"
    )
    print(
        f"Best objective mean:  "
        f"{mean(objectives):.6f}"
    )
    print(
        f"Best objective std:   "
        f"{stdev(objectives):.6f}"
    )
    print(
        f"Best objective min:   "
        f"{min(objectives):.6f}" if objectives
        else "Best objective min:   None"
    )
    print(
        f"Runtime mean:         "
        f"{mean([float(run['wall_time_s']) for run in runs]):.6f}s"
    )
    print(
        f"Evaluations mean:     "
        f"{mean([float(run['evaluations']) for run in runs]):.1f}"
    )
    print()
    print("Optimality gap:       NOT REPORTED (no certified optimum)")
    print()
    print(
        "Artifacts:            "
        f"{OUTPUT}"
    )


if __name__ == "__main__":
    main()
