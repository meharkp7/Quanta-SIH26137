from __future__ import annotations

"""
Step 8 R2 controlled-instance / initial-population validator.

Purpose:
    1. Generate a reproducible 20-customer Step-4 scenario.
    2. Build a constructive feasible route using the repository's actual
       InitialSolutionBuilder.
    3. Encode that route with the real Step-7 continuous representation.
    4. Create deterministic perturbed alternatives.
    5. Evaluate every candidate through the real Step-7
       RouteEvaluator + bounded repair path.
    6. Report feasibility, completeness, route diversity and repair pressure.

This is a validation/protocol tool, not an optimizer.

Run from the repository root:

    python validate_step8_r2_initial_population.py

Optional:
    python validate_step8_r2_initial_population.py --customers 20 --population 20
    python validate_step8_r2_initial_population.py --seed 26157
"""

import argparse
import json
import math
import random
from pathlib import Path

from src.contracts.scenario import Scenario
from src.data.dataset_generator import (
    DynamicDatasetConfig,
    generate_dynamic_scenario,
)
from src.routing.initial_solution import (
    InitialSolutionBuilder,
    InitialSolutionConfig,
)
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_encoding import ContinuousRouteEncoder, Step7RouteEngine


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "step8_research"
    / "r2_initial_population"
)


def build_scenario(customers: int, seed: int) -> Scenario:
    """
    Generate a Step-4 scenario with the repository's own generator.

    The generator itself performs its declared quality checks, including the
    reference-schedule feasibility check.  We additionally validate the
    resulting Scenario through the public contract.
    """
    config = DynamicDatasetConfig(
        customer_count=customers,
        road_junction_count=196,
        extent_m=8_000.0,
        topology_family="grid",
        grid_rows=14,
        grid_cols=14,
        jitter_fraction=0.0,
        one_way_fraction=0.08,
        max_customer_snap_m=500.0,
        window_profile="loose",
        window_slack_s=1_800.0,
        service_duration_s=60.0,
        vehicle_capacity=30.0,
        fleet_size=None,
        customer_spread=0.86,
        cluster_strength=0.62,
        minimum_road_segment_m=30.0,
        demand_min=1,
        demand_max=8,
        seed=seed,
        dataset_split="test",
    )

    scenario = generate_dynamic_scenario(config)

    return Scenario.model_validate(
        scenario.model_dump(mode="json")
    )


def route_signature(route_plan) -> tuple:
    """Return a stable route-level phenotype signature."""
    return tuple(
        (
            str(vehicle_route.vehicle_id),
            tuple(str(cid) for cid in vehicle_route.customer_ids),
        )
        for vehicle_route in route_plan.vehicle_routes
    )


def perturb_keys(
    base_keys: tuple[float, ...],
    *,
    seed: int,
    magnitude: float,
) -> tuple[float, ...]:
    """Create one deterministic bounded perturbation of a base key vector."""
    rng = random.Random(seed)
    values = []

    for value in base_keys:
        delta = rng.uniform(-magnitude, magnitude)
        values.append(min(1.0, max(0.0, float(value) + delta)))

    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=20)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--seed", type=int, default=26157)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    if args.customers < 2:
        raise ValueError("--customers must be >= 2")
    if args.population < 2:
        raise ValueError("--population must be >= 2")

    print("\nStep 8 R2 initial-population validation")
    print("=" * 78)

    scenario = build_scenario(args.customers, args.seed)

    evaluator = RouteEvaluator(scenario)
    encoder = ContinuousRouteEncoder(scenario)
    engine = Step7RouteEngine(scenario, evaluator)

    # The constructive solver is the declared Step-4/Step-8 starting point.
    # It uses the same independent evaluator that downstream optimization
    # uses for feasibility truth.
    builder = InitialSolutionBuilder(
        scenario,
        evaluator,
        config=InitialSolutionConfig(
            heuristic="earliest_deadline",
            allow_unassigned=False,
            prioritize_time_windows=True,
            prioritize_distance=True,
            prioritize_travel_time=True,
            prioritize_waiting_time=True,
            require_capacity_feasibility=True,
            require_connectivity=True,
            require_time_window_feasibility=True,
            planning_time_s=0.0,
            include_future_released_requests=True,
        ),
    )

    print(
        f"Scenario: {scenario.scenario_id} | "
        f"{len(scenario.requests)} customers | "
        f"{len(scenario.fleet)} vehicles | "
        f"{len(scenario.nodes)} nodes | "
        f"{len(scenario.edges)} edges"
    )

    initial = builder.build()

    print()
    print("Constructive seed")
    print("-" * 78)
    print(f"method      : {initial.construction_method}")
    print(f"feasible    : {initial.feasible}")
    print(f"complete    : {initial.complete}")
    print(f"assigned    : {len(initial.assigned_customer_ids)}")
    print(f"unassigned  : {len(initial.unassigned_customer_ids)}")
    print(
        "fitness     : "
        "deferred to the Step-8 FitnessResult after "
        "continuous encode/decode"
    )

    if not initial.feasible or not initial.complete:
        raise RuntimeError(
            "Constructive initial solution is not a complete feasible route; "
            "do not proceed to R2."
        )

    base_encoding = encoder.encode(initial.route_plan)
    base_keys = tuple(float(x) for x in base_encoding.keys)

    # Evaluate the constructive route through the exact Step-7 lifecycle too.
    base_candidate = engine.evaluate_keys(
        base_keys,
        repair=True,
    )

    if not base_candidate.feasible:
        raise RuntimeError(
            "Constructive route became infeasible after Step-7 encode/decode."
        )

    candidates = [
        {
            "index": 0,
            "kind": "constructive_seed",
            "keys": base_keys,
            "candidate": base_candidate,
        }
    ]

    # Produce more perturbation than strictly needed so that the final
    # population can contain multiple distinct phenotypes if repair collapses
    # some perturbations back to the same route.
    attempt = 0
    while len(candidates) < args.population and attempt < args.population * 50:
        attempt += 1

        # Alternate perturbation magnitudes to explore both nearby and more
        # structurally different continuous keys.
        magnitude = (
            0.025 if attempt % 3 == 1
            else 0.075 if attempt % 3 == 2
            else 0.15
        )

        keys = perturb_keys(
            base_keys,
            seed=args.seed + 10_000 + attempt,
            magnitude=magnitude,
        )

        candidate = engine.evaluate_keys(
            keys,
            repair=True,
        )

        if not candidate.feasible:
            continue

        candidates.append(
            {
                "index": len(candidates),
                "kind": "perturbed_feasible",
                "keys": keys,
                "candidate": candidate,
            }
        )

    feasible_count = sum(
        1 for item in candidates if item["candidate"].feasible
    )

    signatures = [
        route_signature(item["candidate"].repaired_plan)
        for item in candidates
    ]
    unique_signatures = set(signatures)

    route_diversity = (
        len(unique_signatures) / len(signatures)
        if signatures
        else 0.0
    )

    print()
    print("Initial population")
    print("-" * 78)
    print(f"requested population : {args.population}")
    print(f"constructed          : {len(candidates)}")
    print(f"feasible             : {feasible_count}/{len(candidates)}")
    print(f"feasible fraction    : {feasible_count / len(candidates):.3f}")
    print(f"unique route plans   : {len(unique_signatures)}")
    print(f"route diversity      : {route_diversity:.3f}")
    print("repair distance      : not used for this initialization gate")

    if len(candidates) < args.population:
        raise RuntimeError(
            "Could not construct the requested number of feasible "
            "perturbed candidates without collapsing/failing. "
            "R2 population should not be launched yet."
        )

    if feasible_count != args.population:
        raise RuntimeError(
            "R2 population contains infeasible candidates."
        )

    if len(unique_signatures) < 2:
        raise RuntimeError(
            "All feasible candidates collapsed to one route phenotype. "
            "This violates the Step 8 requirement for meaningful particle "
            "search diversity."
        )

    rows = []
    for item, signature in zip(candidates, signatures):
        candidate = item["candidate"]
        rows.append(
            {
                "index": item["index"],
                "kind": item["kind"],
                "feasible": candidate.feasible,
                "route_signature": signature,
                "keys": item["keys"],
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "protocol": "step8_r2_initial_population_v1",
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "customers": len(scenario.requests),
            "vehicles": len(scenario.fleet),
            "road_nodes": len(scenario.nodes),
            "road_edges": len(scenario.edges),
            "seed": args.seed,
        },
        "population": {
            "requested": args.population,
            "constructed": len(candidates),
            "feasible_fraction": feasible_count / len(candidates),
            "unique_route_plans": len(unique_signatures),
            "route_diversity": route_diversity,
            "evaluation_metrics": "intentionally omitted from this initialization gate",
        },
        "construction": {
            "method": initial.construction_method,
            "complete": initial.complete,
            "feasible": initial.feasible,
        },
        "candidates": rows,
        "decision": "PASS",
        "notes": [
            "The constructive seed is produced by the repository's "
            "InitialSolutionBuilder.",
            "All candidates are independently evaluated through the "
            "Step-7 engine.",
            "Perturbations are deterministic and bounded to [0, 1].",
            "This artifact validates initialization; it is not optimizer "
            "performance evidence.",
        ],
    }

    manifest_path = args.output / "r2_initial_population.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("PASS — R2 has a complete feasible and non-collapsed initial population.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
