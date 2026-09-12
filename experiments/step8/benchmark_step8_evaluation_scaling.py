from __future__ import annotations

"""
Step 8 evaluation-scaling benchmark.

Generates controlled synthetic Step-4 scenarios using the repository's actual
DynamicDatasetConfig / generate_dynamic_scenario machinery, then measures the
real Step-7 RouteEvaluator + bounded repair pipeline.

The generation phase is measured separately from candidate evaluation so the
reported evaluation cost reflects the operation performed repeatedly by QPSO
/ PSO.

Run from the repository root:

    python benchmark_step8_evaluation_scaling.py

Optional:

    python benchmark_step8_evaluation_scaling.py --sizes 5 10 20 40
    python benchmark_step8_evaluation_scaling.py --repeats 3
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from src.contracts.scenario import Scenario
from src.data.dataset_generator import (
    DynamicDatasetConfig,
    generate_dynamic_scenario,
)
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_encoding import Step7RouteEngine
from src.optim.qpso import RouteFitnessOracle


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "step8_research" / "evaluation_scaling"
)


def build_scenario(customer_count: int, seed: int) -> tuple[Scenario, float]:
    """Generate one controlled Step-4 synthetic scenario and measure generation time."""

    # Keep the road network fixed across scales.  This isolates the effect of
    # increasing customer/problem dimensionality rather than simultaneously
    # increasing road-network complexity.
    config = DynamicDatasetConfig(
        customer_count=customer_count,
        road_junction_count=196,
        extent_m=8_000.0,
        topology_family="grid",
        grid_rows=14,
        grid_cols=14,
        jitter_fraction=0.0,
        one_way_fraction=0.08,
        max_customer_snap_m=500.0,
        window_profile="medium",
        window_slack_s=900.0,
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

    started = time.perf_counter()
    scenario = generate_dynamic_scenario(config)
    generation_seconds = time.perf_counter() - started

    # Re-validate the generated object through the public contract boundary.
    scenario = Scenario.model_validate(
        scenario.model_dump(mode="json")
    )

    return scenario, generation_seconds


def make_candidate(dimensions: int) -> list[float]:
    """Create one deterministic continuous candidate for every scale."""
    return [
        ((i * 37) % 101) / 100.0
        for i in range(dimensions)
    ]


def benchmark_scenario(
    scenario: Scenario,
    generation_seconds: float,
    repeats: int,
    seed: int,
) -> dict:
    """Measure repeated candidate evaluation on one generated scenario."""

    evaluator = RouteEvaluator(scenario)
    engine = Step7RouteEngine(scenario, evaluator)
    oracle = RouteFitnessOracle(
        engine,
        commitments=None,
        planning_time_s=0.0,
        repair=True,
        repair_penalty=0.0,
    )

    dimensions = 2 * len(scenario.requests)
    candidate = make_candidate(dimensions)

    # Warm-up is excluded from measured samples to remove one-time setup
    # effects. The actual evaluator/repair implementation is unchanged.
    oracle(candidate)

    timings: list[float] = []
    repair_distances: list[float] = []
    feasible: list[bool] = []
    fitness_values: list[float] = []

    for _ in range(repeats):
        started = time.perf_counter()
        result = oracle(candidate)
        elapsed = time.perf_counter() - started

        timings.append(elapsed)
        repair_distances.append(float(result.repair_distance))
        feasible.append(bool(result.feasible))
        fitness_values.append(float(result.fitness))

    mean_seconds = statistics.mean(timings)

    return {
        "scenario_id": scenario.scenario_id,
        "seed": seed,
        "customers": len(scenario.requests),
        "vehicles": len(scenario.fleet),
        "road_nodes": len(scenario.nodes),
        "road_edges": len(scenario.edges),
        "dimensions": dimensions,
        "repeats": repeats,
        "generation_seconds": generation_seconds,
        "evaluation_seconds_mean": mean_seconds,
        "evaluation_seconds_median": statistics.median(timings),
        "evaluation_seconds_min": min(timings),
        "evaluation_seconds_max": max(timings),
        "evaluation_seconds_std": (
            statistics.stdev(timings) if len(timings) > 1 else 0.0
        ),
        "repair_distance_mean": statistics.mean(repair_distances),
        "feasible_rate": statistics.mean(feasible),
        "fitness_mean": statistics.mean(fitness_values),
        "estimated_20_particles_x_100_evaluations_hours": (
            mean_seconds * 20 * 100 / 3600.0
        ),
        "estimated_20_particles_x_500_evaluations_hours": (
            mean_seconds * 20 * 500 / 3600.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Step 8 route-evaluation scaling on controlled Step-4 scenarios."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[5, 10, 20, 40],
        help="Customer counts to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured repeated evaluations per scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=26137,
        help="Base deterministic Step-4 generation seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for benchmark artifacts.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if not args.sizes:
        raise ValueError("at least one customer-count size is required")
    if any(size < 2 for size in args.sizes):
        raise ValueError("customer counts must be >= 2")

    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    print("\nStep 8 evaluation-scaling benchmark")
    print("=" * 78)
    print(
        "Controlled Step-4 generator: fixed 196-node road network, "
        "increasing customer count"
    )
    print()

    for size in args.sizes:
        # Use a deterministic but size-specific stream so each scale is
        # reproducible while not being an accidental duplicate scenario.
        scenario_seed = args.seed + size

        print(f"[GEN ] {size} customers (seed={scenario_seed}) ...")
        scenario, generation_seconds = build_scenario(
            customer_count=size,
            seed=scenario_seed,
        )

        print(
            f"       generated in {generation_seconds:.3f}s | "
            f"vehicles={len(scenario.fleet)} | "
            f"nodes={len(scenario.nodes)} | "
            f"edges={len(scenario.edges)}"
        )

        print(f"[EVAL] {size} customers × {args.repeats} repeats ...")
        row = benchmark_scenario(
            scenario=scenario,
            generation_seconds=generation_seconds,
            repeats=args.repeats,
            seed=scenario_seed,
        )
        rows.append(row)

        print(
            f"       mean={row['evaluation_seconds_mean']:.4f}s | "
            f"median={row['evaluation_seconds_median']:.4f}s | "
            f"feasible={row['feasible_rate']:.2f} | "
            f"repair={row['repair_distance_mean']:.4f}"
        )
        print(
            "       projected 20 particles × 100 evaluations: "
            f"{row['estimated_20_particles_x_100_evaluations_hours']:.3f} h"
        )
        print(
            "       projected 20 particles × 500 evaluations: "
            f"{row['estimated_20_particles_x_500_evaluations_hours']:.3f} h"
        )
        print()

    json_path = args.output / "evaluation_scaling.json"
    csv_path = args.output / "evaluation_scaling.csv"

    json_path.write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("=" * 78)
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")


if __name__ == "__main__":
    main()