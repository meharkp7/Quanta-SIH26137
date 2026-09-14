"""Benchmark the POI path-cache precomputation enhancement.

This is an enhancement benchmark, not an official implementation-plan step.
It reports preprocessing time separately from steady-state candidate
execution so the optimization cannot hide eager warm-up cost.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from src.contracts.scenario import Scenario
from src.data.dataset_generator import DynamicDatasetConfig, generate_dynamic_scenario
from src.routing.cost_view import CostView
from src.routing.route_encoding import Step7RouteEngine
from src.routing.route_evaluator import RouteEvaluationConfig, RouteEvaluator
from src.optim.qpso import RouteFitnessOracle


def build_scenario(customer_count: int, seed: int) -> Scenario:
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
    scenario = generate_dynamic_scenario(config)
    return Scenario.model_validate(scenario.model_dump(mode="json"))


def candidate(dimensions: int) -> list[float]:
    return [((i * 37) % 101) / 100.0 for i in range(dimensions)]


def measure(scenario: Scenario, *, eager: bool, repeats: int) -> dict[str, object]:
    cost_view = CostView(scenario.edges, graph_version=scenario.graph_version)
    config = RouteEvaluationConfig(precompute_poi_matrix=eager)

    started = time.perf_counter()
    evaluator = RouteEvaluator(scenario, config=config, cost_view=cost_view)
    construction_s = time.perf_counter() - started

    engine = Step7RouteEngine(scenario, evaluator)
    oracle = RouteFitnessOracle(
        engine,
        commitments=None,
        planning_time_s=0.0,
        repair=True,
        repair_penalty=0.0,
    )
    x = candidate(2 * len(scenario.requests))

    started = time.perf_counter()
    oracle(x)
    first_evaluation_s = time.perf_counter() - started

    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        oracle(x)
        samples.append(time.perf_counter() - started)

    stats = evaluator.path_builder.path_cache.stats()
    precompute = evaluator.poi_precompute_stats
    return {
        "customers": len(scenario.requests),
        "vehicles": len(scenario.fleet),
        "road_nodes": len(scenario.nodes),
        "road_edges": len(scenario.edges),
        "eager_precompute": eager,
        "construction_seconds": construction_s,
        "first_evaluation_seconds": first_evaluation_s,
        "steady_state_mean_seconds": statistics.mean(samples),
        "steady_state_median_seconds": statistics.median(samples),
        "steady_state_min_seconds": min(samples),
        "steady_state_max_seconds": max(samples),
        "cache_size": int(stats["size"]),
        "cache_hits": int(stats["hits"]),
        "cache_misses": int(stats["misses"]),
        "precompute_searches": (
            precompute.path_stats.searches if precompute is not None else 0
        ),
        "precompute_paths_cached": (
            precompute.path_stats.paths_cached if precompute is not None else 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=26137)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/enhancements/poi_precompute.csv"),
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if any(size < 1 for size in args.sizes):
        raise ValueError("customer sizes must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for index, size in enumerate(args.sizes):
        scenario = build_scenario(size, args.seed + index)
        for eager in (False, True):
            row = measure(scenario, eager=eager, repeats=args.repeats)
            rows.append(row)
            print(
                f"customers={size:>3} eager={eager!s:<5} "
                f"construction={row['construction_seconds']:.4f}s "
                f"first={row['first_evaluation_seconds']:.4f}s "
                f"steady={row['steady_state_mean_seconds']:.4f}s "
                f"cache={row['cache_size']}"
            )

    fieldnames = list(rows[0]) if rows else []
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
