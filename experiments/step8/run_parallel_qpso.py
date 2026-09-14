"""Run a real QPSO optimization using the process-pool parallel oracle.

Usage:
    python experiments/step8/run_parallel_qpso.py
    python experiments/step8/run_parallel_qpso.py --workers 8
    python experiments/step8/run_parallel_qpso.py --customers 40 --evaluations 500

IMPORTANT: on macOS and Windows, the default multiprocessing start method
is "spawn", which re-imports this file from scratch in every worker
process. If the optimizer call sits at module level (not inside
`if __name__ == "__main__":`), each spawned worker re-executes it too,
recursively trying to spawn its own pool — that's the
"RuntimeError: An attempt has been made to start a new process before
the current process has finished its bootstrapping phase" you hit.
Everything that triggers process creation must live inside the
`if __name__ == "__main__":` block below, which is exactly what workers
skip when they re-import this file.
"""

from __future__ import annotations

import argparse
import time

from src.contracts.scenario import Scenario
from src.data.dataset_generator import (
    DynamicDatasetConfig,
    generate_dynamic_scenario,
)
from src.optim.qpso import AdaptiveQPSO, AdaptiveQPSOConfig
from src.optim.parallel_oracle import ParallelRouteFitnessOracle


def build_scenario(customer_count: int):
    config = DynamicDatasetConfig(
        customer_count=customer_count,
        road_junction_count=196,
        extent_m=8000.0,
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
        seed=26177,
        dataset_split="test",
    )

    scenario = generate_dynamic_scenario(config)

    return Scenario.model_validate(
        scenario.model_dump(mode="json")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real QPSO optimization with the process-pool "
            "parallel fitness oracle."
        )
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=40,
        help="Number of customers in the generated scenario.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="QPSO swarm size (particles).",
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=500,
        help="Total fitness-evaluation budget across the whole swarm.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes. Defaults to os.cpu_count() "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="QPSO random seed.",
    )
    args = parser.parse_args()

    print(f"Building a {args.customers}-customer scenario...")
    scenario = build_scenario(args.customers)
    dims = 2 * len(scenario.requests)

    oracle = ParallelRouteFitnessOracle(
        scenario,
        commitments=None,
        planning_time_s=0.0,
        repair=True,
        repair_penalty=0.0,
        max_workers=args.workers,
    )

    qpso_config = AdaptiveQPSOConfig(
        dimensions=dims,
        population_size=args.population_size,
        max_evaluations=args.evaluations,
        seed=args.seed,
    )

    qpso = AdaptiveQPSO(qpso_config, oracle)

    import os

    worker_count = args.workers or os.cpu_count()

    print(
        f"Running QPSO: {args.population_size} particles, "
        f"{args.evaluations} evaluation budget, "
        f"{dims} dimensions, {worker_count} worker processes..."
    )

    started = time.perf_counter()
    result = qpso.optimize()
    elapsed = time.perf_counter() - started

    oracle.shutdown()

    print()
    print(f"Wall clock:    {elapsed:.2f}s")
    print(f"Evaluations:   {result.evaluations}")
    print(f"Per-eval avg:  {elapsed / result.evaluations:.4f}s")
    print(f"Best fitness:  {result.best_fitness}")
    print(f"Feasible:      {result.best_result.feasible}")


if __name__ == "__main__":
    # Required on macOS/Windows: multiprocessing's "spawn" start method
    # re-imports this module in each worker process. Code outside this
    # guard would re-run (and re-spawn its own pool) inside every
    # worker too. Only code inside this block ever triggers process
    # creation, and workers never reach it because their re-import sees
    # __name__ == "__mp_main__", not "__main__".
    main()