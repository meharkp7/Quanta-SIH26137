#!/usr/bin/env python3
"""
Step 8 R3 — multi-instance robustness experiment.

Uses the repository's ACTUAL Step-4 / Step-7 / Step-8 APIs and reuses the
scientifically validated R2 population-construction procedure.

Protocol:
    * independently generated scenarios;
    * one diversified feasible initial population per optimizer seed;
    * exact same population for QPSO and matched PSO;
    * same evaluator, repair and evaluation budget;
    * no QPSO-specific tuning toward a desired outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from src.contracts.scenario import Scenario
from src.data.dataset_generator import DynamicDatasetConfig, generate_dynamic_scenario
from src.optim.experiments import run_condition, write_results
from src.optim.qpso import AdaptiveQPSOConfig
from src.optim.common import FitnessOracle
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_encoding import Step7RouteEngine

# Reuse the exact R2 population protocol rather than inventing a weaker
# random initialization for R3.
from run_step8_r2_experiment import build_population


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "step8_research" / "r3_multi_instance"

DEFAULT_SCENARIO_SEEDS = (26157, 26158, 26159, 26160, 26161)
DEFAULT_CUSTOMER_COUNTS = (10, 20, 30)
DEFAULT_OPTIMIZER_SEEDS = (1, 2, 3)


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def make_config(customers: int, scenario_seed: int) -> DynamicDatasetConfig:
    """
    Keep the R2 generator family/protocol fixed while varying the scenario
    seed and customer scale.
    """
    return DynamicDatasetConfig(
        customer_count=customers,
        road_junction_count=196,
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
        fleet_size=None,
        customer_spread=0.85,
        cluster_strength=0.20,
        minimum_road_segment_m=80.0,
        demand_min=1,
        demand_max=8,
        seed=scenario_seed,
        dataset_split="test",
    )


def fresh_oracle_factory(scenario: Scenario):
    """
    Fresh evaluator/engine/oracle per optimizer condition.

    This prevents state/accounting leakage between QPSO and PSO.
    """
    def factory(seed: int) -> FitnessOracle:
        del seed
        evaluator = RouteEvaluator(scenario)
        engine = Step7RouteEngine(scenario, evaluator)
        from src.optim.qpso import RouteFitnessOracle

        return RouteFitnessOracle(
            engine,
            repair=True,
            repair_penalty=0.0,
        )

    return factory


def base_config(
    scenario: Scenario,
    population_size: int,
    evaluations: int,
) -> AdaptiveQPSOConfig:
    return AdaptiveQPSOConfig(
        dimensions=2 * len(scenario.requests),
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=population_size,
        max_evaluations=evaluations,
        seed=1,
        tolerance=0.0,
        stagnation_patience=50,
    )


def run(args: argparse.Namespace) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_runs: list[Any] = []
    scenario_manifest: list[dict[str, Any]] = []
    population_manifest: list[dict[str, Any]] = []

    started_all = time.perf_counter()

    for customers in args.customer_counts:
        for scenario_seed in args.scenario_seeds:
            print()
            print(
                f"R3 scenario | customers={customers} | "
                f"scenario_seed={scenario_seed}"
            )

            scenario = generate_dynamic_scenario(
                make_config(customers, scenario_seed)
            )
            scenario = Scenario.model_validate(
                scenario.model_dump(mode="json")
            )

            print(
                f"  {scenario.scenario_id} | "
                f"{len(scenario.requests)} customers | "
                f"{len(scenario.fleet)} vehicles | "
                f"{len(scenario.nodes)} nodes | "
                f"{len(scenario.edges)} edges"
            )

            scenario_manifest.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_seed": scenario_seed,
                    "customers": customers,
                    "vehicles": len(scenario.fleet),
                    "nodes": len(scenario.nodes),
                    "edges": len(scenario.edges),
                }
            )

            for optimizer_seed in args.optimizer_seeds:
                print(f"  building matched population seed={optimizer_seed} ...")

                population, diagnostics = build_population(
                    scenario,
                    args.population,
                    optimizer_seed,
                )

                population_manifest.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "scenario_seed": scenario_seed,
                        "customers": customers,
                        "optimizer_seed": optimizer_seed,
                        "diagnostics": diagnostics,
                    }
                )

                print(
                    f"    {diagnostics['unique_feasible_population']}/"
                    f"{args.population} unique feasible | "
                    f"route_diversity="
                    f"{diagnostics['route_diversity_mean_pairwise_exact']:.4f}"
                )

                config = base_config(
                    scenario,
                    args.population,
                    args.evaluations,
                )
                oracle_factory = fresh_oracle_factory(scenario)

                # CRITICAL: the exact same population object is supplied to
                # both algorithms for this paired replication.
                qpso_run = run_condition(
                    condition="canonical",
                    algorithm="qpso",
                    seed=optimizer_seed,
                    base_qpso_config=config,
                    oracle_factory=oracle_factory,
                    initial_population=population,
                )

                pso_run = run_condition(
                    condition="matched_pso",
                    algorithm="pso",
                    seed=optimizer_seed,
                    base_qpso_config=config,
                    oracle_factory=oracle_factory,
                    initial_population=population,
                )

                # ExperimentRun has no scenario fields in the repository
                # contract, so retain the run objects for canonical
                # write_results and maintain scenario linkage separately.
                all_runs.extend((qpso_run, pso_run))

                print(
                    f"    QPSO={qpso_run.best_fitness:.3f} | "
                    f"PSO={pso_run.best_fitness:.3f} | "
                    f"delta={qpso_run.best_fitness - pso_run.best_fitness:+.3f}"
                )

    # Canonical Step-8 artifacts for the complete R3 run.
    canonical_dir = OUTPUT / "canonical_runs"
    summaries = ()
    if all_runs:
        from src.optim.experiments import summarize_runs
        summaries = summarize_runs(all_runs)
        write_results(canonical_dir, all_runs, summaries)

    # Machine-readable scenario/pair linkage.
    (OUTPUT / "scenarios.json").write_text(
        json.dumps(scenario_manifest, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "initialization_diagnostics.json").write_text(
        json.dumps(population_manifest, indent=2),
        encoding="utf-8",
    )

    # Flatten paired results from canonical run objects.
    paired_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[int, int, int], dict[str, Any]] = {}

    # We iterate in canonical execution order:
    # scenario, optimizer seed, QPSO, PSO.
    for index in range(0, len(all_runs), 2):
        qpso = all_runs[index]
        pso = all_runs[index + 1]

        scenario_seed = scenario_manifest[len(paired_rows) // len(args.optimizer_seeds)][
            "scenario_seed"
        ] if False else None

        # Recover scenario/customer linkage from execution ordering using a
        # simple cursor below instead of relying on ExperimentRun fields.
        paired_rows.append(
            {
                "pair_index": len(paired_rows),
                "optimizer_seed": int(qpso.seed),
                "qpso_best_fitness": float(qpso.best_fitness),
                "pso_best_fitness": float(pso.best_fitness),
                "qpso_minus_pso": float(
                    qpso.best_fitness - pso.best_fitness
                ),
                "qpso_feasible": bool(qpso.feasible),
                "pso_feasible": bool(pso.feasible),
                "qpso_coordinate_diversity_final": float(
                    qpso.coordinate_diversity_final
                ),
                "pso_coordinate_diversity_final": float(
                    pso.coordinate_diversity_final
                ),
                "qpso_route_diversity_final": float(
                    qpso.route_diversity_final
                ),
                "pso_route_diversity_final": float(
                    pso.route_diversity_final
                ),
            }
        )

    # Add exact scenario linkage in the same nested execution order.
    cursor = 0
    for customers in args.customer_counts:
        for scenario_seed in args.scenario_seeds:
            for optimizer_seed in args.optimizer_seeds:
                row = paired_rows[cursor]
                row["customers"] = customers
                row["scenario_seed"] = scenario_seed
                row["optimizer_seed"] = optimizer_seed
                cursor += 1

    paired_csv = OUTPUT / "paired_runs.csv"
    if paired_rows:
        with paired_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(paired_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(paired_rows)

    # Aggregate by customer scale.
    scale_rows: list[dict[str, Any]] = []
    for customers in args.customer_counts:
        rows = [r for r in paired_rows if r["customers"] == customers]
        deltas = [float(r["qpso_minus_pso"]) for r in rows]

        if not deltas:
            continue

        scale_rows.append(
            {
                "customers": customers,
                "paired_n": len(deltas),
                "mean_qpso_minus_pso": statistics.mean(deltas),
                "median_qpso_minus_pso": statistics.median(deltas),
                "std_qpso_minus_pso": (
                    statistics.stdev(deltas)
                    if len(deltas) > 1 else 0.0
                ),
                "qpso_wins": sum(d < 0 for d in deltas),
                "pso_wins": sum(d > 0 for d in deltas),
                "ties": sum(d == 0 for d in deltas),
                "qpso_feasible_rate": statistics.mean(
                    bool(r["qpso_feasible"]) for r in rows
                ),
                "pso_feasible_rate": statistics.mean(
                    bool(r["pso_feasible"]) for r in rows
                ),
            }
        )

    (OUTPUT / "scale_summary.json").write_text(
        json.dumps(scale_rows, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "protocol": "step8_r3_multi_instance_robustness",
        "customer_counts": list(args.customer_counts),
        "scenario_seeds": list(args.scenario_seeds),
        "optimizer_seeds": list(args.optimizer_seeds),
        "population": args.population,
        "evaluations": args.evaluations,
        "initialization_protocol": "R2_diversified_constructive_plus_route_mutation",
        "matched_population": True,
        "same_evaluator_and_repair": True,
        "qpsp_superiority_not_tuned": True,
        "total_scenarios": len(scenario_manifest),
        "total_paired_runs": len(paired_rows),
        "elapsed_seconds": time.perf_counter() - started_all,
    }
    (OUTPUT / "protocol.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("R3 COMPLETE")
    print("=" * 78)
    print(f"Scenarios: {len(scenario_manifest)}")
    print(f"Paired optimizer runs: {len(paired_rows)}")
    print(f"Artifacts: {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customer-counts", type=parse_ints,
                        default=DEFAULT_CUSTOMER_COUNTS)
    parser.add_argument("--scenario-seeds", type=parse_ints,
                        default=DEFAULT_SCENARIO_SEEDS)
    parser.add_argument("--optimizer-seeds", type=parse_ints,
                        default=DEFAULT_OPTIMIZER_SEEDS)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--evaluations", type=int, default=100)
    args = parser.parse_args()

    if args.population < 2:
        parser.error("--population must be >= 2")
    if args.evaluations < args.population:
        parser.error("--evaluations must be >= --population")

    run(args)


if __name__ == "__main__":
    main()
