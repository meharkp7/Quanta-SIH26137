from __future__ import annotations

"""
Step 8 research runner.

This runner deliberately keeps experiment orchestration outside the routing
implementation.  Each (condition, seed) run receives a fresh
RouteEvaluator -> Step7RouteEngine -> RouteFitnessOracle chain so that
mutable evaluator/repair state cannot leak between paired runs.

The paired experiment protocol remains controlled by run_ablation_suite:
the same seed produces the same initial population for QPSO and matched PSO,
while each algorithm receives an independently constructed oracle.

Usage from the repository root:

    python -m src.optim.run_step8_research

Optional:

    python -m src.optim.run_step8_research --seeds 0 1 2 3 4
    python -m src.optim.run_step8_research --evaluations 100 --particles 20
    python -m src.optim.run_step8_research --output-dir artifacts/step8_research/r1_qpso_vs_pso
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.contracts.scenario import Scenario
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_encoding import Step7RouteEngine
from src.optim.experiments import run_ablation_suite, write_results
from src.optim.qpso import AdaptiveQPSOConfig, RouteFitnessOracle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = (
    PROJECT_ROOT
    / "fixtures"
    / "step3"
    / "base"
    / "scenario.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "step8_research"
    / "r1_qpso_vs_pso"
)


def load_scenario(path: Path) -> Scenario:
    """Load and validate one V1.2 Scenario contract."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Scenario file does not exist: {path}"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    return Scenario.model_validate(data)


def build_oracle_factory(
    scenario_path: Path,
):
    """
    Return a factory that creates an isolated route stack for every run.

    The seed belongs to the optimizer/search process.  The routing evaluator
    itself is deterministic for a fixed scenario snapshot, so the seed is
    intentionally not used to mutate routing semantics.
    """

    def oracle_factory(seed: int) -> RouteFitnessOracle:
        # Load a fresh immutable Scenario object for every run as well.  This
        # prevents future changes to Scenario-backed mutable helpers from
        # accidentally becoming shared experiment state.
        scenario = load_scenario(scenario_path)

        evaluator = RouteEvaluator(scenario)

        engine = Step7RouteEngine(
            scenario,
            evaluator,
        )

        return RouteFitnessOracle(
            engine,
            commitments=None,
            planning_time_s=0.0,
            repair=True,
            repair_penalty=0.0,
        )

    return oracle_factory


def write_protocol(
    output_dir: Path,
    *,
    scenario_path: Path,
    scenario: Scenario,
    particles: int,
    evaluations: int,
    seeds: Sequence[int],
    conditions: Sequence[str],
) -> Path:
    """Persist the declared R1 protocol alongside the measured results."""

    protocol = {
        "experiment": "R1_qpso_vs_matched_pso",
        "purpose": (
            "Controlled Step 8 comparison of canonical QPSO against "
            "classical matched PSO."
        ),
        "scenario": {
            "path": str(scenario_path),
            "scenario_id": scenario.scenario_id,
            "customer_count": len(scenario.requests),
            "vehicle_count": len(scenario.fleet),
            "road_node_count": len(scenario.nodes),
            "road_edge_count": len(scenario.edges),
        },
        "controls": {
            "encoding": "Step7RouteEngine / continuous 2n key encoding",
            "decoder": "shared Step 7 decoder",
            "repair": "shared bounded Step 7 repair",
            "objective": "shared RouteEvaluator objective",
            "initial_population": (
                "identical per seed across QPSO and matched PSO"
            ),
            "seed": "same optimizer seed per paired comparison",
            "population_size": particles,
            "evaluation_budget": evaluations,
            "q_pso_conditions": list(conditions),
            "include_matched_pso": True,
        },
        "fresh_state_policy": {
            "scenario": "new validated Scenario instance per run",
            "route_evaluator": "new instance per run",
            "step7_route_engine": "new instance per run",
            "route_fitness_oracle": "new instance per run",
        },
        "seeds": [int(seed) for seed in seeds],
        "interpretation": {
            "tiny_fixture": (
                "hand-checkable regression fixture; not treated as "
                "performance evidence"
            ),
            "superiority_claim": (
                "not predetermined; report QPSO gains, ties, or failures "
                "from measured results"
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "r1_protocol.json"
    path.write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def run_experiment(
    *,
    scenario_path: Path = DEFAULT_SCENARIO,
    output_dir: Path = DEFAULT_OUTPUT,
    particles: int = 20,
    evaluations: int = 100,
    seeds: Sequence[int] = tuple(range(10)),
    conditions: Sequence[str] = ("canonical",),
) -> None:
    """Run the controlled R1 experiment and persist its results."""

    if particles < 2:
        raise ValueError("particles must be at least 2")
    if evaluations < particles:
        raise ValueError(
            "evaluations must be at least the initial population size"
        )
    if not seeds:
        raise ValueError("at least one seed is required")
    if not conditions:
        raise ValueError("at least one QPSO condition is required")

    scenario = load_scenario(scenario_path)

    oracle_factory = build_oracle_factory(scenario_path)

    base_config = AdaptiveQPSOConfig(
        dimensions=2 * len(scenario.requests),
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=particles,
        max_evaluations=evaluations,
        seed=0,
    )

    runs, summaries = run_ablation_suite(
        base_qpso_config=base_config,
        oracle_factory=oracle_factory,
        seeds=tuple(int(seed) for seed in seeds),
        conditions=tuple(conditions),
        include_pso=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    write_results(
        output_dir,
        runs,
        summaries,
    )

    write_protocol(
        output_dir,
        scenario_path=scenario_path,
        scenario=scenario,
        particles=particles,
        evaluations=evaluations,
        seeds=seeds,
        conditions=conditions,
    )

    print(f"Scenario: {scenario.scenario_id}")
    print(
        f"Size: {len(scenario.requests)} customers / "
        f"{len(scenario.fleet)} vehicles / "
        f"{len(scenario.nodes)} nodes / "
        f"{len(scenario.edges)} edges"
    )
    print(
        f"Protocol: {particles} particles, "
        f"{evaluations} evaluations, "
        f"{len(seeds)} seeds"
    )
    print()
    print("Results:")
    for summary in summaries:
        print(
            f"  {summary.algorithm}/{summary.condition}: "
            f"runs={summary.runs}, "
            f"mean_best={summary.best_fitness_mean:.6f}, "
            f"std={summary.best_fitness_std:.6f}, "
            f"feasible_rate={summary.feasible_rate:.3f}, "
            f"mean_evals={summary.evaluations_mean:.1f}"
        )

    print()
    print(f"Artifacts: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled Step 8 QPSO research experiments."
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO,
        help="Path to the Scenario JSON fixture.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for machine-readable experiment artifacts.",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=20,
        help="Optimizer population size.",
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=100,
        help="Hard evaluator-call budget per run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Optimizer seeds.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["canonical"],
        help="QPSO variant names to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_experiment(
        scenario_path=args.scenario,
        output_dir=args.output_dir,
        particles=args.particles,
        evaluations=args.evaluations,
        seeds=args.seeds,
        conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
