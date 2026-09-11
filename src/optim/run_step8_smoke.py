from __future__ import annotations

import json
from pathlib import Path

from src.contracts.scenario import Scenario
from src.optim.experiments import run_ablation_suite
from src.optim.qpso import AdaptiveQPSOConfig
from src.optim.experiments import DEFAULT_QPSO_CONDITIONS
from src.optim.qpso import RouteFitnessOracle
from src.routing.route_encoding import Step7RouteEngine
from src.routing.route_evaluator import RouteEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_PATH = (
    PROJECT_ROOT
    / "fixtures"
    / "step3"
    / "base"
    / "scenario.json"
)


def load_scenario() -> Scenario:
    data = json.loads(
        SCENARIO_PATH.read_text(encoding="utf-8")
    )
    return Scenario.model_validate(data)


def make_oracle_factory():
    scenario = load_scenario()

    def factory(seed: int):
        # Every run gets an isolated evaluator/engine/oracle.
        evaluator = RouteEvaluator(scenario)

        engine = Step7RouteEngine(
            scenario,
            evaluator,
        )

        return RouteFitnessOracle(
            engine,
            planning_time_s=0.0,
            repair=True,
            repair_penalty=0.0,
        )

    return factory


def main() -> None:
    scenario = load_scenario()

    dimensions = 2 * len(scenario.requests)

    config = AdaptiveQPSOConfig(
        dimensions=dimensions,
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=4,
        max_evaluations=20,
        seed=7,
        tolerance=1e-12,
        stagnation_patience=5,
    )

    oracle_factory = make_oracle_factory()

    runs, summaries = run_ablation_suite(
        base_qpso_config=config,
        oracle_factory=oracle_factory,
        seeds=(7,),
        conditions=DEFAULT_QPSO_CONDITIONS,
        include_pso=True,
    )

    print("\n" + "=" * 70)
    print("STEP 8 — QPSO / PSO SMOKE EXPERIMENT")
    print("=" * 70)

    for summary in summaries:
        print(
            f"{summary.algorithm:>5} | "
            f"{summary.condition:<45} | "
            f"best={summary.best_fitness_mean:.6f} ± "
            f"{summary.best_fitness_std:.6f} | "
            f"feasible={summary.feasible_rate:.2f} | "
            f"evals={summary.evaluations_mean:.0f}"
        )

    print("=" * 70)
    print(f"Total runs: {len(runs)}")
    print("Smoke experiment completed.")


if __name__ == "__main__":
    main()