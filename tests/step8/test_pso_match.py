from __future__ import annotations

from src.optim.common import FitnessOracle, FitnessResult
from src.optim.pso import MatchedPSO, MatchedPSOConfig
from src.optim.qpso import AdaptiveQPSO, AdaptiveQPSOConfig


def objective(position):
    return FitnessResult(
        fitness=sum(x * x for x in position),
        feasible=True,
        route_signature=(
            ("route", tuple(int(x * 10) for x in position)),
        ),
    )


def shared_population():
    return [
        [-0.9, -0.7, -0.5],
        [-0.6, -0.4, -0.2],
        [-0.3, -0.1, 0.1],
        [0.0, 0.2, 0.4],
        [0.3, 0.5, 0.7],
        [0.6, 0.8, 0.9],
    ]


def test_qpso_and_pso_use_equal_evaluation_budgets():
    population = shared_population()

    qpso_config = AdaptiveQPSOConfig(
        dimensions=3,
        lower_bound=-1.0,
        upper_bound=1.0,
        population_size=6,
        max_evaluations=42,
        seed=123,
    )

    pso_config = MatchedPSOConfig(
        dimensions=3,
        lower_bound=-1.0,
        upper_bound=1.0,
        population_size=6,
        max_evaluations=42,
        seed=123,
    )

    qpso_oracle = FitnessOracle(objective)
    pso_oracle = FitnessOracle(objective)

    qpso_result = AdaptiveQPSO(
        qpso_config,
        qpso_oracle,
        initial_population=population,
    ).optimize()

    pso_result = MatchedPSO(
        pso_config,
        pso_oracle,
        initial_population=population,
    ).optimize()

    assert qpso_result.evaluations == 42
    assert pso_result.evaluations == 42

    assert qpso_oracle.calls == pso_oracle.calls == 42


def test_pso_is_deterministic_for_same_seed():
    config_a = MatchedPSOConfig(
        dimensions=3,
        lower_bound=-1.0,
        upper_bound=1.0,
        population_size=8,
        max_evaluations=80,
        seed=44,
    )

    config_b = MatchedPSOConfig(
        dimensions=3,
        lower_bound=-1.0,
        upper_bound=1.0,
        population_size=8,
        max_evaluations=80,
        seed=44,
    )

    result_a = MatchedPSO(
        config_a,
        FitnessOracle(objective),
    ).optimize()

    result_b = MatchedPSO(
        config_b,
        FitnessOracle(objective),
    ).optimize()

    assert result_a == result_b


def test_pso_improves_objective():
    config = MatchedPSOConfig(
        dimensions=4,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=12,
        max_evaluations=240,
        seed=12,
    )

    result = MatchedPSO(
        config,
        FitnessOracle(objective),
    ).optimize()

    assert result.best_fitness < 2.0
    assert result.history_best[-1] <= result.history_best[0]