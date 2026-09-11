from __future__ import annotations

import math

import pytest

from src.optim.common import (
    FitnessOracle,
    FitnessResult,
)
from src.optim.qpso import (
    AdaptiveQPSO,
    AdaptiveQPSOConfig,
)


def sphere(position):
    return FitnessResult(
        fitness=sum(x * x for x in position),
        feasible=True,
        route_signature=(
            ("route", tuple(int(x * 10) for x in position)),
        ),
    )


def make_config(**overrides):
    values = dict(
        dimensions=4,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=12,
        max_evaluations=240,
        seed=17,
    )
    values.update(overrides)
    return AdaptiveQPSOConfig(**values)


def test_qpso_is_deterministic_for_same_seed():
    config_a = make_config(seed=41)
    config_b = make_config(seed=41)

    result_a = AdaptiveQPSO(
        config_a,
        FitnessOracle(sphere),
    ).optimize()

    result_b = AdaptiveQPSO(
        config_b,
        FitnessOracle(sphere),
    ).optimize()

    assert result_a == result_b


def test_qpso_respects_evaluation_budget():
    config = make_config(max_evaluations=37)

    oracle = FitnessOracle(sphere)

    result = AdaptiveQPSO(
        config,
        oracle,
    ).optimize()

    assert result.evaluations == 37
    assert oracle.calls == 37


def test_qpso_never_leaves_bounds():
    config = make_config(
        lower_bound=-1.0,
        upper_bound=1.0,
        max_evaluations=100,
    )

    result = AdaptiveQPSO(
        config,
        FitnessOracle(sphere),
    ).optimize()

    assert all(
        config.lower_bound <= value <= config.upper_bound
        for value in result.best_position
    )


def test_qpso_improves_sphere():
    config = make_config(
        dimensions=3,
        population_size=10,
        max_evaluations=500,
        seed=9,
    )

    result = AdaptiveQPSO(
        config,
        FitnessOracle(sphere),
    ).optimize()

    assert result.best_fitness < 1.0
    assert result.history_best[-1] <= result.history_best[0]


def test_qpso_records_both_diversity_dimensions():
    config = make_config(
        dimensions=3,
        population_size=8,
        max_evaluations=80,
    )

    result = AdaptiveQPSO(
        config,
        FitnessOracle(sphere),
    ).optimize()

    assert result.history_coordinate_diversity
    assert result.history_route_diversity

    assert len(result.history_coordinate_diversity) == (
        len(result.history_best)
    )

    assert len(result.history_route_diversity) == (
        len(result.history_best)
    )


def test_qpso_supports_explicit_initial_population():
    population = [
        [-4.0, -4.0],
        [-3.0, -3.0],
        [-2.0, -2.0],
        [-1.0, -1.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]

    config = AdaptiveQPSOConfig(
        dimensions=2,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=8,
        max_evaluations=32,
        seed=7,
    )

    result = AdaptiveQPSO(
        config,
        FitnessOracle(sphere),
        initial_population=population,
    ).optimize()

    assert result.evaluations == 32
    assert result.best_fitness <= 0.1
    assert math.isfinite(result.best_fitness)


def test_invalid_adaptation_weights_are_rejected():
    with pytest.raises(ValueError):
        make_config(
            progress_weight=0.5,
            diversity_weight=0.5,
            stagnation_weight=0.5,
        )