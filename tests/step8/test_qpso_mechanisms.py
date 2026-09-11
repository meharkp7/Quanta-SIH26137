

import random

import pytest

from src.optim.qpso_mechanisms import (
    AlphaMode,
    AttractorMode,
    MbestMode,
    QPSOMechanismConfig,
    RecoveryMode,
    compute_alpha,
    compute_attractor,
    compute_mbest,
)
from src.optim.qpso_variants import VARIANTS, config_for_variant


def test_all_named_variants_are_available():
    expected = {
        "canonical",
        "adaptive_alpha",
        "local_attractor",
        "weighted_mbest",
        "recovery",
        "adaptive_alpha_local_attractor",
        "adaptive_alpha_weighted_mbest",
        "adaptive_alpha_local_attractor_mbest",
        "full_adaptive",
    }
    assert set(VARIANTS) == expected


def test_canonical_variant_disables_adaptive_mechanisms():
    mechanisms = VARIANTS["canonical"]

    assert mechanisms.alpha_mode is AlphaMode.CANONICAL
    assert mechanisms.attractor_mode is AttractorMode.CANONICAL
    assert mechanisms.mbest_mode is MbestMode.UNIFORM
    assert mechanisms.recovery_mode is RecoveryMode.NONE


def test_variant_changes_only_mechanism_selection():
    # This is deliberately tested through a tiny fake base object shape rather
    # than constructing a full routing problem.
    from src.optim.qpso import AdaptiveQPSOConfig

    base = AdaptiveQPSOConfig(
        dimensions=2,
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=3,
        max_evaluations=6,
        seed=7,
    )

    configured = config_for_variant(base, "canonical")

    assert configured.dimensions == base.dimensions
    assert configured.population_size == base.population_size
    assert configured.max_evaluations == base.max_evaluations
    assert configured.seed == base.seed
    assert configured.mechanisms == VARIANTS["canonical"]


def test_invalid_variant_has_actionable_error():
    from src.optim.qpso import AdaptiveQPSOConfig

    base = AdaptiveQPSOConfig(
        dimensions=2,
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=3,
        max_evaluations=6,
        seed=7,
    )

    with pytest.raises(ValueError, match="unknown QPSO variant"):
        config_for_variant(base, "does_not_exist")


def test_uniform_mbest_is_coordinatewise_mean():
    class Result:
        def __init__(self, fitness):
            self.fitness = fitness

    class Particle:
        def __init__(self, position, fitness):
            self.pbest_position = list(position)
            self.pbest_result = Result(fitness)

    particles = [
        Particle((0.0, 1.0), 3.0),
        Particle((1.0, 0.0), 2.0),
    ]

    assert compute_mbest(
        particles,
        dimensions=2,
        mode=MbestMode.UNIFORM,
    ) == (0.5, 0.5)


def test_weighted_mbest_favors_better_personal_best():
    class Result:
        def __init__(self, fitness):
            self.fitness = fitness

    class Particle:
        def __init__(self, position, fitness):
            self.pbest_position = list(position)
            self.pbest_result = Result(fitness)

    particles = [
        Particle((0.0, 0.0), 0.0),
        Particle((1.0, 1.0), 10.0),
    ]

    result = compute_mbest(
        particles,
        dimensions=2,
        mode=MbestMode.WEIGHTED,
    )

    assert result[0] < 0.5
    assert result[1] < 0.5


def test_canonical_alpha_is_constant():
    class Mechanisms:
        canonical_alpha = 0.7

    class Config:
        mechanisms = Mechanisms()
        alpha_min = 0.4
        alpha_max = 0.9
        progress_weight = 0.45
        diversity_weight = 0.30
        stagnation_weight = 0.25
        stagnation_patience = 5

    config = Config()

    values = {
        compute_alpha(
            config=config,
            coordinate_div=coordinate,
            route_div=route,
            progress=progress,
            stagnation=stagnation,
            mode=AlphaMode.CANONICAL,
        )
        for coordinate, route, progress, stagnation in (
            (0.0, 0.0, 0.0, 0),
            (1.0, 1.0, 1.0, 5),
            (0.2, 0.8, 0.4, 9),
        )
    }

    assert values == {0.7}


def test_blended_attractor_includes_mbest():
    class Result:
        def __init__(self, fitness):
            self.fitness = fitness

    class Particle:
        pbest_position = [0.0, 0.0]
        pbest_result = Result(1.0)

    class Best:
        pbest_position = [1.0, 1.0]
        pbest_result = Result(0.0)

    result = compute_attractor(
        particle=Particle(),
        global_best=Best(),
        mbest=(0.5, 0.5),
        dimensions=2,
        mode=AttractorMode.BLENDED,
        pbest_weight=0.45,
        gbest_weight=0.35,
        mbest_weight=0.20,
    )

    assert result == pytest.approx((0.45, 0.45))


def test_canonical_attractor_excludes_mbest():
    class Result:
        def __init__(self, fitness):
            self.fitness = fitness

    class Particle:
        pbest_position = [0.0, 0.0]
        pbest_result = Result(1.0)

    class Best:
        pbest_position = [1.0, 1.0]
        pbest_result = Result(0.0)

    result_a = compute_attractor(
        particle=Particle(),
        global_best=Best(),
        mbest=(0.0, 0.0),
        dimensions=2,
        mode=AttractorMode.CANONICAL,
        pbest_weight=0.45,
        gbest_weight=0.35,
        mbest_weight=0.20,
    )
    result_b = compute_attractor(
        particle=Particle(),
        global_best=Best(),
        mbest=(1.0, 1.0),
        dimensions=2,
        mode=AttractorMode.CANONICAL,
        pbest_weight=0.45,
        gbest_weight=0.35,
        mbest_weight=0.20,
    )

    assert result_a == result_b


def test_recovery_none_is_explicit():
    mechanisms = QPSOMechanismConfig(
        alpha_mode=AlphaMode.CANONICAL,
        attractor_mode=AttractorMode.CANONICAL,
        mbest_mode=MbestMode.UNIFORM,
        recovery_mode=RecoveryMode.NONE,
    )

    assert mechanisms.recovery_mode is RecoveryMode.NONE