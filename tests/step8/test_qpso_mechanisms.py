from __future__ import annotations

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
    should_recover,
)
from src.optim.qpso_variants import (
    VARIANTS,
    config_for_variant,
)


class FakeResult:
    def __init__(self, fitness: float):
        self.fitness = fitness


class FakeParticle:
    def __init__(
        self,
        position,
        fitness: float,
    ):
        self.pbest_position = list(position)
        self.pbest_result = FakeResult(fitness)


# ---------------------------------------------------------------------------
# Variant configuration
# ---------------------------------------------------------------------------


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


def test_full_adaptive_variant_enables_all_adaptive_mechanisms():
    mechanisms = VARIANTS["full_adaptive"]

    assert mechanisms.alpha_mode is AlphaMode.ADAPTIVE
    assert mechanisms.attractor_mode is AttractorMode.BLENDED
    assert mechanisms.mbest_mode is MbestMode.WEIGHTED
    assert mechanisms.recovery_mode is RecoveryMode.ADAPTIVE_BURST


def test_variant_changes_only_mechanism_selection():
    from src.optim.qpso import AdaptiveQPSOConfig

    base = AdaptiveQPSOConfig(
        dimensions=2,
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=3,
        max_evaluations=6,
        seed=7,
    )

    configured = config_for_variant(
        base,
        "canonical",
    )

    assert configured.dimensions == base.dimensions
    assert configured.lower_bound == base.lower_bound
    assert configured.upper_bound == base.upper_bound
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

    with pytest.raises(
        ValueError,
        match="unknown QPSO variant",
    ):
        config_for_variant(
            base,
            "does_not_exist",
        )


# ---------------------------------------------------------------------------
# Mechanism configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_mode": "adaptive"},
        {"attractor_mode": "canonical"},
        {"mbest_mode": "uniform"},
        {"recovery_mode": "none"},
    ],
)
def test_mechanism_config_rejects_invalid_enum_types(kwargs):
    base = dict(
        alpha_mode=AlphaMode.ADAPTIVE,
        attractor_mode=AttractorMode.BLENDED,
        mbest_mode=MbestMode.WEIGHTED,
        recovery_mode=RecoveryMode.ADAPTIVE_BURST,
    )

    base.update(kwargs)

    with pytest.raises(ValueError):
        QPSOMechanismConfig(**base)


@pytest.mark.parametrize(
    "alpha",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        1.5,
        2.0,
    ],
)
def test_mechanism_config_rejects_invalid_canonical_alpha(alpha):
    with pytest.raises(ValueError):
        QPSOMechanismConfig(
            canonical_alpha=alpha
        )


def test_mechanism_config_accepts_valid_canonical_alpha():
    config = QPSOMechanismConfig(
        canonical_alpha=0.75
    )

    assert config.canonical_alpha == pytest.approx(
        0.75
    )


# ---------------------------------------------------------------------------
# Mean-best
# ---------------------------------------------------------------------------


def test_uniform_mbest_is_coordinatewise_mean():
    particles = [
        FakeParticle((0.0, 1.0), 3.0),
        FakeParticle((1.0, 0.0), 2.0),
    ]

    result = compute_mbest(
        particles,
        dimensions=2,
        mode=MbestMode.UNIFORM,
    )

    assert result == pytest.approx(
        (0.5, 0.5)
    )


def test_uniform_mbest_supports_more_than_two_dimensions():
    particles = [
        FakeParticle(
            (0.0, 1.0, 2.0, 3.0),
            5.0,
        ),
        FakeParticle(
            (2.0, 3.0, 4.0, 5.0),
            1.0,
        ),
        FakeParticle(
            (4.0, 5.0, 6.0, 7.0),
            3.0,
        ),
    ]

    result = compute_mbest(
        particles,
        dimensions=4,
        mode=MbestMode.UNIFORM,
    )

    assert result == pytest.approx(
        (2.0, 3.0, 4.0, 5.0)
    )


def test_weighted_mbest_favors_better_personal_best():
    particles = [
        FakeParticle(
            (0.0, 0.0),
            0.0,
        ),
        FakeParticle(
            (1.0, 1.0),
            10.0,
        ),
    ]

    result = compute_mbest(
        particles,
        dimensions=2,
        mode=MbestMode.WEIGHTED,
    )

    assert result[0] < 0.5
    assert result[1] < 0.5


def test_weighted_mbest_collapses_to_uniform_when_fitnesses_are_equal():
    particles = [
        FakeParticle(
            (0.0, 2.0),
            5.0,
        ),
        FakeParticle(
            (2.0, 0.0),
            5.0,
        ),
    ]

    uniform = compute_mbest(
        particles,
        dimensions=2,
        mode=MbestMode.UNIFORM,
    )

    weighted = compute_mbest(
        particles,
        dimensions=2,
        mode=MbestMode.WEIGHTED,
    )

    assert weighted == pytest.approx(
        uniform
    )


def test_mbest_rejects_empty_population():
    with pytest.raises(
        ValueError,
        match="particles cannot be empty",
    ):
        compute_mbest(
            [],
            dimensions=2,
            mode=MbestMode.UNIFORM,
        )


def test_mbest_rejects_invalid_dimension():
    particles = [
        FakeParticle(
            (0.0, 1.0),
            1.0,
        )
    ]

    with pytest.raises(
        ValueError,
        match="dimension mismatch",
    ):
        compute_mbest(
            particles,
            dimensions=3,
            mode=MbestMode.UNIFORM,
        )


# ---------------------------------------------------------------------------
# Alpha
# ---------------------------------------------------------------------------


def _alpha_test_config():
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

    return Config()


def test_canonical_alpha_is_constant():
    config = _alpha_test_config()

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


def test_adaptive_alpha_is_bounded():
    config = _alpha_test_config()

    for coordinate in (
        -10.0,
        0.0,
        0.5,
        1.0,
        10.0,
    ):
        for route in (
            -10.0,
            0.0,
            0.5,
            1.0,
            10.0,
        ):
            for progress in (
                -10.0,
                0.0,
                0.5,
                1.0,
                10.0,
            ):
                value = compute_alpha(
                    config=config,
                    coordinate_div=coordinate,
                    route_div=route,
                    progress=progress,
                    stagnation=3,
                    mode=AlphaMode.ADAPTIVE,
                )

                assert (
                    config.alpha_min
                    <= value
                    <= config.alpha_max
                )


def test_adaptive_alpha_rejects_negative_stagnation():
    config = _alpha_test_config()

    with pytest.raises(
        ValueError,
        match="stagnation cannot be negative",
    ):
        compute_alpha(
            config=config,
            coordinate_div=0.5,
            route_div=0.5,
            progress=0.5,
            stagnation=-1,
            mode=AlphaMode.ADAPTIVE,
        )


def test_adaptive_alpha_rejects_non_finite_signal():
    config = _alpha_test_config()

    with pytest.raises(
        ValueError,
        match="coordinate_div must be finite",
    ):
        compute_alpha(
            config=config,
            coordinate_div=float("nan"),
            route_div=0.5,
            progress=0.5,
            stagnation=0,
            mode=AlphaMode.ADAPTIVE,
        )


# ---------------------------------------------------------------------------
# Attractor
# ---------------------------------------------------------------------------


def test_canonical_attractor_implements_phi_interpolation():
    particle = FakeParticle(
        (0.0, 0.0),
        1.0,
    )

    global_best = FakeParticle(
        (1.0, 1.0),
        0.0,
    )

    result = compute_attractor(
        particle=particle,
        global_best=global_best,
        mbest=(0.5, 0.5),
        dimensions=2,
        mode=AttractorMode.CANONICAL,
        pbest_weight=0.45,
        gbest_weight=0.35,
        mbest_weight=0.20,
        phi=(0.25, 0.75),
    )

    assert result == pytest.approx(
        (0.75, 0.25)
    )


def test_canonical_attractor_ignores_mbest():
    particle = FakeParticle(
        (0.0, 0.0),
        1.0,
    )

    global_best = FakeParticle(
        (1.0, 1.0),
        0.0,
    )

    result_a = compute_attractor(
        particle=particle,
        global_best=global_best,
        mbest=(0.0, 0.0),
        dimensions=2,
        mode=AttractorMode.CANONICAL,
        pbest_weight=0.45,
        gbest_weight=0.35,
        mbest_weight=0.20,
        phi=(0.25, 0.75),
    )

    result_b = compute_attractor(
        particle=particle,
        global_best=global_best,
        mbest=(1.0, 1.0),
        dimensions=2,
        mode=AttractorMode.CANONICAL,
        pbest_weight=0.45,
        gbest_weight=0.35,
        mbest_weight=0.20,
        phi=(0.25, 0.75),
    )

    assert result_a == pytest.approx(
        result_b
    )


def test_canonical_phi_zero_selects_global_best():
    particle = FakeParticle(
        (0.2, 0.4),
        1.0,
    )

    global_best = FakeParticle(
        (0.8, 0.6),
        0.0,
    )

    result = compute_attractor(
        particle=particle,
        global_best=global_best,
        mbest=(0.5, 0.5),
        dimensions=2,
        mode=AttractorMode.CANONICAL,
        pbest_weight=0.45,
        gbest_weight=0.35,
        mbest_weight=0.20,
        phi=(0.0, 0.0),
    )

    assert result == pytest.approx(
        (0.8, 0.6)
    )


def test_canonical_phi_one_selects_personal_best():
    particle = FakeParticle(
        (0.2, 0.4),
        1.0,
    )

    global_best = FakeParticle(
        (0.8, 0.6),
        0.0,
    )

    result = compute_attractor(
        particle=particle,
        global_best=global_best,
        mbest=(0.5, 0.5),
        dimensions=2,
        mode=AttractorMode.CANONICAL,
        pbest_weight=0.45,
        gbest_weight=0.35,
        mbest_weight=0.20,
        phi=(1.0, 1.0),
    )

    assert result == pytest.approx(
        (0.2, 0.4)
    )


def test_canonical_attractor_backward_compatible_without_phi():
    particle = FakeParticle(
        (0.0, 0.0),
        1.0,
    )

    global_best = FakeParticle(
        (1.0, 1.0),
        0.0,
    )

    result = compute_attractor(
        particle=particle,
        global_best=global_best,
        mbest=(0.5, 0.5),
        dimensions=2,
        mode=AttractorMode.CANONICAL,
        pbest_weight=0.45,
        gbest_weight=0.35,
        mbest_weight=0.20,
    )

    assert result == pytest.approx(
        (0.5, 0.5)
    )


def test_canonical_attractor_rejects_invalid_phi_dimension():
    particle = FakeParticle(
        (0.0, 0.0),
        1.0,
    )

    global_best = FakeParticle(
        (1.0, 1.0),
        0.0,
    )

    with pytest.raises(
        ValueError,
        match="phi dimension mismatch",
    ):
        compute_attractor(
            particle=particle,
            global_best=global_best,
            mbest=(0.5, 0.5),
            dimensions=2,
            mode=AttractorMode.CANONICAL,
            pbest_weight=0.45,
            gbest_weight=0.35,
            mbest_weight=0.20,
            phi=(0.5,),
        )


@pytest.mark.parametrize(
    "phi",
    [
        (-0.1, 0.5),
        (0.5, 1.1),
        (float("nan"), 0.5),
        (float("inf"), 0.5),
    ],
)
def test_canonical_attractor_rejects_invalid_phi_values(phi):
    particle = FakeParticle(
        (0.0, 0.0),
        1.0,
    )

    global_best = FakeParticle(
        (1.0, 1.0),
        0.0,
    )

    with pytest.raises(ValueError):
        compute_attractor(
            particle=particle,
            global_best=global_best,
            mbest=(0.5, 0.5),
            dimensions=2,
            mode=AttractorMode.CANONICAL,
            pbest_weight=0.45,
            gbest_weight=0.35,
            mbest_weight=0.20,
            phi=phi,
        )


def test_blended_attractor_includes_mbest():
    particle = FakeParticle(
        (0.0, 0.0),
        1.0,
    )

    global_best = FakeParticle(
        (1.0, 1.0),
        0.0,
    )

    result = compute_attractor(
        particle=particle,
        global_best=global_best,
        mbest=(0.5, 0.5),
        dimensions=2,
        mode=AttractorMode.BLENDED,
        pbest_weight=0.45,
        gbest_weight=0.35,
        mbest_weight=0.20,
    )

    assert result == pytest.approx(
        (0.45, 0.45)
    )


def test_blended_attractor_normalizes_weights():
    particle = FakeParticle(
        (0.0, 0.0),
        1.0,
    )

    global_best = FakeParticle(
        (1.0, 1.0),
        0.0,
    )

    result = compute_attractor(
        particle=particle,
        global_best=global_best,
        mbest=(0.5, 0.5),
        dimensions=2,
        mode=AttractorMode.BLENDED,
        pbest_weight=4.5,
        gbest_weight=3.5,
        mbest_weight=2.0,
    )

    assert result == pytest.approx(
        (0.45, 0.45)
    )


@pytest.mark.parametrize(
    "weights",
    [
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_blended_attractor_rejects_invalid_weights(
    weights
):
    particle = FakeParticle(
        (0.0, 0.0),
        1.0,
    )

    global_best = FakeParticle(
        (1.0, 1.0),
        0.0,
    )

    with pytest.raises(ValueError):
        compute_attractor(
            particle=particle,
            global_best=global_best,
            mbest=(0.5, 0.5),
            dimensions=2,
            mode=AttractorMode.BLENDED,
            pbest_weight=weights[0],
            gbest_weight=weights[1],
            mbest_weight=weights[2],
        )


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def _recovery_config(
    *,
    mode=RecoveryMode.ADAPTIVE_BURST,
    probability=1.0,
):
    class Mechanisms:
        recovery_mode = mode

    class Config:
        mechanisms = Mechanisms()
        stagnation_patience = 5
        recovery_probability = probability

    return Config()


def test_recovery_none_is_explicit():
    config = _recovery_config(
        mode=RecoveryMode.NONE
    )

    rng = random.Random(7)

    assert should_recover(
        config=config,
        stagnation=100,
        rng=rng,
    ) is False


def test_recovery_does_not_fire_before_patience():
    config = _recovery_config(
        probability=1.0
    )

    rng = random.Random(7)

    assert should_recover(
        config=config,
        stagnation=4,
        rng=rng,
    ) is False


def test_recovery_fires_when_probability_is_one():
    config = _recovery_config(
        probability=1.0
    )

    rng = random.Random(7)

    assert should_recover(
        config=config,
        stagnation=5,
        rng=rng,
    ) is True


def test_recovery_does_not_fire_when_probability_is_zero():
    config = _recovery_config(
        probability=0.0
    )

    rng = random.Random(7)

    assert should_recover(
        config=config,
        stagnation=100,
        rng=rng,
    ) is False


def test_recovery_is_reproducible_with_seeded_rng():
    config = _recovery_config(
        probability=0.5
    )

    rng_a = random.Random(123)
    rng_b = random.Random(123)

    sequence_a = [
        should_recover(
            config=config,
            stagnation=5,
            rng=rng_a,
        )
        for _ in range(20)
    ]

    sequence_b = [
        should_recover(
            config=config,
            stagnation=5,
            rng=rng_b,
        )
        for _ in range(20)
    ]

    assert sequence_a == sequence_b


def test_recovery_rejects_negative_stagnation():
    config = _recovery_config()

    with pytest.raises(
        ValueError,
        match="stagnation cannot be negative",
    ):
        should_recover(
            config=config,
            stagnation=-1,
            rng=random.Random(7),
        )