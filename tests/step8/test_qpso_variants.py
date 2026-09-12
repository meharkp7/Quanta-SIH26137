from __future__ import annotations

import pytest

from src.optim.qpso import AdaptiveQPSOConfig
from src.optim.qpso_mechanisms import (
    AlphaMode,
    AttractorMode,
    MbestMode,
    RecoveryMode,
)
from src.optim.qpso_variants import (
    VARIANTS,
    adaptive_alpha_local_attractor_mbest_qpso,
    adaptive_alpha_local_attractor_qpso,
    adaptive_alpha_qpso,
    adaptive_alpha_weighted_mbest_qpso,
    canonical_qpso,
    config_for_variant,
    full_adaptive_qpso,
    local_attractor_qpso,
    recovery_qpso,
    weighted_mbest_qpso,
)


EXPECTED_VARIANTS = {
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


def _assert_mechanisms(
    config,
    *,
    alpha,
    attractor,
    mbest,
    recovery,
):
    assert config.alpha_mode is alpha
    assert config.attractor_mode is attractor
    assert config.mbest_mode is mbest
    assert config.recovery_mode is recovery


def test_variant_catalogue_contains_expected_variants():
    assert set(VARIANTS) == EXPECTED_VARIANTS


def test_variant_catalogue_is_non_empty():
    assert VARIANTS


def test_variant_catalogue_values_are_valid_mechanism_configs():
    for name, mechanisms in VARIANTS.items():
        assert isinstance(name, str)
        assert name
        assert isinstance(
            mechanisms.alpha_mode,
            AlphaMode,
        )
        assert isinstance(
            mechanisms.attractor_mode,
            AttractorMode,
        )
        assert isinstance(
            mechanisms.mbest_mode,
            MbestMode,
        )
        assert isinstance(
            mechanisms.recovery_mode,
            RecoveryMode,
        )


def test_canonical_variant():
    _assert_mechanisms(
        canonical_qpso(),
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def test_adaptive_alpha_variant():
    _assert_mechanisms(
        adaptive_alpha_qpso(),
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def test_local_attractor_variant():
    _assert_mechanisms(
        local_attractor_qpso(),
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def test_weighted_mbest_variant():
    _assert_mechanisms(
        weighted_mbest_qpso(),
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.NONE,
    )


def test_recovery_variant():
    _assert_mechanisms(
        recovery_qpso(),
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.ADAPTIVE_BURST,
    )


def test_adaptive_alpha_local_attractor_variant():
    _assert_mechanisms(
        adaptive_alpha_local_attractor_qpso(),
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def test_adaptive_alpha_weighted_mbest_variant():
    _assert_mechanisms(
        adaptive_alpha_weighted_mbest_qpso(),
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.NONE,
    )


def test_adaptive_alpha_local_attractor_mbest_variant():
    _assert_mechanisms(
        adaptive_alpha_local_attractor_mbest_qpso(),
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.NONE,
    )


def test_full_adaptive_variant():
    _assert_mechanisms(
        full_adaptive_qpso(),
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.ADAPTIVE_BURST,
    )


@pytest.mark.parametrize(
    "variant",
    sorted(EXPECTED_VARIANTS),
)
def test_catalogue_matches_factory(
    variant,
):
    factory_map = {
        "canonical": canonical_qpso,
        "adaptive_alpha": adaptive_alpha_qpso,
        "local_attractor": local_attractor_qpso,
        "weighted_mbest": weighted_mbest_qpso,
        "recovery": recovery_qpso,
        "adaptive_alpha_local_attractor": (
            adaptive_alpha_local_attractor_qpso
        ),
        "adaptive_alpha_weighted_mbest": (
            adaptive_alpha_weighted_mbest_qpso
        ),
        "adaptive_alpha_local_attractor_mbest": (
            adaptive_alpha_local_attractor_mbest_qpso
        ),
        "full_adaptive": full_adaptive_qpso,
    }

    expected = factory_map[variant]()

    assert VARIANTS[variant] == expected


def test_config_for_variant_changes_only_mechanisms():
    base = AdaptiveQPSOConfig(
        dimensions=4,
        population_size=6,
        lower_bound=-2.0,
        upper_bound=2.0,
        max_evaluations=60,
        seed=123,
        alpha_max=0.91,
        alpha_min=0.47,
        progress_weight=0.51,
        diversity_weight=0.29,
        stagnation_weight=0.20,
        pbest_weight=0.48,
        gbest_weight=0.32,
        mbest_weight=0.20,
        recovery_probability=0.17,
        recovery_scale=0.31,
    )

    result = config_for_variant(
        base,
        "full_adaptive",
    )

    assert result is not base

    assert result.dimensions == base.dimensions
    assert result.population_size == base.population_size
    assert result.lower_bound == base.lower_bound
    assert result.upper_bound == base.upper_bound
    assert result.max_evaluations == base.max_evaluations
    assert result.seed == base.seed

    assert result.alpha_max == base.alpha_max
    assert result.alpha_min == base.alpha_min

    assert (
        result.progress_weight
        == base.progress_weight
    )
    assert (
        result.diversity_weight
        == base.diversity_weight
    )
    assert (
        result.stagnation_weight
        == base.stagnation_weight
    )

    assert result.pbest_weight == base.pbest_weight
    assert result.gbest_weight == base.gbest_weight
    assert result.mbest_weight == base.mbest_weight

    assert (
        result.recovery_probability
        == base.recovery_probability
    )
    assert (
        result.recovery_scale
        == base.recovery_scale
    )

    assert result.mechanisms == VARIANTS[
        "full_adaptive"
    ]


@pytest.mark.parametrize(
    "variant",
    sorted(EXPECTED_VARIANTS),
)
def test_config_for_variant_selects_correct_mechanism(
    variant,
):
    base = AdaptiveQPSOConfig(
        dimensions=3,
        population_size=5,
        lower_bound=0.0,
        upper_bound=1.0,
        max_evaluations=50,
        seed=42,
    )

    result = config_for_variant(
        base,
        variant,
    )

    assert result.mechanisms == VARIANTS[
        variant
    ]


def test_config_for_variant_returns_independent_config():
    base = AdaptiveQPSOConfig(
        dimensions=2,
        population_size=4,
        lower_bound=0.0,
        upper_bound=1.0,
        max_evaluations=40,
        seed=7,
    )

    result = config_for_variant(
        base,
        "adaptive_alpha",
    )

    assert result is not base
    assert result.mechanisms == VARIANTS[
        "adaptive_alpha"
    ]


@pytest.mark.parametrize(
    "variant",
    [
        "",
        "   ",
        "does_not_exist",
        "FULL_ADAPTIVE",
    ],
)
def test_config_for_variant_rejects_unknown_variants(
    variant,
):
    base = AdaptiveQPSOConfig(
        dimensions=2,
        population_size=4,
        lower_bound=0.0,
        upper_bound=1.0,
        max_evaluations=40,
        seed=7,
    )

    with pytest.raises(ValueError):
        config_for_variant(
            base,
            variant,
        )


def test_config_for_variant_strips_surrounding_whitespace():
    base = AdaptiveQPSOConfig(
        dimensions=2,
        population_size=4,
        lower_bound=0.0,
        upper_bound=1.0,
        max_evaluations=40,
        seed=7,
    )

    result = config_for_variant(
        base,
        "  full_adaptive  ",
    )

    assert result.mechanisms == VARIANTS[
        "full_adaptive"
    ]


@pytest.mark.parametrize(
    "invalid_base",
    [
        None,
        object(),
        "config",
        123,
    ],
)
def test_config_for_variant_rejects_invalid_base_config(
    invalid_base,
):
    with pytest.raises(TypeError):
        config_for_variant(
            invalid_base,
            "canonical",
        )


@pytest.mark.parametrize(
    "invalid_variant",
    [
        None,
        1,
        1.5,
        [],
        {},
    ],
)
def test_config_for_variant_rejects_non_string_variant(
    invalid_variant,
):
    base = AdaptiveQPSOConfig(
        dimensions=2,
        population_size=4,
        lower_bound=0.0,
        upper_bound=1.0,
        max_evaluations=40,
        seed=7,
    )

    with pytest.raises(TypeError):
        config_for_variant(
            base,
            invalid_variant,
        )


def test_variant_mechanism_configs_are_immutable():
    mechanisms = VARIANTS["canonical"]

    with pytest.raises(Exception):
        mechanisms.alpha_mode = AlphaMode.ADAPTIVE


def test_variant_catalogue_is_immutable():
    with pytest.raises(TypeError):
        VARIANTS["new_variant"] = canonical_qpso()