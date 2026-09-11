from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from src.optim.qpso import AdaptiveQPSOConfig
from src.optim.qpso_mechanisms import (
    AlphaMode,
    AttractorMode,
    MbestMode,
    RecoveryMode,
    QPSOMechanismConfig,
)


def _mechanisms(
    *,
    alpha: AlphaMode,
    attractor: AttractorMode,
    mbest: MbestMode,
    recovery: RecoveryMode,
) -> QPSOMechanismConfig:
    return QPSOMechanismConfig(
        alpha_mode=alpha,
        attractor_mode=attractor,
        mbest_mode=mbest,
        recovery_mode=recovery,
    )


def canonical_qpso() -> QPSOMechanismConfig:
    """Canonical QPSO comparator."""
    return _mechanisms(
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def adaptive_alpha_qpso() -> QPSOMechanismConfig:
    """Adaptive-alpha-only ablation."""
    return _mechanisms(
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def local_attractor_qpso() -> QPSOMechanismConfig:
    """Blended-attractor-only ablation."""
    return _mechanisms(
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def weighted_mbest_qpso() -> QPSOMechanismConfig:
    """Fitness-weighted mbest-only ablation."""
    return _mechanisms(
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.NONE,
    )


def recovery_qpso() -> QPSOMechanismConfig:
    """Recovery-only ablation."""
    return _mechanisms(
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.ADAPTIVE_BURST,
    )


def adaptive_alpha_local_attractor_qpso() -> QPSOMechanismConfig:
    return _mechanisms(
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def adaptive_alpha_weighted_mbest_qpso() -> QPSOMechanismConfig:
    return _mechanisms(
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.NONE,
    )


def adaptive_alpha_local_attractor_mbest_qpso() -> QPSOMechanismConfig:
    return _mechanisms(
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.NONE,
    )


def full_adaptive_qpso() -> QPSOMechanismConfig:
    """Current fully adaptive configuration."""
    return _mechanisms(
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.ADAPTIVE_BURST,
    )


VARIANTS: Mapping[str, QPSOMechanismConfig] = {
    "canonical": canonical_qpso(),
    "adaptive_alpha": adaptive_alpha_qpso(),
    "local_attractor": local_attractor_qpso(),
    "weighted_mbest": weighted_mbest_qpso(),
    "recovery": recovery_qpso(),
    "adaptive_alpha_local_attractor": adaptive_alpha_local_attractor_qpso(),
    "adaptive_alpha_weighted_mbest": adaptive_alpha_weighted_mbest_qpso(),
    "adaptive_alpha_local_attractor_mbest": adaptive_alpha_local_attractor_mbest_qpso(),
    "full_adaptive": full_adaptive_qpso(),
}


def config_for_variant(
    base_config: AdaptiveQPSOConfig,
    variant: str,
) -> AdaptiveQPSOConfig:
    """Return a copy of base_config with only mechanism selection changed."""
    try:
        mechanisms = VARIANTS[variant]
    except KeyError as exc:
        available = ", ".join(sorted(VARIANTS))
        raise ValueError(
            f"unknown QPSO variant {variant!r}; available: {available}"
        ) from exc

    return replace(base_config, mechanisms=mechanisms)