from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Final, Mapping

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
    """
    Construct a validated QPSO mechanism configuration.

    Keeping construction centralized prevents individual variant factories
    from accidentally drifting in validation or default handling.
    """
    return QPSOMechanismConfig(
        alpha_mode=alpha,
        attractor_mode=attractor,
        mbest_mode=mbest,
        recovery_mode=recovery,
    )


# ---------------------------------------------------------------------------
# Individual mechanism variants
# ---------------------------------------------------------------------------

def canonical_qpso() -> QPSOMechanismConfig:
    """
    Canonical QPSO baseline.

    No adaptive alpha, blended attractor, weighted mbest, or recovery.
    """
    return _mechanisms(
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def adaptive_alpha_qpso() -> QPSOMechanismConfig:
    """
    Ablation enabling adaptive alpha only.
    """
    return _mechanisms(
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def local_attractor_qpso() -> QPSOMechanismConfig:
    """
    Ablation enabling the blended/local attractor only.
    """
    return _mechanisms(
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def weighted_mbest_qpso() -> QPSOMechanismConfig:
    """
    Ablation enabling fitness-weighted mbest only.
    """
    return _mechanisms(
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.NONE,
    )


def recovery_qpso() -> QPSOMechanismConfig:
    """
    Ablation enabling adaptive recovery bursts only.
    """
    return _mechanisms(
        alpha=AlphaMode.CANONICAL,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.ADAPTIVE_BURST,
    )


def adaptive_alpha_local_attractor_qpso() -> QPSOMechanismConfig:
    """
    Combined adaptive-alpha + blended-attractor ablation.
    """
    return _mechanisms(
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.UNIFORM,
        recovery=RecoveryMode.NONE,
    )


def adaptive_alpha_weighted_mbest_qpso() -> QPSOMechanismConfig:
    """
    Combined adaptive-alpha + weighted-mbest ablation.
    """
    return _mechanisms(
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.CANONICAL,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.NONE,
    )


def adaptive_alpha_local_attractor_mbest_qpso() -> QPSOMechanismConfig:
    """
    Combined adaptive-alpha + blended-attractor + weighted-mbest ablation.
    """
    return _mechanisms(
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.NONE,
    )


def full_adaptive_qpso() -> QPSOMechanismConfig:
    """
    Full proposed adaptive QPSO mechanism configuration.
    """
    return _mechanisms(
        alpha=AlphaMode.ADAPTIVE,
        attractor=AttractorMode.BLENDED,
        mbest=MbestMode.WEIGHTED,
        recovery=RecoveryMode.ADAPTIVE_BURST,
    )


# ---------------------------------------------------------------------------
# Registered variant catalogue
# ---------------------------------------------------------------------------

# Keep the public variant names stable because experiment runners and
# persisted traces may reference them.
#
# MappingProxyType prevents accidental mutation of the global catalogue
# while retaining normal Mapping semantics.
_VARIANTS: Final[dict[str, QPSOMechanismConfig]] = {
    "canonical": canonical_qpso(),
    "adaptive_alpha": adaptive_alpha_qpso(),
    "local_attractor": local_attractor_qpso(),
    "weighted_mbest": weighted_mbest_qpso(),
    "recovery": recovery_qpso(),
    "adaptive_alpha_local_attractor": (
        adaptive_alpha_local_attractor_qpso()
    ),
    "adaptive_alpha_weighted_mbest": (
        adaptive_alpha_weighted_mbest_qpso()
    ),
    "adaptive_alpha_local_attractor_mbest": (
        adaptive_alpha_local_attractor_mbest_qpso()
    ),
    "full_adaptive": full_adaptive_qpso(),
}


VARIANTS: Final[
    Mapping[str, QPSOMechanismConfig]
] = MappingProxyType(_VARIANTS)


# ---------------------------------------------------------------------------
# Configuration composition
# ---------------------------------------------------------------------------

def config_for_variant(
    base_config: AdaptiveQPSOConfig,
    variant: str,
) -> AdaptiveQPSOConfig:
    """
    Return a copy of ``base_config`` with only the mechanism selection
    replaced.

    All optimizer-level parameters remain inherited from ``base_config``:

        - population size
        - dimensions
        - bounds
        - evaluation budget
        - seed
        - tolerance
        - alpha parameters
        - recovery parameters
        - mechanism weights

    This is important for controlled ablation experiments: changing a
    variant must not silently change the experimental budget or optimizer
    hyperparameters.

    Parameters
    ----------
    base_config:
        Base AdaptiveQPSO configuration used for the experiment.

    variant:
        Registered variant identifier.

    Returns
    -------
    AdaptiveQPSOConfig
        Independent configuration object using the selected mechanisms.

    Raises
    ------
    TypeError
        If ``base_config`` is not an AdaptiveQPSOConfig or ``variant`` is
        not a string.

    ValueError
        If ``variant`` is not registered.
    """
    if not isinstance(
        base_config,
        AdaptiveQPSOConfig,
    ):
        raise TypeError(
            "base_config must be an AdaptiveQPSOConfig"
        )

    if not isinstance(
        variant,
        str,
    ):
        raise TypeError(
            "variant must be a string"
        )

    normalized_variant = variant.strip()

    if not normalized_variant:
        available = ", ".join(
            sorted(VARIANTS)
        )
        raise ValueError(
            "variant must not be empty; "
            f"available: {available}"
        )

    try:
        mechanisms = VARIANTS[
            normalized_variant
        ]
    except KeyError as exc:
        available = ", ".join(
            sorted(VARIANTS)
        )

        raise ValueError(
            f"unknown QPSO variant "
            f"{variant!r}; "
            f"available: {available}"
        ) from exc

    return replace(
        base_config,
        mechanisms=mechanisms,
    )


__all__ = [
    "VARIANTS",
    "adaptive_alpha_local_attractor_mbest_qpso",
    "adaptive_alpha_local_attractor_qpso",
    "adaptive_alpha_qpso",
    "adaptive_alpha_weighted_mbest_qpso",
    "canonical_qpso",
    "config_for_variant",
    "full_adaptive_qpso",
    "local_attractor_qpso",
    "recovery_qpso",
    "weighted_mbest_qpso",
]