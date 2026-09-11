from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, isfinite
import random
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from src.optim.qpso import _Particle, AdaptiveQPSOConfig


class AlphaMode(str, Enum):
    """Contraction-expansion coefficient strategies."""

    CANONICAL = "canonical"
    ADAPTIVE = "adaptive"


class AttractorMode(str, Enum):
    """Local-attractor composition strategies."""

    CANONICAL = "canonical"
    BLENDED = "blended"


class MbestMode(str, Enum):
    """Personal-best mean strategies."""

    UNIFORM = "uniform"
    WEIGHTED = "weighted"


class RecoveryMode(str, Enum):
    """Exploration recovery strategies."""

    NONE = "none"
    ADAPTIVE_BURST = "adaptive_burst"


@dataclass(frozen=True)
class QPSOMechanismConfig:
    """
    Explicit switches for controlled QPSO ablations.

    Each mechanism is independently selectable so experimental variants
    can be compared without maintaining separate optimizer implementations.
    """

    alpha_mode: AlphaMode = AlphaMode.ADAPTIVE
    attractor_mode: AttractorMode = AttractorMode.BLENDED
    mbest_mode: MbestMode = MbestMode.WEIGHTED
    recovery_mode: RecoveryMode = RecoveryMode.ADAPTIVE_BURST
    canonical_alpha: float = 0.75

    def __post_init__(self) -> None:
        if not isinstance(self.alpha_mode, AlphaMode):
            raise ValueError("alpha_mode must be an AlphaMode")
        if not isinstance(self.attractor_mode, AttractorMode):
            raise ValueError("attractor_mode must be an AttractorMode")
        if not isinstance(self.mbest_mode, MbestMode):
            raise ValueError("mbest_mode must be an MbestMode")
        if not isinstance(self.recovery_mode, RecoveryMode):
            raise ValueError("recovery_mode must be a RecoveryMode")
        if not isfinite(self.canonical_alpha):
            raise ValueError("canonical_alpha must be finite")
        if not 0.0 < self.canonical_alpha < 1.5:
            raise ValueError("canonical_alpha must be in (0, 1.5)")


def compute_mbest(
    particles: Sequence["_Particle"],
    *,
    dimensions: int,
    mode: MbestMode,
) -> tuple[float, ...]:
    """Return either canonical uniform mbest or fitness-weighted mbest."""
    if not particles:
        raise ValueError("particles cannot be empty")
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")

    if mode is MbestMode.UNIFORM:
        return tuple(
            sum(particle.pbest_position[d] for particle in particles)
            / len(particles)
            for d in range(dimensions)
        )

    best = min(p.pbest_result.fitness for p in particles)
    worst = max(p.pbest_result.fitness for p in particles)
    span = max(worst - best, 1e-12)

    weights = [
        exp(-(particle.pbest_result.fitness - best) / span)
        for particle in particles
    ]
    total = sum(weights)

    if total <= 0.0 or not isfinite(total):
        return tuple(
            sum(particle.pbest_position[d] for particle in particles)
            / len(particles)
            for d in range(dimensions)
        )

    return tuple(
        sum(
            weight * particle.pbest_position[d]
            for weight, particle in zip(weights, particles)
        ) / total
        for d in range(dimensions)
    )


def compute_alpha(
    *,
    config: "AdaptiveQPSOConfig",
    coordinate_div: float,
    route_div: float,
    progress: float,
    stagnation: int,
    mode: AlphaMode,
) -> float:
    """Return the configured QPSO contraction-expansion coefficient."""
    if mode is AlphaMode.CANONICAL:
        return config.mechanisms.canonical_alpha

    diversity_signal = 0.5 * (
        max(0.0, min(1.0, coordinate_div))
        + max(0.0, min(1.0, route_div))
    )
    stagnation_signal = min(
        1.0,
        max(0.0, stagnation) / config.stagnation_patience,
    )
    progress_signal = max(0.0, min(1.0, progress))

    raw = (
        config.progress_weight * progress_signal
        + config.diversity_weight * diversity_signal
        + config.stagnation_weight * (1.0 - stagnation_signal)
    )

    alpha = (
        config.alpha_max
        - raw * (config.alpha_max - config.alpha_min)
    )

    if stagnation_signal > 0.5:
        alpha += (
            config.alpha_max - config.alpha_min
        ) * 0.25 * stagnation_signal

    return min(
        config.alpha_max,
        max(config.alpha_min, alpha),
    )


def compute_attractor(
    *,
    particle: "_Particle",
    global_best: "_Particle",
    mbest: Sequence[float],
    dimensions: int,
    mode: AttractorMode,
    pbest_weight: float,
    gbest_weight: float,
    mbest_weight: float,
) -> tuple[float, ...]:
    """Return the selected local-attractor composition."""
    if mode is AttractorMode.CANONICAL:
        # Standard QPSO local attractor:
        # p = phi * pbest + (1 - phi) * gbest.
        # The configured pbest/gbest weights are normalized for the
        # controlled comparator; mbest does not participate.
        total = pbest_weight + gbest_weight
        if total <= 0.0:
            pbest_ratio = 0.5
        else:
            pbest_ratio = pbest_weight / total
        gbest_ratio = 1.0 - pbest_ratio

        return tuple(
            pbest_ratio * particle.pbest_position[d]
            + gbest_ratio * global_best.pbest_position[d]
            for d in range(dimensions)
        )

    return tuple(
        pbest_weight * particle.pbest_position[d]
        + gbest_weight * global_best.pbest_position[d]
        + mbest_weight * mbest[d]
        for d in range(dimensions)
    )


def should_recover(
    *,
    config: "AdaptiveQPSOConfig",
    stagnation: int,
    rng: random.Random,
) -> bool:
    """Return whether the configured recovery mechanism fires."""
    if config.mechanisms.recovery_mode is RecoveryMode.NONE:
        return False

    if stagnation < config.stagnation_patience:
        return False

    return rng.random() < config.recovery_probability
