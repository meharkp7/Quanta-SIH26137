from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, isfinite
import random
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from src.optim.qpso import AdaptiveQPSOConfig, _Particle


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

    Each mechanism is independently selectable so that variants can be
    compared without maintaining separate optimizer implementations.

    The canonical configuration corresponds to the standard QPSO mechanism:
        - canonical alpha
        - canonical local attractor
        - uniform mean-best
        - no recovery

    Adaptive variants may independently enable:
        - adaptive alpha
        - blended attractor
        - weighted mean-best
        - stagnation recovery
    """

    alpha_mode: AlphaMode = AlphaMode.ADAPTIVE
    attractor_mode: AttractorMode = AttractorMode.BLENDED
    mbest_mode: MbestMode = MbestMode.WEIGHTED
    recovery_mode: RecoveryMode = RecoveryMode.ADAPTIVE_BURST

    canonical_alpha: float = 0.75

    def __post_init__(self) -> None:
        if not isinstance(self.alpha_mode, AlphaMode):
            raise ValueError(
                "alpha_mode must be an AlphaMode"
            )

        if not isinstance(self.attractor_mode, AttractorMode):
            raise ValueError(
                "attractor_mode must be an AttractorMode"
            )

        if not isinstance(self.mbest_mode, MbestMode):
            raise ValueError(
                "mbest_mode must be a MbestMode"
            )

        if not isinstance(self.recovery_mode, RecoveryMode):
            raise ValueError(
                "recovery_mode must be a RecoveryMode"
            )

        if not isfinite(float(self.canonical_alpha)):
            raise ValueError(
                "canonical_alpha must be finite"
            )

        if not 0.0 < float(self.canonical_alpha) < 1.5:
            raise ValueError(
                "canonical_alpha must be in (0, 1.5)"
            )


def _validate_dimension(
    *,
    name: str,
    values: Sequence[float],
    dimensions: int,
) -> None:
    """Validate that a coordinate vector has the expected dimensionality."""
    if dimensions <= 0:
        raise ValueError(
            "dimensions must be positive"
        )

    if len(values) != dimensions:
        raise ValueError(
            f"{name} dimension mismatch: "
            f"expected {dimensions}, got {len(values)}"
        )


def _validate_finite_sequence(
    values: Sequence[float],
    *,
    name: str,
) -> None:
    """Reject NaN and infinite mechanism inputs."""
    for index, value in enumerate(values):
        if not isfinite(float(value)):
            raise ValueError(
                f"{name}[{index}] must be finite"
            )


def compute_mbest(
    particles: Sequence["_Particle"],
    *,
    dimensions: int,
    mode: MbestMode,
) -> tuple[float, ...]:
    """
    Compute the mean-best vector from particle personal-best positions.

    UNIFORM
        Canonical arithmetic mean:

            m_d = (1 / S) * sum_p P_pd

    WEIGHTED
        Fitness-weighted mean where better personal-best fitness receives
        greater weight. This is an adaptive experimental mechanism and is
        intentionally not used by the canonical QPSO variant.

    Lower fitness is assumed to be better.
    """
    if not particles:
        raise ValueError(
            "particles cannot be empty"
        )

    if dimensions <= 0:
        raise ValueError(
            "dimensions must be positive"
        )

    if not isinstance(mode, MbestMode):
        raise ValueError(
            "mode must be an MbestMode"
        )

    for index, particle in enumerate(particles):
        _validate_dimension(
            name=f"particle[{index}].pbest_position",
            values=particle.pbest_position,
            dimensions=dimensions,
        )

        _validate_finite_sequence(
            particle.pbest_position,
            name=f"particle[{index}].pbest_position",
        )

        if not isfinite(float(particle.pbest_result.fitness)):
            raise ValueError(
                f"particle[{index}].pbest_result.fitness must be finite"
            )

    if mode is MbestMode.UNIFORM:
        return tuple(
            sum(
                particle.pbest_position[d]
                for particle in particles
            )
            / len(particles)
            for d in range(dimensions)
        )

    if mode is MbestMode.WEIGHTED:
        best = min(
            float(particle.pbest_result.fitness)
            for particle in particles
        )

        worst = max(
            float(particle.pbest_result.fitness)
            for particle in particles
        )

        span = worst - best

        # When all fitnesses are equal there is no information with which
        # to distinguish particles. Fall back to the canonical mean.
        if span <= 1e-12:
            return tuple(
                sum(
                    particle.pbest_position[d]
                    for particle in particles
                )
                / len(particles)
                for d in range(dimensions)
            )

        weights = [
            exp(
                -(
                    float(particle.pbest_result.fitness)
                    - best
                )
                / span
            )
            for particle in particles
        ]

        total = sum(weights)

        if (
            total <= 0.0
            or not isfinite(total)
        ):
            return tuple(
                sum(
                    particle.pbest_position[d]
                    for particle in particles
                )
                / len(particles)
                for d in range(dimensions)
            )

        return tuple(
            sum(
                weight * particle.pbest_position[d]
                for weight, particle in zip(
                    weights,
                    particles,
                )
            )
            / total
            for d in range(dimensions)
        )

    raise ValueError(
        f"unsupported Mbest mode: {mode}"
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
    """
    Compute the QPSO contraction-expansion coefficient.

    CANONICAL
        Returns the explicitly configured canonical coefficient.

    ADAPTIVE
        Combines recent optimization progress, coordinate-space diversity,
        route/phenotype diversity, and stagnation into a bounded coefficient.

    All adaptive signals are clipped to [0, 1] before use.

    The function is deterministic: randomness belongs to the optimizer.
    """
    if not isinstance(mode, AlphaMode):
        raise ValueError(
            "mode must be an AlphaMode"
        )

    if not isfinite(float(coordinate_div)):
        raise ValueError(
            "coordinate_div must be finite"
        )

    if not isfinite(float(route_div)):
        raise ValueError(
            "route_div must be finite"
        )

    if not isfinite(float(progress)):
        raise ValueError(
            "progress must be finite"
        )

    if stagnation < 0:
        raise ValueError(
            "stagnation cannot be negative"
        )

    if mode is AlphaMode.CANONICAL:
        alpha = float(
            config.mechanisms.canonical_alpha
        )

        if not isfinite(alpha):
            raise ValueError(
                "canonical alpha must be finite"
            )

        return alpha

    if mode is AlphaMode.ADAPTIVE:
        coordinate_signal = min(
            1.0,
            max(
                0.0,
                float(coordinate_div),
            ),
        )

        route_signal = min(
            1.0,
            max(
                0.0,
                float(route_div),
            ),
        )

        diversity_signal = 0.5 * (
            coordinate_signal
            + route_signal
        )

        patience = max(
            1,
            int(config.stagnation_patience),
        )

        stagnation_signal = min(
            1.0,
            max(
                0.0,
                float(stagnation)
                / patience,
            ),
        )

        progress_signal = min(
            1.0,
            max(
                0.0,
                float(progress),
            ),
        )

        raw_signal = (
            float(config.progress_weight)
            * progress_signal
            + float(config.diversity_weight)
            * diversity_signal
            + float(config.stagnation_weight)
            * (1.0 - stagnation_signal)
        )

        alpha = (
            float(config.alpha_max)
            - raw_signal
            * (
                float(config.alpha_max)
                - float(config.alpha_min)
            )
        )

        # Stagnation deliberately restores some exploratory pressure.
        if stagnation_signal > 0.5:
            alpha += (
                float(config.alpha_max)
                - float(config.alpha_min)
            ) * 0.25 * stagnation_signal

        return min(
            float(config.alpha_max),
            max(
                float(config.alpha_min),
                alpha,
            ),
        )

    raise ValueError(
        f"unsupported alpha mode: {mode}"
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
    phi: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """
    Compute the QPSO local attractor.

    Canonical QPSO
    --------------
    The required canonical formulation is:

        c_pd =
            phi_pd * P_pd
            + (1 - phi_pd) * B_d

    where phi_pd is sampled independently from Uniform(0, 1) by the
    optimizer.

    ``phi`` is therefore supplied by the optimizer rather than sampled here.
    This preserves seeded RNG ownership and reproducibility.

    For backward compatibility with the existing mechanism-level API,
    ``phi=None`` uses phi=0.5 for every coordinate. The actual optimizer must
    explicitly provide stochastic phi when executing canonical QPSO.

    Blended mode
    ------------
    Uses the configured weighted combination:

        c_d =
            w_p P_d
            + w_g B_d
            + w_m m_d

    with the weights normalized to sum to one.

    The canonical branch intentionally ignores ``mbest`` and all attractor
    weights.
    """
    if not isinstance(mode, AttractorMode):
        raise ValueError(
            "mode must be an AttractorMode"
        )

    if dimensions <= 0:
        raise ValueError(
            "dimensions must be positive"
        )

    _validate_dimension(
        name="particle.pbest_position",
        values=particle.pbest_position,
        dimensions=dimensions,
    )

    _validate_dimension(
        name="global_best.pbest_position",
        values=global_best.pbest_position,
        dimensions=dimensions,
    )

    _validate_dimension(
        name="mbest",
        values=mbest,
        dimensions=dimensions,
    )

    _validate_finite_sequence(
        particle.pbest_position,
        name="particle.pbest_position",
    )

    _validate_finite_sequence(
        global_best.pbest_position,
        name="global_best.pbest_position",
    )

    _validate_finite_sequence(
        mbest,
        name="mbest",
    )

    if mode is AttractorMode.CANONICAL:
        if phi is None:
            # Backward-compatible deterministic fallback for direct callers.
            phi_values = (0.5,) * dimensions
        else:
            _validate_dimension(
                name="phi",
                values=phi,
                dimensions=dimensions,
            )

            _validate_finite_sequence(
                phi,
                name="phi",
            )

            phi_values = tuple(
                float(value)
                for value in phi
            )

        for index, value in enumerate(phi_values):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"phi[{index}] must be in [0, 1]"
                )

        return tuple(
            phi_values[d]
            * float(particle.pbest_position[d])
            + (
                1.0
                - phi_values[d]
            )
            * float(global_best.pbest_position[d])
            for d in range(dimensions)
        )

    if mode is AttractorMode.BLENDED:
        weights = (
            float(pbest_weight),
            float(gbest_weight),
            float(mbest_weight),
        )

        if any(
            not isfinite(weight)
            for weight in weights
        ):
            raise ValueError(
                "attractor weights must be finite"
            )

        if any(
            weight < 0.0
            for weight in weights
        ):
            raise ValueError(
                "attractor weights cannot be negative"
            )

        total_weight = sum(weights)

        if total_weight <= 0.0:
            raise ValueError(
                "at least one attractor weight must be positive"
            )

        return tuple(
            (
                weights[0]
                * float(particle.pbest_position[d])
                + weights[1]
                * float(global_best.pbest_position[d])
                + weights[2]
                * float(mbest[d])
            )
            / total_weight
            for d in range(dimensions)
        )

    raise ValueError(
        f"unsupported attractor mode: {mode}"
    )


def should_recover(
    *,
    config: "AdaptiveQPSOConfig",
    stagnation: int,
    rng: random.Random,
) -> bool:
    """
    Determine whether the configured recovery mechanism should fire.

    Recovery is intentionally stochastic and therefore receives the
    optimizer-owned RNG.

    NONE
        Never recover.

    ADAPTIVE_BURST
        Recovery becomes eligible once stagnation reaches the configured
        patience threshold, then fires with the configured probability.
    """
    if not isinstance(
        config.mechanisms.recovery_mode,
        RecoveryMode,
    ):
        raise ValueError(
            "config.mechanisms.recovery_mode must be a RecoveryMode"
        )

    if stagnation < 0:
        raise ValueError(
            "stagnation cannot be negative"
        )

    if not isinstance(rng, random.Random):
        raise ValueError(
            "rng must be an instance of random.Random"
        )

    if config.mechanisms.recovery_mode is RecoveryMode.NONE:
        return False

    if (
        config.mechanisms.recovery_mode
        is RecoveryMode.ADAPTIVE_BURST
    ):
        if stagnation < config.stagnation_patience:
            return False

        return (
            rng.random()
            < config.recovery_probability
        )

    raise ValueError(
        "unsupported recovery mode: "
        f"{config.mechanisms.recovery_mode}"
    )