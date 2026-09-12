from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import random
from typing import Callable, Sequence

from src.optim.common import (
    FitnessResult,
    OptimizationConfig,
    OptimizationError,
    OptimizationResult,
    Vector,
    clamp_vector,
    coordinate_diversity,
    fitness_better,
    route_diversity,
)


@dataclass(frozen=True)
class PSOConfig(OptimizationConfig):
    """
    Configuration for the matched classical PSO baseline.

    The baseline uses conventional global-best PSO:

        v(t+1) =
            w(t) v(t)
            + c1 r1 (pbest - x)
            + c2 r2 (gbest - x)

        x(t+1) = x(t) + v(t+1)

    The implementation is deliberately matched to QPSO on the common
    optimization contract: bounds, population size, seed, initial
    population,
    evaluation budget and feasibility semantics.
    """

    inertia_max: float = 0.90
    inertia_min: float = 0.40

    cognitive: float = 1.49618
    social: float = 1.49618

    velocity_fraction: float = 0.20

    def __post_init__(self) -> None:
        super().__post_init__()

        values = (
            self.inertia_max,
            self.inertia_min,
            self.cognitive,
            self.social,
            self.velocity_fraction,
        )

        if any(not isfinite(float(value)) for value in values):
            raise OptimizationError(
                "PSO parameters must be finite"
            )

        if not (
            0.0 <= self.inertia_min <= self.inertia_max
        ):
            raise OptimizationError(
                "inertia bounds must satisfy "
                "0 <= inertia_min <= inertia_max"
            )

        if self.cognitive < 0.0:
            raise OptimizationError(
                "cognitive coefficient must be non-negative"
            )

        if self.social < 0.0:
            raise OptimizationError(
                "social coefficient must be non-negative"
            )

        if not (
            0.0 < self.velocity_fraction <= 1.0
        ):
            raise OptimizationError(
                "velocity_fraction must be in (0, 1]"
            )


@dataclass
class _Particle:
    """Mutable internal PSO particle state."""

    position: list[float]
    velocity: list[float]
    pbest_position: list[float]
    pbest_result: FitnessResult


class ParticleSwarmOptimizer:
    """
    Classical global-best PSO.

    This class intentionally contains no routing-specific behavior.
    Domain-specific decoding, repair and feasibility evaluation belong to
    the supplied oracle.

    The oracle is accepted through a callable contract rather than a strict
    FitnessOracle isinstance check. This permits instrumented/decorated
    oracles to participate in Step 8 experiments without changing optimizer
    semantics.
    """

    def __init__(
        self,
        config: PSOConfig,
        oracle: Callable[
            [Sequence[float]],
            FitnessResult,
        ],
        initial_population: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if not isinstance(config, PSOConfig):
            raise OptimizationError(
                "config must be a PSOConfig"
            )

        if not callable(oracle):
            raise OptimizationError(
                "oracle must be callable"
            )

        self.config = config
        self.oracle = oracle
        self._rng = random.Random(config.seed)

        self._initial_population = (
            None
            if initial_population is None
            else tuple(
                tuple(float(value) for value in row)
                for row in initial_population
            )
        )

        if self._initial_population is not None:
            self._validate_initial_population(
                self._initial_population
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_initial_population(
        self,
        population: Sequence[Sequence[float]],
    ) -> None:
        if len(population) != self.config.population_size:
            raise OptimizationError(
                "initial_population size must equal population_size"
            )

        for index, position in enumerate(population):
            if len(position) != self.config.dimensions:
                raise OptimizationError(
                    "initial_population dimension mismatch "
                    f"at index {index}: expected "
                    f"{self.config.dimensions}, "
                    f"got {len(position)}"
                )

            for dimension, value in enumerate(position):
                if not isfinite(float(value)):
                    raise OptimizationError(
                        "initial_population contains a "
                        f"non-finite value at "
                        f"[{index}][{dimension}]"
                    )

    @staticmethod
    def _validate_fitness_result(
        result: FitnessResult,
    ) -> None:
        if not isinstance(result, FitnessResult):
            raise OptimizationError(
                "fitness oracle must return FitnessResult"
            )

        if not isfinite(float(result.fitness)):
            raise OptimizationError(
                "fitness oracle returned non-finite fitness"
            )

        if not isfinite(float(result.repair_distance)):
            raise OptimizationError(
                "fitness oracle returned non-finite repair distance"
            )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initial_positions(self) -> list[list[float]]:
        if self._initial_population is not None:
            return [
                list(
                    clamp_vector(
                        position,
                        self.config.lower_bound,
                        self.config.upper_bound,
                    )
                )
                for position in self._initial_population
            ]

        return [
            [
                self._rng.uniform(
                    self.config.lower_bound,
                    self.config.upper_bound,
                )
                for _ in range(self.config.dimensions)
            ]
            for _ in range(self.config.population_size)
        ]

    def _velocity_limit(self) -> float:
        return (
            self.config.velocity_fraction
            * (
                self.config.upper_bound
                - self.config.lower_bound
            )
        )

    def _initial_velocity(self) -> list[float]:
        limit = self._velocity_limit()

        return [
            self._rng.uniform(-limit, limit)
            for _ in range(self.config.dimensions)
        ]

    def _initialize_particles(
        self,
    ) -> tuple[list[_Particle], int]:
        positions = self._initial_positions()

        particles: list[_Particle] = []
        evaluations = 0

        for position in positions:
            if evaluations >= self.config.max_evaluations:
                raise OptimizationError(
                    "evaluation budget exhausted during "
                    "initial population evaluation"
                )

            result = self.oracle(position)
            self._validate_fitness_result(result)

            evaluations += 1

            particles.append(
                _Particle(
                    position=list(position),
                    velocity=self._initial_velocity(),
                    pbest_position=list(position),
                    pbest_result=result,
                )
            )

        return particles, evaluations

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _better(
        self,
        candidate: FitnessResult,
        incumbent: FitnessResult,
    ) -> bool:
        """
        Shared feasibility-first comparison.

        A feasible solution dominates an infeasible solution regardless of
        raw penalty/fitness magnitude. Among solutions with equal feasibility,
        lower fitness wins subject to the configured tolerance.
        """
        return fitness_better(
            candidate,
            incumbent,
            tolerance=self.config.tolerance,
        )

    def _best_particle_index(
        self,
        particles: Sequence[_Particle],
    ) -> int:
        if not particles:
            raise OptimizationError(
                "cannot select best particle from empty population"
            )

        best_index = 0

        for index in range(1, len(particles)):
            if self._better(
                particles[index].pbest_result,
                particles[best_index].pbest_result,
            ):
                best_index = index

        return best_index

    # ------------------------------------------------------------------
    # Inertia schedule
    # ------------------------------------------------------------------

    def _inertia(
        self,
        iteration: int,
        max_iterations: int,
    ) -> float:
        """
        Linearly decrease inertia from inertia_max to inertia_min.

        Iteration zero corresponds to the first post-initialization update.
        """
        if max_iterations <= 1:
            return self.config.inertia_min

        ratio = min(
            1.0,
            max(
                0.0,
                iteration / (max_iterations - 1),
            ),
        )

        return (
            self.config.inertia_max
            + ratio
            * (
                self.config.inertia_min
                - self.config.inertia_max
            )
        )

    # ------------------------------------------------------------------
    # Particle update
    # ------------------------------------------------------------------

    def _update_particle(
        self,
        particle: _Particle,
        global_best: _Particle,
        inertia: float,
    ) -> list[float]:
        """
        Generate one conventional PSO candidate.

        r1 and r2 are independently sampled for every coordinate.
        """

        velocity_limit = self._velocity_limit()

        next_velocity: list[float] = []
        next_position: list[float] = []

        for dimension in range(self.config.dimensions):
            r1 = self._rng.random()
            r2 = self._rng.random()

            current = float(
                particle.position[dimension]
            )

            personal_best = float(
                particle.pbest_position[dimension]
            )

            global_best_value = float(
                global_best.pbest_position[dimension]
            )

            velocity = (
                inertia
                * particle.velocity[dimension]
                + self.config.cognitive
                * r1
                * (
                    personal_best
                    - current
                )
                + self.config.social
                * r2
                * (
                    global_best_value
                    - current
                )
            )

            if not isfinite(velocity):
                raise OptimizationError(
                    "PSO generated a non-finite velocity"
                )

            velocity = max(
                -velocity_limit,
                min(
                    velocity_limit,
                    velocity,
                ),
            )

            position = current + velocity

            if not isfinite(position):
                raise OptimizationError(
                    "PSO generated a non-finite position"
                )

            next_velocity.append(velocity)
            next_position.append(position)

        particle.velocity = next_velocity

        return list(
            clamp_vector(
                next_position,
                self.config.lower_bound,
                self.config.upper_bound,
            )
        )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self) -> OptimizationResult:
        """
        Run PSO until the exact configured evaluation budget is consumed.

        The initial population counts toward the budget. If the remaining
        budget is smaller than the population size, only the required prefix
        of the generated population is evaluated.
        """

        particles, evaluations = (
            self._initialize_particles()
        )

        global_best = particles[
            self._best_particle_index(particles)
        ]

        # Detach the global-best state from mutable particle state.
        global_best = _Particle(
            position=list(global_best.position),
            velocity=list(global_best.velocity),
            pbest_position=list(global_best.pbest_position),
            pbest_result=global_best.pbest_result,
        )

        history_best: list[float] = [
            float(
                global_best.pbest_result.fitness
            )
        ]

        history_mean: list[float] = [
            sum(
                float(
                    particle.pbest_result.fitness
                )
                for particle in particles
            )
            / len(particles)
        ]

        history_coordinate: list[float] = [
            coordinate_diversity(
                [
                    particle.position
                    for particle in particles
                ]
            )
        ]

        history_route: list[float] = [
            route_diversity(
                [
                    particle.pbest_result
                    for particle in particles
                ]
            )
        ]

        iterations = 0

        # The number of full iterations is used only to define the inertia
        # schedule. The evaluation loop itself remains strictly budget-based.
        remaining_evaluations = (
            self.config.max_evaluations
            - self.config.population_size
        )

        max_iterations = (
            remaining_evaluations
            + self.config.population_size
            - 1
        ) // self.config.population_size

        while evaluations < self.config.max_evaluations:
            inertia = self._inertia(
                iterations,
                max_iterations,
            )

            candidate_positions = [
                self._update_particle(
                    particle,
                    global_best,
                    inertia,
                )
                for particle in particles
            ]

            remaining = (
                self.config.max_evaluations
                - evaluations
            )

            evaluated_count = min(
                len(candidate_positions),
                remaining,
            )

            for index in range(evaluated_count):
                particle = particles[index]
                candidate_position = (
                    candidate_positions[index]
                )

                result = self.oracle(
                    candidate_position
                )
                self._validate_fitness_result(result)

                evaluations += 1

                particle.position = list(
                    candidate_position
                )

                # ------------------------------------------------------
                # Personal best
                # ------------------------------------------------------
                if self._better(
                    result,
                    particle.pbest_result,
                ):
                    particle.pbest_position = list(
                        candidate_position
                    )
                    particle.pbest_result = result

                # ------------------------------------------------------
                # Global best
                # ------------------------------------------------------
                if self._better(
                    result,
                    global_best.pbest_result,
                ):
                    global_best = _Particle(
                        position=list(
                            candidate_position
                        ),
                        velocity=list(
                            particle.velocity
                        ),
                        pbest_position=list(
                            candidate_position
                        ),
                        pbest_result=result,
                    )

            iterations += 1

            history_best.append(
                float(
                    global_best.pbest_result.fitness
                )
            )

            history_mean.append(
                sum(
                    float(
                        particle.pbest_result.fitness
                    )
                    for particle in particles
                )
                / len(particles)
            )

            history_coordinate.append(
                coordinate_diversity(
                    [
                        particle.position
                        for particle in particles
                    ]
                )
            )

            history_route.append(
                route_diversity(
                    [
                        particle.pbest_result
                        for particle in particles
                    ]
                )
            )

        oracle_calls = getattr(
            self.oracle,
            "calls",
            None,
        )

        if (
            oracle_calls is not None
            and int(oracle_calls) != evaluations
        ):
            raise OptimizationError(
                "optimizer/oracle evaluation accounting mismatch: "
                f"optimizer={evaluations}, "
                f"oracle={oracle_calls}"
            )

        return OptimizationResult(
            best_position=tuple(
                global_best.pbest_position
            ),
            best_fitness=float(
                global_best.pbest_result.fitness
            ),
            best_result=global_best.pbest_result,
            evaluations=evaluations,
            iterations=iterations,
            history_best=tuple(history_best),
            history_mean=tuple(history_mean),
            history_coordinate_diversity=tuple(
                history_coordinate
            ),
            history_route_diversity=tuple(
                history_route
            ),
            seed=self.config.seed,
        )


# ----------------------------------------------------------------------
# Public compatibility API
# ----------------------------------------------------------------------

# Existing Step 8 experiment/test code uses MatchedPSO as the public
# comparator name. Keep the implementation class available as well.
MatchedPSO = ParticleSwarmOptimizer


# Some external code may refer to the descriptive configuration name.
MatchedPSOConfig = PSOConfig


__all__ = [
    "PSOConfig",
    "MatchedPSOConfig",
    "ParticleSwarmOptimizer",
    "MatchedPSO",
]