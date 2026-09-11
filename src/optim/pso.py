from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from src.optim.common import (
    FitnessOracle,
    FitnessResult,
    OptimizationConfig,
    OptimizationError,
    OptimizationResult,
    Vector,
    clamp_vector,
    coordinate_diversity,
    route_diversity,
)


@dataclass(frozen=True)
class MatchedPSOConfig(OptimizationConfig):
    """
    Configuration for the matched classical PSO comparator.

    This is intentionally a conventional global-best PSO:
      v <- w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)

    The configuration inherits the common optimization contract so that
    PSO and QPSO can be compared under the same dimensionality, population
    size, bounds, seed and evaluation budget.
    """

    inertia_max: float = 0.90
    inertia_min: float = 0.40

    cognitive: float = 1.49618
    social: float = 1.49618

    velocity_fraction: float = 0.20

    def __post_init__(self) -> None:
        super().__post_init__()

        if not 0.0 <= self.inertia_min <= self.inertia_max:
            raise OptimizationError(
                "inertia bounds must satisfy 0 <= inertia_min <= inertia_max"
            )

        if self.cognitive < 0.0:
            raise OptimizationError(
                "cognitive coefficient must be non-negative"
            )

        if self.social < 0.0:
            raise OptimizationError(
                "social coefficient must be non-negative"
            )

        if not 0.0 < self.velocity_fraction <= 1.0:
            raise OptimizationError(
                "velocity_fraction must be in (0, 1]"
            )


# Backward-compatible name used by the experiment runner.
PSOConfig = MatchedPSOConfig


@dataclass
class _Particle:
    position: list[float]
    velocity: list[float]
    pbest_position: list[float]
    pbest_result: FitnessResult


class ParticleSwarmOptimizer:
    """
    Classical global-best PSO using the common optimization contract.

    Important comparison properties:
      * same continuous search-space dimensionality as QPSO
      * same bounds
      * same population size
      * same seed
      * same initial population when supplied
      * same evaluation-budget semantics
      * same FitnessOracle interface

    No route-specific repair or heuristic is performed inside PSO.
    Route feasibility remains the responsibility of the shared oracle.
    """

    def __init__(
        self,
        config: MatchedPSOConfig,
        oracle: FitnessOracle,
        initial_population: Sequence[Sequence[float]] | None = None,
    ) -> None:
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
            if len(self._initial_population) != config.population_size:
                raise OptimizationError(
                    "initial_population size must equal population_size"
                )

            if any(
                len(row) != config.dimensions
                for row in self._initial_population
            ):
                raise OptimizationError(
                    "initial_population dimension mismatch"
                )

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

    def _initial_velocity_limit(self) -> float:
        return self.config.velocity_fraction * (
            self.config.upper_bound - self.config.lower_bound
        )

    def _inertia(
        self,
        iteration: int,
        max_iterations: int,
    ) -> float:
        """
        Linearly decrease inertia from inertia_max to inertia_min.
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

        return self.config.inertia_max + ratio * (
            self.config.inertia_min - self.config.inertia_max
        )

    def optimize(self) -> OptimizationResult:
        positions = self._initial_positions()
        velocity_limit = self._initial_velocity_limit()

        particles: list[_Particle] = []
        evaluations = 0

        # ---------------------------------------------------------------
        # Initial population
        # ---------------------------------------------------------------
        for position in positions:
            result = self.oracle(position)
            evaluations += 1

            velocity = [
                self._rng.uniform(
                    -velocity_limit,
                    velocity_limit,
                )
                for _ in range(self.config.dimensions)
            ]

            particles.append(
                _Particle(
                    position=list(position),
                    velocity=velocity,
                    pbest_position=list(position),
                    pbest_result=result,
                )
            )

        global_best = min(
            particles,
            key=lambda particle: particle.pbest_result.fitness,
        )

        history_best = [
            global_best.pbest_result.fitness
        ]

        history_mean = [
            sum(
                particle.pbest_result.fitness
                for particle in particles
            )
            / len(particles)
        ]

        history_coordinate = [
            coordinate_diversity(
                [particle.position for particle in particles]
            )
        ]

        history_route = [
            route_diversity(
                [particle.pbest_result for particle in particles]
            )
        ]

        iterations = 0

        # Number of population-sized update rounds required to exhaust
        # the remaining evaluation budget.
        max_iterations = max(
            1,
            (
                self.config.max_evaluations
                - self.config.population_size
                + self.config.population_size
                - 1
            )
            // self.config.population_size,
        )

        # ---------------------------------------------------------------
        # Main PSO loop
        # ---------------------------------------------------------------
        while evaluations < self.config.max_evaluations:
            inertia = self._inertia(
                iterations,
                max_iterations,
            )

            iterations += 1

            remaining = (
                self.config.max_evaluations - evaluations
            )

            # Only evaluate as many particles as the remaining budget
            # allows. This guarantees an exact evaluation-budget contract.
            active_particles = particles[:remaining]

            for particle in active_particles:
                for dimension in range(self.config.dimensions):
                    r1 = self._rng.random()
                    r2 = self._rng.random()

                    particle.velocity[dimension] = (
                        inertia * particle.velocity[dimension]
                        + self.config.cognitive
                        * r1
                        * (
                            particle.pbest_position[dimension]
                            - particle.position[dimension]
                        )
                        + self.config.social
                        * r2
                        * (
                            global_best.pbest_position[dimension]
                            - particle.position[dimension]
                        )
                    )

                    particle.velocity[dimension] = min(
                        velocity_limit,
                        max(
                            -velocity_limit,
                            particle.velocity[dimension],
                        ),
                    )

                    particle.position[dimension] += (
                        particle.velocity[dimension]
                    )

                particle.position = list(
                    clamp_vector(
                        particle.position,
                        self.config.lower_bound,
                        self.config.upper_bound,
                    )
                )

                result = self.oracle(particle.position)
                evaluations += 1

                # -------------------------------------------------------
                # Personal-best update
                # -------------------------------------------------------
                if (
                    result.fitness
                    < particle.pbest_result.fitness
                    - self.config.tolerance
                ):
                    particle.pbest_position = list(
                        particle.position
                    )
                    particle.pbest_result = result

                    # ---------------------------------------------------
                    # Global-best update
                    # ---------------------------------------------------
                    if (
                        result.fitness
                        < global_best.pbest_result.fitness
                        - self.config.tolerance
                    ):
                        global_best = particle

            history_best.append(
                global_best.pbest_result.fitness
            )

            history_mean.append(
                sum(
                    particle.pbest_result.fitness
                    for particle in particles
                )
                / len(particles)
            )

            history_coordinate.append(
                coordinate_diversity(
                    [particle.position for particle in particles]
                )
            )

            history_route.append(
                route_diversity(
                    [particle.pbest_result for particle in particles]
                )
            )

        return OptimizationResult(
            best_position=tuple(
                global_best.pbest_position
            ),
            best_fitness=global_best.pbest_result.fitness,
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


# Public comparator name used throughout Step 8.
MatchedPSO = ParticleSwarmOptimizer


__all__ = [
    "MatchedPSOConfig",
    "PSOConfig",
    "ParticleSwarmOptimizer",
    "MatchedPSO",
]