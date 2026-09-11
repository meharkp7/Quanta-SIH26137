from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from src.optim.common import (
    FitnessOracle,
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
    """Classical PSO configuration matched to the QPSO experiment."""

    inertia: float = 0.7298
    cognitive: float = 1.49618
    social: float = 1.49618

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.inertia < 0.0:
            raise OptimizationError("inertia must be non-negative")

        if self.cognitive < 0.0:
            raise OptimizationError("cognitive coefficient must be non-negative")

        if self.social < 0.0:
            raise OptimizationError("social coefficient must be non-negative")


@dataclass
class _Particle:
    position: list[float]
    velocity: list[float]
    pbest_position: list[float]
    pbest_fitness: float


class MatchedPSO:
    """
    Standard velocity-based PSO comparator.

    Matching rules:
      * same dimensionality,
      * same bounds,
      * same population size,
      * same seed,
      * same initial population when supplied,
      * same evaluation budget,
      * same objective oracle.
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
                tuple(float(x) for x in row)
                for row in initial_population
            )
        )

        if self._initial_population is not None:
            if len(self._initial_population) != config.population_size:
                raise OptimizationError(
                    "initial_population size must equal population_size"
                )

            for position in self._initial_population:
                if len(position) != config.dimensions:
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

    def optimize(self) -> OptimizationResult:
        positions = self._initial_positions()

        velocity_range = (
            self.config.upper_bound
            - self.config.lower_bound
        )

        particles: list[_Particle] = []
        evaluations = 0

        results = []

        for position in positions:
            result = self.oracle(position)
            evaluations += 1
            results.append(result)

        for position, result in zip(positions, results):
            particles.append(
                _Particle(
                    position=list(position),
                    velocity=[
                        self._rng.uniform(
                            -velocity_range,
                            velocity_range,
                        )
                        * 0.1
                        for _ in range(self.config.dimensions)
                    ],
                    pbest_position=list(position),
                    pbest_fitness=result.fitness,
                )
            )

        global_index = min(
            range(len(particles)),
            key=lambda index: particles[index].pbest_fitness,
        )

        best_position = list(
            particles[global_index].pbest_position
        )
        best_result = results[global_index]

        history_best = [best_result.fitness]
        history_mean = [
            sum(result.fitness for result in results)
            / len(results)
        ]
        history_coordinate = [
            coordinate_diversity(
                [particle.position for particle in particles]
            )
        ]
        history_route = [
            route_diversity(results)
        ]

        iterations = 0

        while evaluations < self.config.max_evaluations:
            iterations += 1

            candidate_positions: list[list[float]] = []

            for particle in particles:
                candidate_velocity = []

                for d in range(self.config.dimensions):
                    r1 = self._rng.random()
                    r2 = self._rng.random()

                    velocity = (
                        self.config.inertia
                        * particle.velocity[d]
                        + self.config.cognitive
                        * r1
                        * (
                            particle.pbest_position[d]
                            - particle.position[d]
                        )
                        + self.config.social
                        * r2
                        * (
                            best_position[d]
                            - particle.position[d]
                        )
                    )

                    velocity = max(
                        -velocity_range,
                        min(velocity_range, velocity),
                    )

                    candidate_velocity.append(velocity)

                candidate = [
                    particle.position[d]
                    + candidate_velocity[d]
                    for d in range(self.config.dimensions)
                ]

                particle.velocity = candidate_velocity

                candidate_positions.append(
                    list(
                        clamp_vector(
                            candidate,
                            self.config.lower_bound,
                            self.config.upper_bound,
                        )
                    )
                )

            remaining = self.config.max_evaluations - evaluations
            current_results = []

            for index in range(
                min(len(candidate_positions), remaining)
            ):
                result = self.oracle(candidate_positions[index])
                evaluations += 1
                current_results.append(result)

                particle = particles[index]
                particle.position = candidate_positions[index]

                if result.fitness < particle.pbest_fitness:
                    particle.pbest_fitness = result.fitness
                    particle.pbest_position = list(
                        candidate_positions[index]
                    )

                if result.fitness < best_result.fitness:
                    best_result = result
                    best_position = list(
                        candidate_positions[index]
                    )

            history_best.append(best_result.fitness)

            history_mean.append(
                sum(
                    particle.pbest_fitness
                    for particle in particles
                )
                / len(particles)
            )

            history_coordinate.append(
                coordinate_diversity(
                    [particle.position for particle in particles]
                )
            )

            # Only current evaluated results are meaningful for phenotype
            # diversity. If a hard budget truncates the iteration, retain
            # the previous observation.
            if len(current_results) == len(particles):
                history_route.append(
                    route_diversity(current_results)
                )
            else:
                history_route.append(history_route[-1])

            if evaluations >= self.config.max_evaluations:
                break

        return OptimizationResult(
            best_position=tuple(best_position),
            best_fitness=best_result.fitness,
            best_result=best_result,
            evaluations=evaluations,
            iterations=iterations,
            history_best=tuple(history_best),
            history_mean=tuple(history_mean),
            history_coordinate_diversity=tuple(
                history_coordinate
            ),
            history_route_diversity=tuple(history_route),
            seed=self.config.seed,
        )