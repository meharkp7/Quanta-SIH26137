from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from src.optim.common import Vector
from src.optim.qpso import RouteFitnessOracle


class PredictiveInitializationError(ValueError):
    """Raised when a controlled initial population cannot be constructed."""


@dataclass(frozen=True)
class InitialPopulationResult:
    population: tuple[Vector, ...]
    fitnesses: tuple[float, ...]
    attempts: int

    @property
    def best_fitness(self) -> float:
        return min(self.fitnesses)

    @property
    def worst_fitness(self) -> float:
        return max(self.fitnesses)


def build_nonoptimal_population(
    *,
    oracle: RouteFitnessOracle,
    dimensions: int,
    population_size: int,
    lower_bound: float,
    upper_bound: float,
    seed: int,
    required_min_fitness: float,
    tolerance: float = 1e-12,
    max_attempts: int = 100_000,
) -> InitialPopulationResult:
    """
    Construct a deterministic initial population whose members are all
    strictly worse than the supplied reference fitness.

    This changes only experimental initialization.

    It does NOT:
      - modify the objective;
      - modify the evaluator;
      - modify repair;
      - modify QPSO;
      - modify fitness values.

    The same seed produces the same sampled population.
    """

    if dimensions <= 0:
        raise PredictiveInitializationError(
            "dimensions must be positive."
        )

    if population_size <= 0:
        raise PredictiveInitializationError(
            "population_size must be positive."
        )

    if lower_bound >= upper_bound:
        raise PredictiveInitializationError(
            "lower_bound must be smaller than upper_bound."
        )

    if max_attempts <= 0:
        raise PredictiveInitializationError(
            "max_attempts must be positive."
        )

    rng = random.Random(seed)

    retained: list[Vector] = []
    fitnesses: list[float] = []

    attempts = 0

    while len(retained) < population_size:

        if attempts >= max_attempts:
            raise PredictiveInitializationError(
                "Could not construct a fully non-optimal population "
                f"after {max_attempts} attempts. "
                f"Collected {len(retained)}/{population_size} particles."
            )

        attempts += 1

        position = tuple(
            rng.uniform(
                lower_bound,
                upper_bound,
            )
            for _ in range(dimensions)
        )

        result = oracle(position)

        if result.fitness > required_min_fitness + tolerance:

            retained.append(position)
            fitnesses.append(float(result.fitness))

    return InitialPopulationResult(
        population=tuple(retained),
        fitnesses=tuple(fitnesses),
        attempts=attempts,
    )