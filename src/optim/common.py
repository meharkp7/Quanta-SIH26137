from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Sequence


Vector = tuple[float, ...]


class OptimizationError(ValueError):
    """Raised when an optimizer receives an invalid configuration or state."""


@dataclass(frozen=True)
class FitnessResult:
    """
    Result returned by a fitness oracle.

    Lower fitness is better among solutions with the same feasibility status.

    Optimization selection is feasibility-first:
        feasible > infeasible
    and only then:
        lower fitness > higher fitness

    The optimizer therefore never prefers an infeasible solution merely
    because its numerical penalty is smaller than the objective of a feasible
    solution.
    """

    fitness: float
    feasible: bool = True
    route_signature: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    repair_distance: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(float(self.fitness)):
            raise OptimizationError("fitness must be finite")

        if not isfinite(float(self.repair_distance)):
            raise OptimizationError("repair_distance must be finite")

        if self.repair_distance < 0.0:
            raise OptimizationError("repair_distance must be non-negative")


def fitness_better(
    candidate: FitnessResult,
    incumbent: FitnessResult,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """
    Determine whether ``candidate`` should replace ``incumbent``.

    Step-8 optimization uses lexicographic feasibility-first dominance:

        1. A feasible solution always dominates an infeasible solution.
        2. If both are feasible, lower objective fitness wins.
        3. If both are infeasible, lower penalized fitness wins.
        4. Differences within ``tolerance`` are treated as ties.

    This comparison must be shared by QPSO and the matched PSO comparator so
    that their search semantics remain identical apart from the update rule.

    Parameters
    ----------
    candidate:
        Candidate optimization result.

    incumbent:
        Current result against which the candidate is compared.

    tolerance:
        Numerical tolerance used when comparing equal-feasibility solutions.

    Returns
    -------
    bool
        ``True`` when the candidate should replace the incumbent.
    """
    if tolerance < 0.0:
        raise OptimizationError("tolerance must be non-negative")

    # Feasibility dominates numerical fitness.
    if candidate.feasible != incumbent.feasible:
        return candidate.feasible

    # Same feasibility state: lower fitness is better.
    return candidate.fitness < incumbent.fitness - tolerance


def best_result_index(
    results: Sequence[FitnessResult],
    *,
    tolerance: float = 1e-12,
) -> int:
    """
    Return the index of the best result under feasibility-first ordering.

    This deliberately avoids Python's plain ``min(..., key=fitness)`` because
    numerical penalty values must never allow an infeasible result to outrank
    a feasible one.
    """
    if not results:
        raise OptimizationError("results cannot be empty")

    best_index = 0

    for index in range(1, len(results)):
        if fitness_better(
            results[index],
            results[best_index],
            tolerance=tolerance,
        ):
            best_index = index

    return best_index


@dataclass(frozen=True)
class OptimizationConfig:
    """Shared optimizer configuration."""

    dimensions: int
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    population_size: int = 20
    max_evaluations: int = 1000
    seed: int = 0
    tolerance: float = 1e-12
    stagnation_patience: int = 25

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise OptimizationError("dimensions must be positive")

        if not (
            isfinite(float(self.lower_bound))
            and isfinite(float(self.upper_bound))
        ):
            raise OptimizationError("bounds must be finite")

        if self.lower_bound >= self.upper_bound:
            raise OptimizationError(
                "lower_bound must be strictly smaller than upper_bound"
            )

        if self.population_size < 2:
            raise OptimizationError("population_size must be at least 2")

        if self.max_evaluations < self.population_size:
            raise OptimizationError(
                "max_evaluations must be >= population_size"
            )

        if self.tolerance < 0.0:
            raise OptimizationError("tolerance must be non-negative")

        if self.stagnation_patience < 1:
            raise OptimizationError("stagnation_patience must be positive")


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable summary of an optimization run."""

    best_position: Vector
    best_fitness: float
    best_result: FitnessResult
    evaluations: int
    iterations: int
    history_best: tuple[float, ...]
    history_mean: tuple[float, ...]
    history_coordinate_diversity: tuple[float, ...]
    history_route_diversity: tuple[float, ...]
    seed: int

    def __post_init__(self) -> None:
        if not self.best_position:
            raise OptimizationError("best_position cannot be empty")

        if not isfinite(float(self.best_fitness)):
            raise OptimizationError("best_fitness must be finite")

        if self.best_fitness != self.best_result.fitness:
            raise OptimizationError(
                "best_fitness must match best_result.fitness"
            )

        if len(self.history_best) != len(self.history_mean):
            raise OptimizationError("history lengths must match")

        if len(self.history_best) != len(
            self.history_coordinate_diversity
        ):
            raise OptimizationError("coordinate diversity history mismatch")

        if len(self.history_best) != len(self.history_route_diversity):
            raise OptimizationError("route diversity history mismatch")

        if self.evaluations < 0:
            raise OptimizationError("evaluations cannot be negative")

        if self.iterations < 0:
            raise OptimizationError("iterations cannot be negative")


class FitnessOracle:
    """
    Callable fitness boundary.

    The optimizer knows nothing about routing. The oracle translates a
    continuous candidate into a domain-specific result.

    The oracle itself does not decide which candidate is better. Selection
    semantics belong to ``fitness_better`` so that every optimizer and
    experiment uses the same feasibility-first rule.
    """

    def __init__(
        self,
        evaluate: Callable[[Vector], FitnessResult],
    ) -> None:
        self._evaluate = evaluate
        self.calls = 0

    def __call__(self, position: Sequence[float]) -> FitnessResult:
        vector = tuple(float(x) for x in position)

        result = self._evaluate(vector)

        if not isinstance(result, FitnessResult):
            raise OptimizationError(
                "fitness oracle must return FitnessResult"
            )

        self.calls += 1
        return result


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a scalar into a closed interval."""

    if not isfinite(value):
        raise OptimizationError("cannot clamp a non-finite value")

    if not (
        isfinite(float(lower))
        and isfinite(float(upper))
        and lower <= upper
    ):
        raise OptimizationError("invalid clamp bounds")

    return min(max(value, lower), upper)


def clamp_vector(
    position: Sequence[float],
    lower: float,
    upper: float,
) -> Vector:
    """Clamp every coordinate into the configured search domain."""

    return tuple(clamp(float(x), lower, upper) for x in position)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise OptimizationError("vectors must have equal dimensions")

    total = 0.0
    for x, y in zip(a, b):
        delta = float(x) - float(y)
        total += delta * delta

    return total**0.5


def coordinate_diversity(
    population: Sequence[Sequence[float]],
) -> float:
    """
    Normalized coordinate-space diversity.

    This is intentionally generic. Routing-specific phenotype diversity is
    supplied independently by route signatures.
    """

    if not population:
        return 0.0

    dimensions = len(population[0])
    if dimensions == 0:
        return 0.0

    if any(len(vector) != dimensions for vector in population):
        raise OptimizationError("population vectors have inconsistent sizes")

    mean = tuple(
        sum(float(vector[d]) for vector in population) / len(population)
        for d in range(dimensions)
    )

    average_distance = sum(
        euclidean_distance(vector, mean) for vector in population
    ) / len(population)

    # For [0, 1]^d the maximum useful scale is sqrt(d)/2.
    normalization = max(dimensions**0.5 / 2.0, 1e-12)

    return min(1.0, average_distance / normalization)


def route_diversity(
    results: Sequence[FitnessResult],
) -> float:
    """
    Fraction of pairwise route-signature disagreement.

    A missing route signature is treated as one unique phenotype. This keeps
    the optimizer usable with generic or partially instrumented oracles.
    """

    if len(results) < 2:
        return 0.0

    signatures = [
        result.route_signature
        if result.route_signature is not None
        else (("__missing__",),)
        for result in results
    ]

    pairs = 0
    disagreements = 0

    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            pairs += 1
            if signatures[i] != signatures[j]:
                disagreements += 1

    return disagreements / pairs if pairs else 0.0