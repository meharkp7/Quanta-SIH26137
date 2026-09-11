"""
Observational instrumentation for Step 8 routing optimization.

Step 8B-1
---------

The instrumentation layer consumes already-produced optimization candidates.
It never evaluates, repairs, mutates, or otherwise changes them.

Its primary purpose is to establish whether different representations of the
same population carry different information about future optimization
progress.

The layer is intentionally optimizer-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence

from src.optim.common import FitnessResult
from src.optim.diversity import (
    PopulationDiversity,
    RouteDistance,
    population_diversity,
    population_repair_pressure,
)
from src.routing.route_plan import RoutePlan


class InstrumentationError(ValueError):
    """Raised when an instrumentation snapshot is invalid."""


@dataclass(frozen=True)
class CandidateObservation:
    """
    Immutable observation of one already-evaluated candidate.

    No object held here is modified by the instrumentation layer.
    """

    position: tuple[float, ...]
    decoded_plan: RoutePlan
    repaired_plan: RoutePlan
    fitness: float
    feasible: bool
    repair_distance: float

    def __post_init__(self) -> None:
        if not self.position:
            raise InstrumentationError(
                "candidate position cannot be empty"
            )

        for coordinate in self.position:
            if not isfinite(float(coordinate)):
                raise InstrumentationError(
                    "candidate coordinates must be finite"
                )

        if not isfinite(float(self.fitness)):
            raise InstrumentationError(
                "candidate fitness must be finite"
            )

        if not isfinite(float(self.repair_distance)):
            raise InstrumentationError(
                "repair_distance must be finite"
            )

        if self.repair_distance < 0.0:
            raise InstrumentationError(
                "repair_distance must be non-negative"
            )


@dataclass(frozen=True)
class PopulationObservation:
    """
    Immutable observation of one optimization iteration.

    ``iteration`` is an externally assigned optimizer iteration index.

    ``evaluations`` is the cumulative number of objective evaluations at the
    time this observation was captured.

    ``best_fitness`` and ``mean_fitness`` are computed from the supplied
    candidate observations, not from an independent evaluator call.
    """

    iteration: int
    evaluations: int
    candidates: tuple[CandidateObservation, ...]
    diversity: PopulationDiversity
    repair_pressure: RouteDistance
    best_fitness: float
    mean_fitness: float
    feasible_rate: float
    best_index: int

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise InstrumentationError(
                "iteration must be non-negative"
            )

        if self.evaluations < 0:
            raise InstrumentationError(
                "evaluations must be non-negative"
            )

        if not self.candidates:
            raise InstrumentationError(
                "population observation cannot be empty"
            )

        if not 0 <= self.best_index < len(self.candidates):
            raise InstrumentationError(
                "best_index is outside the candidate population"
            )

        if not isfinite(float(self.best_fitness)):
            raise InstrumentationError(
                "best_fitness must be finite"
            )

        if not isfinite(float(self.mean_fitness)):
            raise InstrumentationError(
                "mean_fitness must be finite"
            )

        if not 0.0 <= self.feasible_rate <= 1.0:
            raise InstrumentationError(
                "feasible_rate must lie in [0, 1]"
            )

        if self.candidates[self.best_index].fitness != self.best_fitness:
            raise InstrumentationError(
                "best_index does not correspond to best_fitness"
            )


@dataclass(frozen=True)
class OptimizationTrace:
    """
    Immutable sequence of population observations.

    Iterations must be strictly increasing.
    Evaluation counts must be non-decreasing.
    """

    observations: tuple[PopulationObservation, ...]

    def __post_init__(self) -> None:
        previous_iteration = -1
        previous_evaluations = -1

        for observation in self.observations:
            if observation.iteration <= previous_iteration:
                raise InstrumentationError(
                    "trace iterations must be strictly increasing"
                )

            if observation.evaluations < previous_evaluations:
                raise InstrumentationError(
                    "trace evaluation counts must be non-decreasing"
                )

            previous_iteration = observation.iteration
            previous_evaluations = observation.evaluations

    @property
    def iterations(self) -> tuple[int, ...]:
        return tuple(
            observation.iteration
            for observation in self.observations
        )

    @property
    def best_fitness(self) -> tuple[float, ...]:
        return tuple(
            observation.best_fitness
            for observation in self.observations
        )

    @property
    def mean_fitness(self) -> tuple[float, ...]:
        return tuple(
            observation.mean_fitness
            for observation in self.observations
        )

    @property
    def coordinate_diversity(self) -> tuple[float, ...]:
        return tuple(
            observation.diversity.genotype_coordinate
            for observation in self.observations
        )

    @property
    def decoded_assignment_diversity(self) -> tuple[float, ...]:
        return tuple(
            observation.diversity.decoded_assignment
            for observation in self.observations
        )

    @property
    def decoded_precedence_diversity(self) -> tuple[float, ...]:
        return tuple(
            observation.diversity.decoded_precedence
            for observation in self.observations
        )

    @property
    def decoded_structure_diversity(self) -> tuple[float, ...]:
        return tuple(
            observation.diversity.decoded_structure
            for observation in self.observations
        )

    @property
    def repaired_assignment_diversity(self) -> tuple[float, ...]:
        return tuple(
            observation.diversity.repaired_assignment
            for observation in self.observations
        )

    @property
    def repaired_precedence_diversity(self) -> tuple[float, ...]:
        return tuple(
            observation.diversity.repaired_precedence
            for observation in self.observations
        )

    @property
    def repaired_structure_diversity(self) -> tuple[float, ...]:
        return tuple(
            observation.diversity.repaired_structure
            for observation in self.observations
        )

    @property
    def repair_assignment_pressure(self) -> tuple[float, ...]:
        return tuple(
            observation.repair_pressure.assignment
            for observation in self.observations
        )

    @property
    def repair_precedence_pressure(self) -> tuple[float, ...]:
        return tuple(
            observation.repair_pressure.precedence
            for observation in self.observations
        )

    @property
    def repair_structure_pressure(self) -> tuple[float, ...]:
        return tuple(
            observation.repair_pressure.structure
            for observation in self.observations
        )

    @property
    def feasible_rate(self) -> tuple[float, ...]:
        return tuple(
            observation.feasible_rate
            for observation in self.observations
        )

def observation_from_candidates(
    *,
    iteration: int,
    evaluations: int,
    candidates: Sequence[CandidateObservation],
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> PopulationObservation:
    """
    Construct one immutable population observation from already-computed
    candidate results.

    No evaluator or repairer is invoked.
    """

    candidate_tuple = tuple(candidates)

    if not candidate_tuple:
        raise InstrumentationError(
            "cannot observe an empty candidate population"
        )

    positions = tuple(
        candidate.position
        for candidate in candidate_tuple
    )

    decoded_plans = tuple(
        candidate.decoded_plan
        for candidate in candidate_tuple
    )

    repaired_plans = tuple(
        candidate.repaired_plan
        for candidate in candidate_tuple
    )

    diversity = population_diversity(
        positions,
        decoded_plans,
        repaired_plans,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    repair_pressure = population_repair_pressure(
        decoded_plans,
        repaired_plans,
    )

    best_index = min(
        range(len(candidate_tuple)),
        key=lambda index: (
            candidate_tuple[index].fitness,
            index,
        ),
    )

    fitness_values = tuple(
        candidate.fitness
        for candidate in candidate_tuple
    )

    feasible_count = sum(
        candidate.feasible
        for candidate in candidate_tuple
    )

    return PopulationObservation(
        iteration=int(iteration),
        evaluations=int(evaluations),
        candidates=candidate_tuple,
        diversity=diversity,
        repair_pressure=repair_pressure,
        best_fitness=fitness_values[best_index],
        mean_fitness=sum(fitness_values) / len(fitness_values),
        feasible_rate=feasible_count / len(candidate_tuple),
        best_index=best_index,
    )


def candidate_observation_from_fitness(
    *,
    position: Sequence[float],
    decoded_plan: RoutePlan,
    repaired_plan: RoutePlan,
    result: FitnessResult,
) -> CandidateObservation:
    """
    Convert an existing optimizer fitness result into an observation.

    This adapter intentionally does not re-run the objective oracle.
    """

    return CandidateObservation(
        position=tuple(float(value) for value in position),
        decoded_plan=decoded_plan,
        repaired_plan=repaired_plan,
        fitness=float(result.fitness),
        feasible=bool(result.feasible),
        repair_distance=float(result.repair_distance),
    )


class OptimizationTraceRecorder:
    """
    Mutable recording facade around immutable observations.

    The recorder itself is deliberately small: it only stores observations
    and never participates in optimization.
    """

    def __init__(
        self,
        *,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
    ) -> None:
        if not isfinite(float(lower_bound)):
            raise InstrumentationError(
                "lower_bound must be finite"
            )

        if not isfinite(float(upper_bound)):
            raise InstrumentationError(
                "upper_bound must be finite"
            )

        if lower_bound >= upper_bound:
            raise InstrumentationError(
                "lower_bound must be strictly smaller than upper_bound"
            )

        self._lower_bound = float(lower_bound)
        self._upper_bound = float(upper_bound)
        self._observations: list[PopulationObservation] = []

    def record(
        self,
        *,
        iteration: int,
        evaluations: int,
        candidates: Sequence[CandidateObservation],
    ) -> PopulationObservation:
        observation = observation_from_candidates(
            iteration=iteration,
            evaluations=evaluations,
            candidates=candidates,
            lower_bound=self._lower_bound,
            upper_bound=self._upper_bound,
        )

        if self._observations:
            previous = self._observations[-1]

            if observation.iteration <= previous.iteration:
                raise InstrumentationError(
                    "recorded iterations must be strictly increasing"
                )

            if observation.evaluations < previous.evaluations:
                raise InstrumentationError(
                    "recorded evaluations must be non-decreasing"
                )

        self._observations.append(observation)

        return observation

    def snapshot(self) -> OptimizationTrace:
        return OptimizationTrace(
            observations=tuple(self._observations)
        )

    def clear(self) -> None:
        """
        Remove recorded observations.

        Clearing instrumentation has no effect on optimization state.
        """

        self._observations.clear()