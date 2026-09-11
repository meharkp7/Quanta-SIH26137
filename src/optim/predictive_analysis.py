from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable, Sequence


class PredictiveAnalysisError(ValueError):
    """Raised when predictive-analysis inputs are invalid."""


@dataclass(frozen=True)
class SearchState:
    """
    Search-state observables recorded at one optimizer iteration.

    All diversity/pressure quantities are normalized to [0, 1].
    best_fitness is minimized by convention.
    """

    iteration: int
    evaluations: int

    best_fitness: float

    coordinate_diversity: float
    decoded_assignment_diversity: float
    decoded_precedence_diversity: float
    decoded_structure_diversity: float

    repaired_assignment_diversity: float
    repaired_precedence_diversity: float
    repaired_structure_diversity: float

    repair_pressure: float
    feasible_rate: float

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise PredictiveAnalysisError("iteration must be non-negative")

        if self.evaluations < 0:
            raise PredictiveAnalysisError("evaluations must be non-negative")

        finite_fields = (
            "best_fitness",
            "coordinate_diversity",
            "decoded_assignment_diversity",
            "decoded_precedence_diversity",
            "decoded_structure_diversity",
            "repaired_assignment_diversity",
            "repaired_precedence_diversity",
            "repaired_structure_diversity",
            "repair_pressure",
            "feasible_rate",
        )

        for field_name in finite_fields:
            value = float(getattr(self, field_name))
            if not isfinite(value):
                raise PredictiveAnalysisError(
                    f"{field_name} must be finite"
                )

        bounded_fields = (
            "coordinate_diversity",
            "decoded_assignment_diversity",
            "decoded_precedence_diversity",
            "decoded_structure_diversity",
            "repaired_assignment_diversity",
            "repaired_precedence_diversity",
            "repaired_structure_diversity",
            "repair_pressure",
            "feasible_rate",
        )

        for field_name in bounded_fields:
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise PredictiveAnalysisError(
                    f"{field_name} must lie in [0, 1], got {value}"
                )


@dataclass(frozen=True)
class HorizonSample:
    """
    One supervised observation for a future-improvement horizon.

    future_improvement is positive when the best objective improves.
    """

    iteration: int
    horizon: int

    coordinate_diversity: float
    decoded_assignment_diversity: float
    decoded_precedence_diversity: float
    decoded_structure_diversity: float

    repaired_assignment_diversity: float
    repaired_precedence_diversity: float
    repaired_structure_diversity: float

    repair_pressure: float
    feasible_rate: float

    best_fitness: float
    future_best_fitness: float
    future_improvement: float

    @property
    def improved(self) -> bool:
        return self.future_improvement > 0.0


@dataclass(frozen=True)
class CorrelationResult:
    """Rank-correlation result between one signal and future improvement."""

    signal: str
    horizon: int
    coefficient: float
    sample_count: int

    @property
    def absolute_coefficient(self) -> float:
        return abs(self.coefficient)


@dataclass(frozen=True)
class PredictiveAnalysisResult:
    """
    Complete correlation analysis across requested horizons.
    """

    horizons: tuple[int, ...]
    samples: tuple[HorizonSample, ...]
    correlations: tuple[CorrelationResult, ...]

    def correlations_for(
        self,
        *,
        horizon: int,
    ) -> tuple[CorrelationResult, ...]:
        return tuple(
            result
            for result in self.correlations
            if result.horizon == horizon
        )

    def strongest_signal(
        self,
        *,
        horizon: int,
    ) -> CorrelationResult | None:
        candidates = self.correlations_for(horizon=horizon)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda result: result.absolute_coefficient,
        )


SIGNALS: tuple[str, ...] = (
    "coordinate_diversity",
    "decoded_assignment_diversity",
    "decoded_precedence_diversity",
    "decoded_structure_diversity",
    "repaired_assignment_diversity",
    "repaired_precedence_diversity",
    "repaired_structure_diversity",
    "repair_pressure",
    "feasible_rate",
)


def build_horizon_samples(
    states: Sequence[SearchState],
    *,
    horizons: Iterable[int] = (1, 5, 10, 20),
) -> tuple[HorizonSample, ...]:
    """
    Construct future-improvement labels without evaluating the optimizer.

    For a state at iteration t and horizon h:

        improvement = best_fitness(t) - best_fitness(t+h)

    Because this is a minimization problem, positive values indicate
    improvement.

    Only states for which t+h exists are included.
    """

    ordered = _validate_and_sort_states(states)
    requested_horizons = _validate_horizons(horizons)

    if not ordered:
        return ()

    samples: list[HorizonSample] = []

    for horizon in requested_horizons:
        for index, state in enumerate(ordered):
            target_index = index + horizon

            if target_index >= len(ordered):
                break

            future = ordered[target_index]

            # The horizon is defined in optimizer iterations, not evaluations.
            # Requiring exact iteration spacing prevents silently assigning a
            # 5-iteration label to irregularly recorded states.
            if future.iteration - state.iteration != horizon:
                continue

            improvement = state.best_fitness - future.best_fitness

            samples.append(
                HorizonSample(
                    iteration=state.iteration,
                    horizon=horizon,
                    coordinate_diversity=state.coordinate_diversity,
                    decoded_assignment_diversity=(
                        state.decoded_assignment_diversity
                    ),
                    decoded_precedence_diversity=(
                        state.decoded_precedence_diversity
                    ),
                    decoded_structure_diversity=(
                        state.decoded_structure_diversity
                    ),
                    repaired_assignment_diversity=(
                        state.repaired_assignment_diversity
                    ),
                    repaired_precedence_diversity=(
                        state.repaired_precedence_diversity
                    ),
                    repaired_structure_diversity=(
                        state.repaired_structure_diversity
                    ),
                    repair_pressure=state.repair_pressure,
                    feasible_rate=state.feasible_rate,
                    best_fitness=state.best_fitness,
                    future_best_fitness=future.best_fitness,
                    future_improvement=improvement,
                )
            )

    return tuple(samples)


def analyze_predictive_relationships(
    states: Sequence[SearchState],
    *,
    horizons: Iterable[int] = (1, 5, 10, 20),
) -> PredictiveAnalysisResult:
    """
    Measure Spearman rank correlation between each search-state signal and
    future improvement.

    No optimizer behavior is changed and no new evaluations are performed.
    """

    requested_horizons = _validate_horizons(horizons)
    samples = build_horizon_samples(
        states,
        horizons=requested_horizons,
    )

    correlations: list[CorrelationResult] = []

    for horizon in requested_horizons:
        horizon_samples = tuple(
            sample
            for sample in samples
            if sample.horizon == horizon
        )

        if len(horizon_samples) < 2:
            continue

        target = tuple(
            sample.future_improvement
            for sample in horizon_samples
        )

        for signal in SIGNALS:
            values = tuple(
                float(getattr(sample, signal))
                for sample in horizon_samples
            )

            coefficient = _spearman(values, target)

            correlations.append(
                CorrelationResult(
                    signal=signal,
                    horizon=horizon,
                    coefficient=coefficient,
                    sample_count=len(horizon_samples),
                )
            )

    return PredictiveAnalysisResult(
        horizons=requested_horizons,
        samples=samples,
        correlations=tuple(correlations),
    )


def _validate_and_sort_states(
    states: Sequence[SearchState],
) -> tuple[SearchState, ...]:
    ordered = tuple(states)

    seen_iterations: set[int] = set()

    for state in ordered:
        if state.iteration in seen_iterations:
            raise PredictiveAnalysisError(
                f"duplicate iteration {state.iteration}"
            )
        seen_iterations.add(state.iteration)

    return tuple(
        sorted(
            ordered,
            key=lambda state: state.iteration,
        )
    )


def _validate_horizons(
    horizons: Iterable[int],
) -> tuple[int, ...]:
    values = tuple(int(value) for value in horizons)

    if not values:
        raise PredictiveAnalysisError(
            "at least one horizon is required"
        )

    if any(value <= 0 for value in values):
        raise PredictiveAnalysisError(
            "horizons must be positive"
        )

    if len(set(values)) != len(values):
        raise PredictiveAnalysisError(
            "horizons must be unique"
        )

    return values


def _rank(values: Sequence[float]) -> tuple[float, ...]:
    """
    Average-rank implementation supporting ties.

    Ranks start at 1.0.
    """

    indexed = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )

    ranks = [0.0] * len(values)

    index = 0
    while index < len(indexed):
        end = index + 1
        value = indexed[index][1]

        while end < len(indexed) and indexed[end][1] == value:
            end += 1

        average_rank = (index + 1 + end) / 2.0

        for position in range(index, end):
            original_index = indexed[position][0]
            ranks[original_index] = average_rank

        index = end

    return tuple(ranks)


def _pearson(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    if len(first) != len(second):
        raise PredictiveAnalysisError(
            "correlation vectors must have equal length"
        )

    if len(first) < 2:
        raise PredictiveAnalysisError(
            "at least two observations are required"
        )

    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)

    numerator = sum(
        (a - mean_first) * (b - mean_second)
        for a, b in zip(first, second)
    )

    first_ss = sum(
        (a - mean_first) ** 2
        for a in first
    )

    second_ss = sum(
        (b - mean_second) ** 2
        for b in second
    )

    denominator = sqrt(first_ss * second_ss)

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def _spearman(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return _pearson(
        _rank(first),
        _rank(second),
    )