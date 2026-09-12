from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any, Sequence

from .common import FitnessResult
from .instrumentation import (
    CandidateObservation,
    OptimizationTraceRecorder,
)


class RouteObservationError(RuntimeError):
    """Raised when route-observation invariants are violated."""


@dataclass(frozen=True)
class ObservedRouteEvaluation:
    """
    Immutable record of one exact route-fitness evaluation.

    The first three fields are the original Step-8 contract. Timing fields
    are optional telemetry and therefore have backward-compatible defaults.
    """

    position: tuple[float, ...]
    candidate: Any
    fitness_result: FitnessResult

    evaluation_time_s: float = 0.0
    repair_time_s: float = 0.0
    elapsed_s: float = 0.0
    evaluation_index: int = 0

    def __post_init__(self) -> None:
        position = tuple(
            float(value)
            for value in self.position
        )

        for index, value in enumerate(position):
            if not isfinite(value):
                raise RouteObservationError(
                    "observation position contains "
                    f"non-finite value at index {index}"
                )

        if not isinstance(
            self.fitness_result,
            FitnessResult,
        ):
            raise RouteObservationError(
                "fitness_result must be a FitnessResult"
            )

        for name, value in (
            ("evaluation_time_s", self.evaluation_time_s),
            ("repair_time_s", self.repair_time_s),
            ("elapsed_s", self.elapsed_s),
        ):
            numeric = float(value)

            if not isfinite(numeric):
                raise RouteObservationError(
                    f"{name} must be finite"
                )

            if numeric < 0.0:
                raise RouteObservationError(
                    f"{name} must be non-negative"
                )

        if int(self.evaluation_index) != self.evaluation_index:
            raise RouteObservationError(
                "evaluation_index must be an integer"
            )

        if self.evaluation_index < 0:
            raise RouteObservationError(
                "evaluation_index must be non-negative"
            )

        object.__setattr__(
            self,
            "position",
            position,
        )
        object.__setattr__(
            self,
            "evaluation_time_s",
            float(self.evaluation_time_s),
        )
        object.__setattr__(
            self,
            "repair_time_s",
            float(self.repair_time_s),
        )
        object.__setattr__(
            self,
            "elapsed_s",
            float(self.elapsed_s),
        )
        object.__setattr__(
            self,
            "evaluation_index",
            int(self.evaluation_index),
        )

    @property
    def feasible(self) -> bool:
        """Whether the evaluated route was feasible."""

        return bool(
            self.fitness_result.feasible
        )

    @property
    def fitness(self) -> float:
        """Numerical fitness returned by the underlying oracle."""

        return float(
            self.fitness_result.fitness
        )

    @property
    def repair_distance(self) -> float:
        """Repair displacement returned by the underlying oracle."""

        return float(
            self.fitness_result.repair_distance
        )


class InstrumentedRouteFitnessOracle:
    """
    Observational wrapper around an existing route fitness oracle.

    Guarantees:

    * the wrapped oracle is called exactly once per __call__;
    * its exact FitnessResult is returned unchanged;
    * the exact candidate exposed by the oracle is recorded;
    * no re-evaluation or route reconstruction occurs;
    * timing is observational only.
    """

    def __init__(
        self,
        oracle,
    ) -> None:
        if not callable(oracle):
            raise RouteObservationError(
                "oracle must be callable"
            )

        self.oracle = oracle
        self._observations: list[
            ObservedRouteEvaluation
        ] = []
        self._started_at: float | None = None
        self._evaluation_count = 0

    @property
    def observations(
        self,
    ) -> tuple[ObservedRouteEvaluation, ...]:
        """Immutable snapshot of all observations."""

        return tuple(
            self._observations
        )

    @property
    def calls(self) -> int:
        """Number of underlying oracle calls."""

        return self._evaluation_count

    @property
    def last_candidate(self):
        """Most recently observed candidate."""

        observation = self.last()

        if observation is None:
            return None

        return observation.candidate

    def last(
        self,
    ) -> ObservedRouteEvaluation | None:
        """
        Return the most recent observation.

        This preserves the original Step-8 observer API used by the tests and
        existing trace code.
        """

        if not self._observations:
            return None

        return self._observations[-1]

    @property
    def total_elapsed_s(self) -> float:
        """Wall-clock time elapsed since the first evaluation."""

        if self._started_at is None:
            return 0.0

        return max(
            0.0,
            perf_counter() - self._started_at,
        )

    def clear(self) -> None:
        """
        Clear observations and observer timing state.

        The wrapped oracle itself is never modified.
        """

        self._observations.clear()
        self._started_at = None
        self._evaluation_count = 0

    def __call__(
        self,
        position: Sequence[float],
    ) -> FitnessResult:
        vector = tuple(
            float(value)
            for value in position
        )

        for index, value in enumerate(vector):
            if not isfinite(value):
                raise RouteObservationError(
                    "position contains non-finite value "
                    f"at index {index}"
                )

        if self._started_at is None:
            self._started_at = perf_counter()

        evaluation_started = perf_counter()

        # EXACTLY ONE underlying oracle call.
        result = self.oracle(vector)

        evaluation_time_s = (
            perf_counter()
            - evaluation_started
        )

        if not isinstance(
            result,
            FitnessResult,
        ):
            raise RouteObservationError(
                "wrapped oracle must return a FitnessResult"
            )

        candidate = getattr(
            self.oracle,
            "last_candidate",
            None,
        )

        if candidate is None:
            raise RouteObservationError(
                "wrapped route oracle did not expose "
                "last_candidate after evaluation"
            )

        repair_time_s = _candidate_repair_time(
            candidate
        )

        self._evaluation_count += 1

        elapsed_s = max(
            0.0,
            perf_counter() - self._started_at,
        )

        observation = ObservedRouteEvaluation(
            position=vector,
            candidate=candidate,
            fitness_result=result,
            evaluation_time_s=evaluation_time_s,
            repair_time_s=repair_time_s,
            elapsed_s=elapsed_s,
            evaluation_index=self._evaluation_count,
        )

        self._observations.append(
            observation
        )

        return result


def _candidate_repair_distance(
    candidate: object,
) -> float:
    """
    Return a candidate's repair displacement when directly exposed.

    This function is observational only. The authoritative optimizer fitness
    remains FitnessResult.repair_distance.
    """

    direct = getattr(
        candidate,
        "repair_distance",
        None,
    )

    if direct is not None:
        distance = float(direct)

        if not isfinite(distance):
            raise RouteObservationError(
                "candidate repair_distance must be finite"
            )

        if distance < 0.0:
            raise RouteObservationError(
                "candidate repair_distance must be non-negative"
            )

        return distance

    repair_result = getattr(
        candidate,
        "repair_result",
        None,
    )

    if repair_result is None:
        return 0.0

    direct = getattr(
        repair_result,
        "repair_distance",
        None,
    )

    if direct is None:
        return 0.0

    distance = float(direct)

    if not isfinite(distance):
        raise RouteObservationError(
            "repair_result repair_distance must be finite"
        )

    if distance < 0.0:
        raise RouteObservationError(
            "repair_result repair_distance must be "
            "non-negative"
        )

    return distance


def _candidate_repair_time(
    candidate: object,
) -> float:
    """
    Return exact Step-7 repair elapsed time when available.

    Legacy/minimal test fixtures may contain a repair_result without timing.
    In that case the absence of telemetry is represented as zero rather than
    breaking the original observer contract.
    """

    repair_result = getattr(
        candidate,
        "repair_result",
        None,
    )

    if repair_result is None:
        return 0.0

    elapsed = getattr(
        repair_result,
        "elapsed_time_s",
        None,
    )

    if elapsed is None:
        return 0.0

    elapsed = float(elapsed)

    if not isfinite(elapsed):
        raise RouteObservationError(
            "candidate repair elapsed_time_s must be finite"
        )

    if elapsed < 0.0:
        raise RouteObservationError(
            "candidate repair elapsed_time_s must be "
            "non-negative"
        )

    return elapsed


class RoutePopulationTraceAdapter:
    """
    Convert route observations into the existing generic trace contract.

    This adapter never evaluates an oracle and never mutates candidates.
    """

    def __init__(
        self,
        recorder: OptimizationTraceRecorder,
    ) -> None:
        if not isinstance(
            recorder,
            OptimizationTraceRecorder,
        ):
            raise RouteObservationError(
                "recorder must be an OptimizationTraceRecorder"
            )

        self.recorder = recorder

    def record_population(
        self,
        *,
        iteration: int,
        evaluations: int,
        observations: Sequence[
            ObservedRouteEvaluation
        ],
    ):
        if iteration < 0:
            raise RouteObservationError(
                "iteration must be non-negative"
            )

        if evaluations <= 0:
            raise RouteObservationError(
                "evaluations must be positive"
            )

        if not observations:
            raise RouteObservationError(
                "observations cannot be empty"
            )

        candidates: list[
            CandidateObservation
        ] = []

        for observation in observations:
            if not isinstance(
                observation,
                ObservedRouteEvaluation,
            ):
                raise RouteObservationError(
                    "population contains an invalid "
                    "ObservedRouteEvaluation"
                )

            candidates.append(
                self._to_candidate_observation(
                    observation
                )
            )

        return self.recorder.record(
            iteration=iteration,
            evaluations=evaluations,
            candidates=tuple(candidates),
        )

    @staticmethod
    def _to_candidate_observation(
        observation: ObservedRouteEvaluation,
    ) -> CandidateObservation:
        candidate = observation.candidate

        decoded_plan = getattr(
            getattr(
                candidate,
                "decoded",
                None,
            ),
            "route_plan",
            None,
        )

        repaired_plan = getattr(
            candidate,
            "repaired_plan",
            None,
        )

        if decoded_plan is None:
            raise RouteObservationError(
                "candidate does not expose "
                "decoded.route_plan"
            )

        if repaired_plan is None:
            raise RouteObservationError(
                "candidate does not expose repaired_plan"
            )

        # IMPORTANT:
        # CandidateObservation's established contract is:
        #
        #   position
        #   decoded_plan
        #   repaired_plan
        #   fitness
        #   feasible
        #   repair_distance
        #
        # Do not pass FitnessResult itself or observer-only timing fields into
        # the generic instrumentation layer.
        return CandidateObservation(
            position=tuple(
                observation.position
            ),
            decoded_plan=decoded_plan,
            repaired_plan=repaired_plan,
            fitness=float(
                observation.fitness_result.fitness
            ),
            feasible=bool(
                observation.fitness_result.feasible
            ),
            repair_distance=float(
                observation.fitness_result.repair_distance
            ),
        )


__all__ = [
    "InstrumentedRouteFitnessOracle",
    "ObservedRouteEvaluation",
    "RouteObservationError",
    "RoutePopulationTraceAdapter",
]