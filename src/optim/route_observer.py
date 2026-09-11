from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.optim.common import FitnessResult, OptimizationError, Vector
from src.optim.instrumentation import (
    CandidateObservation,
    OptimizationTraceRecorder,
    PopulationObservation,
)


class RouteObservationError(OptimizationError):
    """Raised when routing observation state is inconsistent."""


@dataclass(frozen=True)
class ObservedRouteEvaluation:
    """
    Exact routing artifact produced by one RouteFitnessOracle evaluation.

    The candidate is the exact RouteCandidate already produced by Step 7.
    No second decoding, repair, or evaluation occurs.
    """

    position: Vector
    candidate: object
    fitness_result: FitnessResult

    def __post_init__(self) -> None:
        if not self.position:
            raise RouteObservationError(
                "position cannot be empty"
            )

        for value in self.position:
            if not isinstance(value, (int, float)):
                raise RouteObservationError(
                    "position coordinates must be numeric"
                )

        if self.candidate is None:
            raise RouteObservationError(
                "candidate cannot be None"
            )

        if not isinstance(self.fitness_result, FitnessResult):
            raise RouteObservationError(
                "fitness_result must be a FitnessResult"
            )


class InstrumentedRouteFitnessOracle:
    """
    Observational wrapper around RouteFitnessOracle.

    The wrapped oracle remains the sole authority for:
      * route decoding,
      * feasibility repair,
      * route evaluation,
      * fitness construction.

    This wrapper calls the oracle exactly once and records the exact
    RouteCandidate exposed by that evaluation.
    """

    def __init__(self, oracle) -> None:
        if not callable(oracle):
            raise RouteObservationError(
                "oracle must be callable"
            )

        if not hasattr(oracle, "last_candidate"):
            raise RouteObservationError(
                "oracle must expose last_candidate"
            )

        self.oracle = oracle
        self._observations: list[ObservedRouteEvaluation] = []

    @property
    def calls(self) -> int:
        """Number of evaluations observed through this wrapper."""

        return len(self._observations)

    @property
    def observations(self) -> tuple[ObservedRouteEvaluation, ...]:
        """Immutable snapshot of all observed evaluations."""

        return tuple(self._observations)

    def __call__(self, position: Sequence[float]) -> FitnessResult:
        vector = tuple(float(value) for value in position)

        result = self.oracle(vector)

        candidate = self.oracle.last_candidate

        if candidate is None:
            raise RouteObservationError(
                "oracle did not expose a candidate after evaluation"
            )

        self._observations.append(
            ObservedRouteEvaluation(
                position=vector,
                candidate=candidate,
                fitness_result=result,
            )
        )

        return result

    def clear(self) -> None:
        """Discard observations without changing the wrapped oracle."""

        self._observations.clear()

    def last(self) -> ObservedRouteEvaluation:
        """Return the most recent observation."""

        if not self._observations:
            raise RouteObservationError(
                "no route evaluations have been observed"
            )

        return self._observations[-1]


class RoutePopulationTraceAdapter:
    """
    Converts already-observed route evaluations into generic trace records.

    This adapter is strictly observational. It never invokes the oracle,
    decodes a position, repairs a route, or evaluates a candidate.
    """

    def __init__(
        self,
        recorder: OptimizationTraceRecorder,
    ) -> None:
        self.recorder = recorder

    def record_population(
        self,
        *,
        iteration: int,
        evaluations: int,
        observations: Sequence[ObservedRouteEvaluation],
    ) -> PopulationObservation:
        if not observations:
            raise RouteObservationError(
                "cannot record an empty population"
            )

        candidates = tuple(
            CandidateObservation(
                position=observation.position,
                decoded_plan=observation.candidate.decoded.route_plan,
                repaired_plan=observation.candidate.repaired_plan,
                fitness=observation.fitness_result.fitness,
                feasible=observation.fitness_result.feasible,
                repair_distance=observation.fitness_result.repair_distance,
            )
            for observation in observations
        )

        return self.recorder.record(
            iteration=iteration,
            evaluations=evaluations,
            candidates=candidates,
        )