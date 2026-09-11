from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .common import OptimizationResult
from .instrumentation import (
    OptimizationTrace,
    OptimizationTraceRecorder,
)
from .route_observer import (
    InstrumentedRouteFitnessOracle,
    ObservedRouteEvaluation,
    RoutePopulationTraceAdapter,
)
from .qpso import AdaptiveQPSO, AdaptiveQPSOConfig


class TraceRunnerError(RuntimeError):
    """Raised when a QPSO trace cannot be constructed consistently."""


@dataclass(frozen=True)
class TraceRunResult:
    """
    Complete result of an optimizer run together with its captured
    evaluation-population trace.
    """

    optimization_result: OptimizationResult
    trace: OptimizationTrace
    observations: tuple[ObservedRouteEvaluation, ...]

    def __post_init__(self) -> None:
        if not self.observations:
            raise TraceRunnerError(
                "trace run must contain at least one observed evaluation"
            )

        if not self.trace.observations:
            raise TraceRunnerError(
                "trace run must contain at least one population observation"
            )

        if self.trace.observations[-1].evaluations != (
            self.optimization_result.evaluations
        ):
            raise TraceRunnerError(
                "trace final evaluation count does not match optimizer"
            )


class QPSOTraceRunner:
    """
    Runs AdaptiveQPSO while capturing the exact evaluation stream.

    The optimizer itself is not modified. Every position is evaluated
    exactly once through InstrumentedRouteFitnessOracle, which records
    the exact RouteCandidate returned by the underlying route oracle.

    The resulting evaluation stream is then grouped into population-sized
    batches for search-state analysis.
    """

    def __init__(
        self,
        *,
        config: AdaptiveQPSOConfig,
        oracle,
        initial_population: Sequence[Sequence[float]] | None = None,
        recorder: OptimizationTraceRecorder | None = None,
    ) -> None:
        self.config = config
        self.recorder = recorder or OptimizationTraceRecorder()

        if isinstance(oracle, InstrumentedRouteFitnessOracle):
            self.instrumented_oracle = oracle
        else:
            self.instrumented_oracle = InstrumentedRouteFitnessOracle(oracle)

        self.initial_population = (
            None
            if initial_population is None
            else tuple(tuple(float(value) for value in position)
                       for position in initial_population)
        )

    def run(self) -> TraceRunResult:
        """
        Execute QPSO once and construct its exact population trace.
        """

        self.instrumented_oracle.clear()
        self.recorder.clear()

        optimizer = AdaptiveQPSO(
            config=self.config,
            oracle=self.instrumented_oracle,
            initial_population=self.initial_population,
        )

        optimization_result = optimizer.optimize()

        observations = self.instrumented_oracle.observations

        if len(observations) != optimization_result.evaluations:
            raise TraceRunnerError(
                "captured evaluation count does not match optimizer: "
                f"{len(observations)} != "
                f"{optimization_result.evaluations}"
            )

        if not observations:
            raise TraceRunnerError(
                "optimizer produced no observable evaluations"
            )

        self._record_population_batches(observations)

        trace = self.recorder.snapshot()

        self._validate_trace(
            trace=trace,
            observations=observations,
            optimization_result=optimization_result,
        )

        return TraceRunResult(
            optimization_result=optimization_result,
            trace=trace,
            observations=observations,
        )

    def _record_population_batches(
        self,
        observations: Sequence[ObservedRouteEvaluation],
    ) -> None:
        """
        Convert the sequential evaluation stream into population batches.

        QPSO evaluates the initial population first and then evaluates
        subsequent candidate populations. The optimizer's population size
        therefore provides the natural batching boundary.

        A final incomplete batch is preserved rather than discarded.
        """

        population_size = self.config.population_size

        if population_size <= 0:
            raise TraceRunnerError(
                "population_size must be positive"
            )

        adapter = RoutePopulationTraceAdapter(
            recorder=self.recorder,
        )

        for start in range(0, len(observations), population_size):
            batch = tuple(
                observations[start:start + population_size]
            )

            if not batch:
                continue

            evaluations = start + len(batch)

            adapter.record_population(
                iteration=start // population_size,
                evaluations=evaluations,
                observations=batch,
            )

    @staticmethod
    def _validate_trace(
        *,
        trace: OptimizationTrace,
        observations: Sequence[ObservedRouteEvaluation],
        optimization_result: OptimizationResult,
    ) -> None:
        """
        Validate invariants between the optimizer result and captured trace.

        The optimizer reports the best fitness seen over the entire run.
        A later population is not required to contain that incumbent again.
        Therefore validation checks that the optimizer's best fitness was
        actually observed somewhere in the captured evaluation stream,
        rather than requiring it to appear in the final population.
        """

        if not trace.observations:
            raise TraceRunnerError(
                "trace must contain at least one population"
            )

        if trace.iterations[0] != 0:
            raise TraceRunnerError(
                "trace must begin at iteration 0"
            )

        if len(observations) != optimization_result.evaluations:
            raise TraceRunnerError(
                "captured observations do not match optimizer evaluation count"
            )

        if trace.observations[-1].evaluations != (
            optimization_result.evaluations
        ):
            raise TraceRunnerError(
                "trace final evaluation count does not match optimizer"
            )

        # Every recorded population must contain at least one candidate.
        for observation in trace.observations:
            if not observation.candidates:
                raise TraceRunnerError(
                    "trace contains an empty population observation"
                )

        # Evaluation counts must be cumulative and strictly increasing.
        previous_evaluations = 0

        for observation in trace.observations:
            if observation.evaluations <= previous_evaluations:
                raise TraceRunnerError(
                    "trace evaluation counts must be strictly increasing"
                )

            previous_evaluations = observation.evaluations

        # Iterations must be strictly increasing.
        previous_iteration = -1

        for observation in trace.observations:
            if observation.iteration <= previous_iteration:
                raise TraceRunnerError(
                    "trace iterations must be strictly increasing"
                )

            previous_iteration = observation.iteration

        # The optimizer's global best must correspond to an actual observed
        # evaluation. We intentionally do NOT require it to occur in the
        # final population because QPSO maintains a global incumbent.
        observed_fitnesses = tuple(
            evaluation.fitness_result.fitness
            for evaluation in observations
        )

        if not observed_fitnesses:
            raise TraceRunnerError(
                "trace contains no candidate fitness values"
            )

        best_observed = min(observed_fitnesses)

        if abs(
            best_observed - optimization_result.best_fitness
        ) > 1e-12:
            raise TraceRunnerError(
                "optimizer reported a best fitness that was not observed "
                "in the captured evaluation stream: "
                f"{optimization_result.best_fitness} != {best_observed}"
            )

        # The final trace population must agree with the final observed
        # evaluation count. This is deliberately independent of the global
        # incumbent fitness.
        if trace.observations[-1].evaluations != len(observations):
            raise TraceRunnerError(
                "trace final population does not cover the complete "
                "evaluation stream"
            )


def run_qpso_with_trace(
    config: AdaptiveQPSOConfig,
    oracle,
    *,
    initial_population: Sequence[Sequence[float]] | None = None,
    recorder: OptimizationTraceRecorder | None = None,
) -> TraceRunResult:
    """
    Convenience wrapper for running QPSO with exact trajectory capture.
    """

    return QPSOTraceRunner(
        config=config,
        oracle=oracle,
        initial_population=initial_population,
        recorder=recorder,
    ).run()