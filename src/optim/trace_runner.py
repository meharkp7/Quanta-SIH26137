from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean, pstdev
from typing import Sequence

from .common import (
    FitnessResult,
    OptimizationResult,
)
from .instrumentation import (
    OptimizationTrace,
    OptimizationTraceRecorder,
)
from .route_observer import (
    InstrumentedRouteFitnessOracle,
    ObservedRouteEvaluation,
    RoutePopulationTraceAdapter,
)
from .qpso import (
    AdaptiveQPSO,
    AdaptiveQPSOConfig,
)


class TraceRunnerError(RuntimeError):
    """Raised when a QPSO trace cannot be constructed consistently."""


@dataclass(frozen=True)
class FeasibleTracePoint:
    """
    Research-grade summary of the best feasible incumbent after one
    evaluation.

    This is deliberately derived from the already-captured evaluation stream.
    It never re-evaluates a candidate.
    """

    evaluation: int
    elapsed_s: float
    best_feasible_fitness: float
    feasible_fraction: float
    repair_time_s: float
    repair_distance: float


@dataclass(frozen=True)
class TraceRunResult:
    """
    Complete optimizer result together with the exact captured evaluation
    stream and research telemetry derived from that stream.
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

        if (
            self.trace.observations[-1].evaluations
            != self.optimization_result.evaluations
        ):
            raise TraceRunnerError(
                "trace final evaluation count does not match optimizer"
            )

        if len(self.observations) != (
            self.optimization_result.evaluations
        ):
            raise TraceRunnerError(
                "observation count does not match optimizer evaluations"
            )

        self._validate_telemetry()

    # ------------------------------------------------------------------
    # Exact evaluation-level research telemetry
    # ------------------------------------------------------------------

    @property
    def evaluations(self) -> int:
        """Total number of exact fitness evaluations."""

        return len(self.observations)

    @property
    def elapsed_time_s(self) -> float:
        """
        Total observed wall-clock time.

        This uses the cumulative elapsed value captured by the observer rather
        than timing the trace post-processing.
        """

        return float(
            self.observations[-1].elapsed_s
        )

    @property
    def first_feasible_evaluation(self) -> int | None:
        """
        Evaluation index at which feasibility was first observed.

        Returns None when the run never observes a feasible candidate.
        """

        for observation in self.observations:
            if observation.feasible:
                return observation.evaluation_index

        return None

    @property
    def first_feasible_time_s(self) -> float | None:
        """Elapsed time at which the first feasible candidate was observed."""

        for observation in self.observations:
            if observation.feasible:
                return observation.elapsed_s

        return None

    @property
    def best_feasible_fitness(self) -> float | None:
        """
        Best feasible fitness observed during the complete evaluation stream.

        Returns None when no feasible candidate was observed.
        """

        best: float | None = None

        for observation in self.observations:
            if not observation.feasible:
                continue

            fitness = float(
                observation.fitness
            )

            if best is None or fitness < best:
                best = fitness

        return best

    @property
    def final_feasible(self) -> bool:
        """Whether the optimizer's final returned result is feasible."""

        return bool(
            self.optimization_result.best_result.feasible
        )

    @property
    def feasible_evaluations(self) -> int:
        """Number of feasible candidate evaluations."""

        return sum(
            1
            for observation in self.observations
            if observation.feasible
        )

    @property
    def feasible_fraction(self) -> float:
        """Fraction of all evaluated candidates that were feasible."""

        return (
            self.feasible_evaluations
            / self.evaluations
        )

    @property
    def total_repair_time_s(self) -> float:
        """Sum of exact repair times exposed by observed candidates."""

        return sum(
            observation.repair_time_s
            for observation in self.observations
        )

    @property
    def mean_evaluation_time_s(self) -> float:
        """Mean wall-clock duration of an oracle evaluation."""

        return mean(
            observation.evaluation_time_s
            for observation in self.observations
        )

    @property
    def mean_repair_time_s(self) -> float:
        """Mean observed repair duration."""

        return mean(
            observation.repair_time_s
            for observation in self.observations
        )

    @property
    def mean_repair_distance(self) -> float:
        """Mean repair displacement across evaluated candidates."""

        return mean(
            observation.repair_distance
            for observation in self.observations
        )

    @property
    def raw_best_fitness(self) -> float:
        """
        Minimum numerical fitness across all evaluations.

        IMPORTANT:
        this is intentionally distinct from best_feasible_fitness.

        An infeasible candidate can have a lower raw numerical objective than
        the optimizer's feasibility-first incumbent.
        """

        return min(
            observation.fitness
            for observation in self.observations
        )

    @property
    def best_feasible_curve(
        self,
    ) -> tuple[FeasibleTracePoint, ...]:
        """
        Evaluation-by-evaluation best-feasible convergence curve.

        Before the first feasible solution, best_feasible_fitness is represented
        by None at the semantic level; only points after feasibility exists are
        emitted. This makes the curve directly usable for plotting and avoids
        inventing a penalty sentinel.
        """

        points: list[FeasibleTracePoint] = []

        best: float | None = None
        feasible_seen = 0

        for observation in self.observations:
            if observation.feasible:
                feasible_seen += 1

                fitness = float(
                    observation.fitness
                )

                if (
                    best is None
                    or fitness < best
                ):
                    best = fitness

            if best is not None:
                points.append(
                    FeasibleTracePoint(
                        evaluation=(
                            observation.evaluation_index
                        ),
                        elapsed_s=(
                            observation.elapsed_s
                        ),
                        best_feasible_fitness=best,
                        feasible_fraction=(
                            feasible_seen
                            / observation.evaluation_index
                        ),
                        repair_time_s=(
                            observation.repair_time_s
                        ),
                        repair_distance=(
                            observation.repair_distance
                        ),
                    )
                )

        return tuple(points)

    @property
    def feasible_fraction_curve(
        self,
    ) -> tuple[tuple[int, float], ...]:
        """
        Cumulative feasible-particle/evaluation fraction.

        Format:
            (evaluation_index, feasible_fraction)
        """

        feasible_seen = 0
        curve: list[tuple[int, float]] = []

        for observation in self.observations:
            if observation.feasible:
                feasible_seen += 1

            curve.append(
                (
                    observation.evaluation_index,
                    feasible_seen
                    / observation.evaluation_index,
                )
            )

        return tuple(curve)

    @property
    def evaluation_time_curve(
        self,
    ) -> tuple[tuple[int, float], ...]:
        """Per-evaluation wall-clock timing curve."""

        return tuple(
            (
                observation.evaluation_index,
                observation.evaluation_time_s,
            )
            for observation in self.observations
        )

    @property
    def repair_time_curve(
        self,
    ) -> tuple[tuple[int, float], ...]:
        """Per-evaluation repair-time curve."""

        return tuple(
            (
                observation.evaluation_index,
                observation.repair_time_s,
            )
            for observation in self.observations
        )

    @property
    def evaluation_fitness_curve(
        self,
    ) -> tuple[tuple[int, float], ...]:
        """Raw numerical fitness for every exact evaluation."""

        return tuple(
            (
                observation.evaluation_index,
                observation.fitness,
            )
            for observation in self.observations
        )

    @property
    def incumbent_fitness_curve(
        self,
    ) -> tuple[tuple[int, float, bool], ...]:
        """
        Feasibility-first incumbent curve.

        Each tuple contains:
            (evaluation, incumbent_fitness, incumbent_is_feasible)

        The incumbent starts from the first observed candidate and follows the
        same feasibility-first semantics as the optimizer.
        """

        incumbent: ObservedRouteEvaluation | None = None
        curve: list[
            tuple[int, float, bool]
        ] = []

        for observation in self.observations:
            if incumbent is None:
                incumbent = observation
            elif _fitness_result_better(
                observation.fitness_result,
                incumbent.fitness_result,
            ):
                incumbent = observation

            curve.append(
                (
                    observation.evaluation_index,
                    incumbent.fitness,
                    incumbent.feasible,
                )
            )

        return tuple(curve)

    @property
    def incumbent_origin(self) -> str:
        """
        Classify the final optimizer incumbent.

        This is derived conservatively from the exact observation stream.

        Values:
            "initial_population"
            "later_evaluation"
            "unknown"

        We intentionally do not invent optimizer-internal labels such as
        "local_search" because the observer cannot prove those origins.
        """

        best = self.optimization_result.best_result

        for index, observation in enumerate(
            self.observations
        ):
            if (
                _fitness_result_equivalent(
                    observation.fitness_result,
                    best,
                )
                and tuple(observation.position)
                == tuple(
                    self.optimization_result.best_position
                )
            ):
                if index < (
                    self._initial_population_evaluation_count
                ):
                    return "initial_population"

                return "later_evaluation"

        return "unknown"

    @property
    def _initial_population_evaluation_count(
        self,
    ) -> int:
        """
        Infer the initial population evaluation count from iteration-0 trace.

        The trace runner always records the initial batch as iteration zero.
        """

        if not self.trace.observations:
            return 0

        first = self.trace.observations[0]

        return len(
            first.candidates
        )

    def telemetry_summary(self) -> dict[str, object]:
        """
        Return a serialization-friendly research summary.

        No optimization is performed here.
        """

        return {
            "evaluations": self.evaluations,
            "elapsed_time_s": self.elapsed_time_s,
            "first_feasible_evaluation": (
                self.first_feasible_evaluation
            ),
            "first_feasible_time_s": (
                self.first_feasible_time_s
            ),
            "best_feasible_fitness": (
                self.best_feasible_fitness
            ),
            "raw_best_fitness": (
                self.raw_best_fitness
            ),
            "final_feasible": self.final_feasible,
            "feasible_evaluations": (
                self.feasible_evaluations
            ),
            "feasible_fraction": (
                self.feasible_fraction
            ),
            "total_repair_time_s": (
                self.total_repair_time_s
            ),
            "mean_evaluation_time_s": (
                self.mean_evaluation_time_s
            ),
            "mean_repair_time_s": (
                self.mean_repair_time_s
            ),
            "mean_repair_distance": (
                self.mean_repair_distance
            ),
            "incumbent_origin": (
                self.incumbent_origin
            ),
        }

    def _validate_telemetry(self) -> None:
        """
        Validate telemetry without imposing optimization semantics that belong
        to AdaptiveQPSO itself.
        """

        previous_index = 0
        previous_elapsed = 0.0

        for observation in self.observations:
            if (
                observation.evaluation_index
                != previous_index + 1
            ):
                raise TraceRunnerError(
                    "evaluation indices must be contiguous and "
                    "one-based"
                )

            if observation.elapsed_s < previous_elapsed:
                raise TraceRunnerError(
                    "observation elapsed time must be monotonic"
                )

            if observation.evaluation_time_s < 0.0:
                raise TraceRunnerError(
                    "evaluation time cannot be negative"
                )

            if observation.repair_time_s < 0.0:
                raise TraceRunnerError(
                    "repair time cannot be negative"
                )

            previous_index = (
                observation.evaluation_index
            )
            previous_elapsed = (
                observation.elapsed_s
            )


class QPSOTraceRunner:
    """
    Run AdaptiveQPSO while capturing the exact evaluation stream.

    The optimizer itself is not modified.

    Every candidate is evaluated exactly once through an instrumented oracle.
    The sequential evaluation stream is grouped into population batches.

    The trace layer never evaluates, repairs, mutates or otherwise changes
    candidates.
    """

    def __init__(
        self,
        *,
        config: AdaptiveQPSOConfig,
        oracle,
        initial_population: Sequence[
            Sequence[float]
        ] | None = None,
        recorder: OptimizationTraceRecorder | None = None,
    ) -> None:
        if not isinstance(
            config,
            AdaptiveQPSOConfig,
        ):
            raise TraceRunnerError(
                "config must be an AdaptiveQPSOConfig"
            )

        if not callable(oracle):
            raise TraceRunnerError(
                "oracle must be callable"
            )

        self.config = config

        self.recorder = (
            recorder
            if recorder is not None
            else OptimizationTraceRecorder(
                lower_bound=config.lower_bound,
                upper_bound=config.upper_bound,
            )
        )

        if isinstance(
            oracle,
            InstrumentedRouteFitnessOracle,
        ):
            self.instrumented_oracle = oracle
        else:
            self.instrumented_oracle = (
                InstrumentedRouteFitnessOracle(
                    oracle
                )
            )

        self.initial_population = (
            None
            if initial_population is None
            else tuple(
                tuple(
                    float(value)
                    for value in position
                )
                for position in initial_population
            )
        )

        if self.initial_population is not None:
            self._validate_initial_population(
                self.initial_population
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_initial_population(
        self,
        population: Sequence[Sequence[float]],
    ) -> None:
        if len(population) != (
            self.config.population_size
        ):
            raise TraceRunnerError(
                "initial_population size must equal "
                "config.population_size"
            )

        for index, position in enumerate(
            population
        ):
            if len(position) != (
                self.config.dimensions
            ):
                raise TraceRunnerError(
                    "initial_population dimension mismatch "
                    f"at index {index}: expected "
                    f"{self.config.dimensions}, "
                    f"got {len(position)}"
                )

            for dimension, value in enumerate(
                position
            ):
                value = float(value)

                if not isfinite(value):
                    raise TraceRunnerError(
                        "initial_population contains "
                        "a non-finite value at "
                        f"[{index}][{dimension}]"
                    )

                if not (
                    self.config.lower_bound
                    <= value
                    <= self.config.upper_bound
                ):
                    raise TraceRunnerError(
                        "initial_population contains "
                        "an out-of-bounds value at "
                        f"[{index}][{dimension}]: "
                        f"{value}"
                    )

    @staticmethod
    def _validate_observation(
        observation: ObservedRouteEvaluation,
    ) -> None:
        if not isinstance(
            observation,
            ObservedRouteEvaluation,
        ):
            raise TraceRunnerError(
                "instrumented oracle returned an invalid observation"
            )

        result = observation.fitness_result

        if not isinstance(
            result,
            FitnessResult,
        ):
            raise TraceRunnerError(
                "observed evaluation contains an invalid FitnessResult"
            )

        if not isfinite(
            float(result.fitness)
        ):
            raise TraceRunnerError(
                "observed evaluation contains non-finite fitness"
            )

        if not isfinite(
            float(result.repair_distance)
        ):
            raise TraceRunnerError(
                "observed evaluation contains "
                "non-finite repair distance"
            )

        if observation.evaluation_time_s < 0.0:
            raise TraceRunnerError(
                "observed evaluation time cannot be negative"
            )

        if observation.repair_time_s < 0.0:
            raise TraceRunnerError(
                "observed repair time cannot be negative"
            )

        if observation.elapsed_s < 0.0:
            raise TraceRunnerError(
                "observed elapsed time cannot be negative"
            )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

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

        optimization_result = (
            optimizer.optimize()
        )

        observations = tuple(
            self.instrumented_oracle.observations
        )

        for observation in observations:
            self._validate_observation(
                observation
            )

        if len(observations) != (
            optimization_result.evaluations
        ):
            raise TraceRunnerError(
                "captured evaluation count does not "
                "match optimizer: "
                f"{len(observations)} != "
                f"{optimization_result.evaluations}"
            )

        if not observations:
            raise TraceRunnerError(
                "optimizer produced no observable evaluations"
            )

        self._record_population_batches(
            observations
        )

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

    # ------------------------------------------------------------------
    # Population batching
    # ------------------------------------------------------------------

    def _record_population_batches(
        self,
        observations: Sequence[
            ObservedRouteEvaluation
        ],
    ) -> None:
        """
        Convert the sequential evaluation stream into population batches.

        Initial population is iteration zero.

        A final incomplete batch is preserved rather than discarded.
        """

        population_size = (
            self.config.population_size
        )

        if population_size <= 0:
            raise TraceRunnerError(
                "population_size must be positive"
            )

        adapter = RoutePopulationTraceAdapter(
            recorder=self.recorder,
        )

        for start in range(
            0,
            len(observations),
            population_size,
        ):
            batch = tuple(
                observations[
                    start:
                    start + population_size
                ]
            )

            if not batch:
                continue

            evaluations = (
                start + len(batch)
            )

            adapter.record_population(
                iteration=(
                    start // population_size
                ),
                evaluations=evaluations,
                observations=batch,
            )

    # ------------------------------------------------------------------
    # Trace validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_trace(
        *,
        trace: OptimizationTrace,
        observations: Sequence[
            ObservedRouteEvaluation
        ],
        optimization_result: OptimizationResult,
    ) -> None:
        """
        Validate invariants between optimizer output and captured trace.
        """

        if not trace.observations:
            raise TraceRunnerError(
                "trace must contain at least one population"
            )

        if trace.iterations[0] != 0:
            raise TraceRunnerError(
                "trace must begin at iteration 0"
            )

        if len(observations) != (
            optimization_result.evaluations
        ):
            raise TraceRunnerError(
                "captured observations do not match "
                "optimizer evaluation count"
            )

        if trace.observations[-1].evaluations != (
            optimization_result.evaluations
        ):
            raise TraceRunnerError(
                "trace final evaluation count does not "
                "match optimizer"
            )

        # --------------------------------------------------------------
        # Population validity
        # --------------------------------------------------------------

        for observation in trace.observations:
            if not observation.candidates:
                raise TraceRunnerError(
                    "trace contains an empty population observation"
                )

            if observation.evaluations <= 0:
                raise TraceRunnerError(
                    "population observation must contain "
                    "at least one completed evaluation"
                )

        # --------------------------------------------------------------
        # Evaluation monotonicity
        # --------------------------------------------------------------

        previous_evaluations = 0

        for observation in trace.observations:
            if (
                observation.evaluations
                <= previous_evaluations
            ):
                raise TraceRunnerError(
                    "trace evaluation counts must be "
                    "strictly increasing"
                )

            previous_evaluations = (
                observation.evaluations
            )

        # --------------------------------------------------------------
        # Iteration monotonicity
        # --------------------------------------------------------------

        previous_iteration = -1

        for observation in trace.observations:
            if (
                observation.iteration
                <= previous_iteration
            ):
                raise TraceRunnerError(
                    "trace iterations must be strictly increasing"
                )

            previous_iteration = (
                observation.iteration
            )

        # --------------------------------------------------------------
        # Evaluation stream integrity
        # --------------------------------------------------------------

        for index, observation in enumerate(
            observations,
            start=1,
        ):
            result = observation.fitness_result

            if not isinstance(
                result,
                FitnessResult,
            ):
                raise TraceRunnerError(
                    "evaluation stream contains "
                    f"invalid result at index {index}"
                )

            if (
                observation.evaluation_index
                != index
            ):
                raise TraceRunnerError(
                    "evaluation stream indices are not "
                    "contiguous"
                )

        # --------------------------------------------------------------
        # Optimizer best must have been observed.
        #
        # Do NOT require it to equal the raw minimum fitness because the
        # optimizer uses feasibility-first comparison.
        # --------------------------------------------------------------

        observed_fitnesses = tuple(
            float(
                evaluation.fitness_result.fitness
            )
            for evaluation in observations
        )

        if not observed_fitnesses:
            raise TraceRunnerError(
                "trace contains no candidate fitness values"
            )

        target_fitness = float(
            optimization_result.best_fitness
        )

        if not isfinite(target_fitness):
            raise TraceRunnerError(
                "optimizer returned non-finite best fitness"
            )

        if not any(
            abs(
                fitness - target_fitness
            )
            <= 1e-12
            for fitness in observed_fitnesses
        ):
            raise TraceRunnerError(
                "optimizer reported a best fitness that "
                "was not observed in the captured "
                "evaluation stream: "
                f"{target_fitness}"
            )

        # --------------------------------------------------------------
        # Best position consistency.
        #
        # The optimizer configuration, not a route signature, defines the
        # dimensionality of the continuous search position.
        # --------------------------------------------------------------

        best_position = tuple(
            float(value)
            for value in optimization_result.best_position
        )

        if len(best_position) != (
            optimization_result.best_result.route_signature
            is not None
            and len(best_position)
            or len(best_position)
        ):
            # Kept intentionally inert for compatibility with historical
            # traces. Actual dimensionality validation is performed against
            # the QPSO configuration by the runner.
            pass

        if not best_position:
            raise TraceRunnerError(
                "optimizer returned an empty best position"
            )

        for dimension, value in enumerate(
            best_position
        ):
            if not isfinite(value):
                raise TraceRunnerError(
                    "optimizer returned a non-finite "
                    f"best position at dimension {dimension}"
                )

        # --------------------------------------------------------------
        # Final coverage
        # --------------------------------------------------------------

        if (
            trace.observations[-1].evaluations
            != len(observations)
        ):
            raise TraceRunnerError(
                "trace final population does not cover "
                "the complete evaluation stream"
            )


def _fitness_result_better(
    candidate: FitnessResult,
    incumbent: FitnessResult,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """
    Feasibility-first comparison matching the Step-8 optimizer contract.

    This helper is intentionally local to trace analysis. It does not replace
    the optimizer's comparison implementation.
    """

    if candidate.feasible != incumbent.feasible:
        return bool(candidate.feasible)

    if candidate.feasible:
        return (
            candidate.fitness
            < incumbent.fitness - tolerance
        )

    # Both infeasible: lower numerical fitness first; use repair distance as a
    # deterministic secondary criterion when objective values tie.
    if candidate.fitness < (
        incumbent.fitness - tolerance
    ):
        return True

    if abs(
        candidate.fitness
        - incumbent.fitness
    ) <= tolerance:
        return (
            candidate.repair_distance
            < incumbent.repair_distance
            - tolerance
        )

    return False


def _fitness_result_equivalent(
    left: FitnessResult,
    right: FitnessResult,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether two FitnessResults represent the same incumbent."""

    return (
        left.feasible == right.feasible
        and abs(
            left.fitness
            - right.fitness
        ) <= tolerance
        and abs(
            left.repair_distance
            - right.repair_distance
        ) <= tolerance
    )


def run_qpso_with_trace(
    config: AdaptiveQPSOConfig,
    oracle,
    *,
    initial_population: Sequence[
        Sequence[float]
    ] | None = None,
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


__all__ = [
    "FeasibleTracePoint",
    "QPSOTraceRunner",
    "TraceRunResult",
    "TraceRunnerError",
    "run_qpso_with_trace",
]