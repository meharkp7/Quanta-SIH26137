from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.optim.predictive_analysis import (
    CorrelationResult,
    HorizonSample,
    PredictiveAnalysisResult,
    SearchState,
    analyze_predictive_relationships,
    build_horizon_samples,
)

from src.optim.trace_experiments import (
    TraceExperimentRun,
    run_qpso_trace_experiment,
)


# ============================================================
# ERRORS
# ============================================================


class PredictiveExperimentError(ValueError):
    """Raised when predictive-experiment inputs are invalid."""


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class PredictiveExperimentConfig:
    """
    Configuration for predictive analysis.

    Horizons are measured in optimizer iterations, not raw
    objective-function evaluations.
    """

    horizons: tuple[int, ...] = (1, 2, 3)

    @classmethod
    def from_horizons(
        cls,
        horizons: Sequence[int],
    ) -> "PredictiveExperimentConfig":

        normalized = tuple(
            int(horizon)
            for horizon in horizons
        )

        if not normalized:
            raise PredictiveExperimentError(
                "At least one prediction horizon is required."
            )

        if any(
            horizon <= 0
            for horizon in normalized
        ):
            raise PredictiveExperimentError(
                "Prediction horizons must be positive."
            )

        if len(set(normalized)) != len(normalized):
            raise PredictiveExperimentError(
                "Prediction horizons must be unique."
            )

        return cls(
            horizons=normalized
        )


# ============================================================
# RESULT CONTAINERS
# ============================================================


@dataclass(frozen=True)
class PredictiveExperimentRun:
    """
    One trace-backed predictive experiment.

    The original optimization trace is retained together with
    the SearchState representation used for prediction analysis.
    """

    trace_run: TraceExperimentRun
    states: tuple[SearchState, ...]

    def __post_init__(self) -> None:

        if not self.states:
            raise PredictiveExperimentError(
                "Predictive experiment requires at least one SearchState."
            )

    @property
    def condition(self) -> str:
        return self.trace_run.condition

    @property
    def algorithm(self) -> str:
        return self.trace_run.algorithm

    @property
    def seed(self) -> int:
        return self.trace_run.seed

    @property
    def evaluations(self) -> int:
        return self.trace_run.evaluations

    @property
    def iterations(self) -> int:
        return self.trace_run.iterations

    @property
    def best_fitness(self) -> float:
        return self.trace_run.best_fitness


@dataclass(frozen=True)
class PredictiveExperimentBatch:
    """
    Collection of predictive experiment runs.
    """

    runs: tuple[PredictiveExperimentRun, ...]

    def __post_init__(self) -> None:

        if not self.runs:
            raise PredictiveExperimentError(
                "Predictive experiment batch cannot be empty."
            )


@dataclass(frozen=True)
class PredictiveConditionResult:
    """
    Predictive-analysis result for one optimization condition.
    """

    condition: str

    runs: tuple[PredictiveExperimentRun, ...]

    analysis: PredictiveAnalysisResult

    def strongest_signal(
        self,
        *,
        horizon: int,
    ) -> CorrelationResult | None:

        return self.analysis.strongest_signal(
            horizon=horizon
        )


# ============================================================
# TRACE → SEARCH STATE
# ============================================================


def trace_to_search_states(
    run: TraceExperimentRun,
) -> tuple[SearchState, ...]:
    """
    Convert population-level optimization observations into
    SearchState objects.

    Every signal recorded by the instrumentation layer is
    preserved explicitly.
    """

    states: list[SearchState] = []

    for observation in run.trace.observations:

        diversity = observation.diversity
        repair_pressure = observation.repair_pressure

        states.append(
            SearchState(
                iteration=observation.iteration,

                evaluations=observation.evaluations,

                best_fitness=observation.best_fitness,

                coordinate_diversity=(
                    diversity.genotype_coordinate
                ),

                decoded_assignment_diversity=(
                    diversity.decoded_assignment
                ),

                decoded_precedence_diversity=(
                    diversity.decoded_precedence
                ),

                decoded_structure_diversity=(
                    diversity.decoded_structure
                ),

                repaired_assignment_diversity=(
                    diversity.repaired_assignment
                ),

                repaired_precedence_diversity=(
                    diversity.repaired_precedence
                ),

                repaired_structure_diversity=(
                    diversity.repaired_structure
                ),

                repair_pressure=(
                    repair_pressure.exact
                ),

                feasible_rate=(
                    observation.feasible_rate
                ),
            )
        )

    return tuple(states)


# ============================================================
# SINGLE TRACE
# ============================================================


def run_predictive_experiment(
    trace_run: TraceExperimentRun,
) -> PredictiveExperimentRun:
    """
    Convert one completed trace experiment into the
    SearchState representation.
    """

    states = trace_to_search_states(
        trace_run
    )

    return PredictiveExperimentRun(
        trace_run=trace_run,
        states=states,
    )


# ============================================================
# MULTI-SEED TRACE GENERATION
# ============================================================


def run_predictive_trace_batch(
    *,
    condition: str,
    seeds: Sequence[int],
    base_qpso_config,
    oracle_factory,
) -> PredictiveExperimentBatch:
    """
    Run one QPSO condition across multiple independent seeds.

    Each resulting optimization trace is converted into
    SearchState observations.
    """

    normalized_seeds = tuple(
        int(seed)
        for seed in seeds
    )

    if not normalized_seeds:
        raise PredictiveExperimentError(
            "At least one seed is required."
        )

    runs: list[PredictiveExperimentRun] = []

    for seed in normalized_seeds:

        trace_run = run_qpso_trace_experiment(
            condition=condition,
            seed=seed,
            base_qpso_config=base_qpso_config,
            oracle_factory=oracle_factory,
        )

        runs.append(
            run_predictive_experiment(
                trace_run
            )
        )

    return PredictiveExperimentBatch(
        runs=tuple(runs)
    )


# ============================================================
# FUTURE-IMPROVEMENT SAMPLES
# ============================================================


def future_improvement_samples(
    states: Sequence[SearchState],
    horizon: int,
) -> tuple[HorizonSample, ...]:
    """
    Build future-improvement samples for one trace.

    For minimization:

        improvement(t, h)
            = best_fitness(t)
              - best_fitness(t+h)

    Positive values therefore indicate improvement.
    """

    if horizon <= 0:
        raise PredictiveExperimentError(
            "Prediction horizon must be positive."
        )

    return tuple(
        sample
        for sample in build_horizon_samples(
            states,
            horizons=(horizon,),
        )
    )


# ============================================================
# AGGREGATE SAMPLES ACROSS SEEDS
# ============================================================


def aggregate_horizon_samples(
    runs: Sequence[PredictiveExperimentRun],
    horizon: int,
) -> tuple[HorizonSample, ...]:
    """
    Pool horizon samples from multiple independent seeds.

    Samples remain individual observations; they are not averaged
    before correlation analysis.
    """

    if horizon <= 0:
        raise PredictiveExperimentError(
            "Prediction horizon must be positive."
        )

    samples: list[HorizonSample] = []

    for run in runs:

        samples.extend(
            future_improvement_samples(
                run.states,
                horizon,
            )
        )

    return tuple(samples)


# ============================================================
# ANALYZE ONE CONDITION
# ============================================================


def analyze_condition(
    runs: Sequence[PredictiveExperimentRun],
    *,
    horizons: Sequence[int] = (1, 2, 3),
) -> PredictiveConditionResult:
    """
    Analyze predictive relationships for one optimization
    condition across all supplied seeds.

    The existing predictive_analysis module performs the actual
    Spearman analysis.
    """

    if not runs:
        raise PredictiveExperimentError(
            "At least one run is required."
        )

    condition_names = {
        run.condition
        for run in runs
    }

    if len(condition_names) != 1:
        raise PredictiveExperimentError(
            "analyze_condition() requires exactly one condition."
        )

    normalized_horizons = tuple(
        int(horizon)
        for horizon in horizons
    )

    if not normalized_horizons:
        raise PredictiveExperimentError(
            "At least one prediction horizon is required."
        )

    if any(
        horizon <= 0
        for horizon in normalized_horizons
    ):
        raise PredictiveExperimentError(
            "Prediction horizons must be positive."
        )

    if len(set(normalized_horizons)) != len(
        normalized_horizons
    ):
        raise PredictiveExperimentError(
            "Prediction horizons must be unique."
        )

    # --------------------------------------------------------
    # Pool states across seeds.
    #
    # We intentionally do NOT pool raw evaluations.
    # Each seed remains a separate optimization trajectory.
    # --------------------------------------------------------

    pooled_states: list[SearchState] = []

    for run in runs:
        pooled_states.extend(
            run.states
        )

    # --------------------------------------------------------
    # Build horizon samples independently first.
    #
    # This gives us the exact supervised observations that
    # the statistical analysis consumes.
    # --------------------------------------------------------

    pooled_samples: list[HorizonSample] = []

    for run in runs:

        pooled_samples.extend(
            build_horizon_samples(
                run.states,
                horizons=normalized_horizons,
            )
        )

    # --------------------------------------------------------
    # Run the authoritative predictive-analysis implementation.
    #
    # analyze_predictive_relationships() constructs its own
    # horizon samples from SearchState trajectories.
    #
    # We analyze each seed separately and then concatenate
    # the resulting samples/correlations so that iteration 0
    # from different seeds is not treated as the same state.
    # --------------------------------------------------------

    analyses: list[
        PredictiveAnalysisResult
    ] = []

    for run in runs:

        analysis = analyze_predictive_relationships(
            run.states,
            horizons=normalized_horizons,
        )

        analyses.append(
            analysis
        )

    # --------------------------------------------------------
    # Reconstruct a combined result.
    #
    # Correlations must be calculated over the pooled samples,
    # not by averaging per-seed correlation coefficients.
    # --------------------------------------------------------

    combined_correlations: list[
        CorrelationResult
    ] = []

    for horizon in normalized_horizons:

        horizon_samples = tuple(
            sample
            for sample in pooled_samples
            if sample.horizon == horizon
        )

        if len(horizon_samples) < 2:
            continue

        target = tuple(
            sample.future_improvement
            for sample in horizon_samples
        )

        # Import the internal rank-correlation implementation
        # only for this pooled experiment.
        from src.optim.predictive_analysis import _spearman

        for signal in (
            "coordinate_diversity",
            "decoded_assignment_diversity",
            "decoded_precedence_diversity",
            "decoded_structure_diversity",
            "repaired_assignment_diversity",
            "repaired_precedence_diversity",
            "repaired_structure_diversity",
            "repair_pressure",
            "feasible_rate",
        ):

            values = tuple(
                float(
                    getattr(
                        sample,
                        signal,
                    )
                )
                for sample in horizon_samples
            )

            coefficient = _spearman(
                values,
                target,
            )

            combined_correlations.append(
                CorrelationResult(
                    signal=signal,
                    horizon=horizon,
                    coefficient=coefficient,
                    sample_count=len(horizon_samples),
                )
            )

    combined_analysis = PredictiveAnalysisResult(
        horizons=normalized_horizons,
        samples=tuple(
            pooled_samples
        ),
        correlations=tuple(
            combined_correlations
        ),
    )

    return PredictiveConditionResult(
        condition=next(
            iter(condition_names)
        ),
        runs=tuple(runs),
        analysis=combined_analysis,
    )


# ============================================================
# ANALYZE MULTIPLE CONDITIONS
# ============================================================


def analyze_conditions(
    batch: PredictiveExperimentBatch,
    *,
    horizons: Sequence[int] = (1, 2, 3),
) -> tuple[PredictiveConditionResult, ...]:
    """
    Analyze every optimization condition independently.
    """

    grouped: dict[
        str,
        list[PredictiveExperimentRun],
    ] = {}

    for run in batch.runs:

        grouped.setdefault(
            run.condition,
            [],
        ).append(run)

    results: list[
        PredictiveConditionResult
    ] = []

    for condition in sorted(grouped):

        results.append(
            analyze_condition(
                grouped[condition],
                horizons=horizons,
            )
        )

    return tuple(results)


# ============================================================
# SERIALIZATION-FRIENDLY SUMMARY
# ============================================================


def summarize_predictive_results(
    results: Sequence[PredictiveConditionResult],
) -> list[dict[str, object]]:
    """
    Convert predictive results into flat dictionaries suitable
    for CSV/JSON reporting.
    """

    rows: list[
        dict[str, object]
    ] = []

    for condition_result in results:

        for horizon in condition_result.analysis.horizons:

            correlations = (
                condition_result.analysis.correlations_for(
                    horizon=horizon
                )
            )

            for correlation in correlations:

                rows.append(
                    {
                        "condition": (
                            condition_result.condition
                        ),
                        "horizon": (
                            horizon
                        ),
                        "signal": (
                            correlation.signal
                        ),
                        "coefficient": (
                            correlation.coefficient
                        ),
                        "absolute_coefficient": (
                            correlation.absolute_coefficient
                        ),
                        "sample_count": (
                            correlation.sample_count
                        ),
                    }
                )

    return rows