from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.optim.common import FitnessOracle, Vector
from src.optim.qpso import AdaptiveQPSOConfig
from src.optim.qpso_variants import VARIANTS, config_for_variant
from src.optim.route_observer import ObservedRouteEvaluation
from src.optim.trace_runner import QPSOTraceRunner, TraceRunResult

from .experiments import OracleFactory, _initial_population


class TraceExperimentError(RuntimeError):
    """Raised when a trace-enabled experiment is inconsistent."""


@dataclass(frozen=True)
class TraceExperimentSpec:
    """
    One trace-enabled QPSO experiment.

    A trace experiment deliberately operates on one algorithm/condition/seed
    combination. Multi-seed aggregation remains the responsibility of the
    regular experiment suite.
    """

    condition: str
    seed: int


@dataclass(frozen=True)
class TraceExperimentRun:
    """
    Persistable wrapper around one complete QPSO trace run.

    The OptimizationTrace contains the population-level search state while
    observations retain the exact evaluation-level routing artifacts.
    """

    condition: str
    algorithm: str
    seed: int
    result: TraceRunResult

    def __post_init__(self) -> None:
        if self.algorithm != "qpso":
            raise TraceExperimentError(
                "TraceExperimentRun currently supports QPSO only"
            )

        if self.condition not in VARIANTS:
            raise TraceExperimentError(
                f"Unknown QPSO condition: {self.condition!r}"
            )

        if self.result.observations is None:
            raise TraceExperimentError(
                "trace result observations cannot be None"
            )

    @property
    def optimization_result(self):
        """Underlying optimizer result."""

        return self.result.optimization_result

    @property
    def trace(self):
        """Population-level OptimizationTrace."""

        return self.result.trace

    @property
    def observations(self) -> tuple[ObservedRouteEvaluation, ...]:
        """Exact evaluation-level routing observations."""

        return self.result.observations

    @property
    def best_fitness(self) -> float:
        return float(self.optimization_result.best_fitness)

    @property
    def evaluations(self) -> int:
        return int(self.optimization_result.evaluations)

    @property
    def iterations(self) -> int:
        return int(self.optimization_result.iterations)


def run_qpso_trace_experiment(
    *,
    condition: str,
    seed: int,
    base_qpso_config: AdaptiveQPSOConfig,
    oracle_factory: OracleFactory,
    initial_population: Sequence[Vector] | None = None,
) -> TraceExperimentRun:
    """
    Run one QPSO condition while preserving the complete routing trace.

    If an initial population is supplied, it is reused exactly. This is
    important when this trace run is intended to correspond to one of the
    matched ablation runs.
    """

    if condition not in VARIANTS:
        raise ValueError(
            f"unknown QPSO condition: {condition!r}"
        )

    oracle = oracle_factory(seed)

    config = config_for_variant(
        base_qpso_config,
        condition,
    )

    if initial_population is None:
        population = _initial_population(
            base_qpso_config,
            seed,
        )
    else:
        population = tuple(
            tuple(float(value) for value in position)
            for position in initial_population
        )

    _validate_initial_population(
        config=config,
        population=population,
    )

    trace_result = QPSOTraceRunner(
        config=config,
        oracle=oracle,
        initial_population=population,
    ).run()

    return TraceExperimentRun(
        condition=condition,
        algorithm="qpso",
        seed=int(seed),
        result=trace_result,
    )


def run_trace_ablation(
    *,
    base_qpso_config: AdaptiveQPSOConfig,
    oracle_factory: OracleFactory,
    seeds: Sequence[int],
    conditions: Sequence[str],
) -> tuple[TraceExperimentRun, ...]:
    """
    Run a trace-enabled QPSO ablation suite.

    For every seed, exactly one initial population is generated and reused
    across every requested QPSO condition. This preserves the same matched
    initialization protocol as the regular ablation suite.
    """

    normalized_seeds = tuple(int(seed) for seed in seeds)
    normalized_conditions = tuple(conditions)

    if not normalized_seeds:
        raise ValueError(
            "At least one seed is required."
        )

    unknown_conditions = tuple(
        condition
        for condition in normalized_conditions
        if condition not in VARIANTS
    )

    if unknown_conditions:
        raise ValueError(
            "unknown QPSO conditions: "
            + ", ".join(unknown_conditions)
        )

    runs: list[TraceExperimentRun] = []

    for seed in normalized_seeds:
        initial_population = _initial_population(
            base_qpso_config,
            seed,
        )

        for condition in normalized_conditions:
            runs.append(
                run_qpso_trace_experiment(
                    condition=condition,
                    seed=seed,
                    base_qpso_config=base_qpso_config,
                    oracle_factory=oracle_factory,
                    initial_population=initial_population,
                )
            )

    return tuple(runs)


def _validate_initial_population(
    *,
    config: AdaptiveQPSOConfig,
    population: Sequence[Vector],
) -> None:
    """Validate a supplied population before handing it to QPSO."""

    if len(population) != config.population_size:
        raise TraceExperimentError(
            "initial population size mismatch: "
            f"{len(population)} != {config.population_size}"
        )

    for index, position in enumerate(population):
        if len(position) != config.dimensions:
            raise TraceExperimentError(
                "initial population dimension mismatch at index "
                f"{index}: {len(position)} != {config.dimensions}"
            )

        for dimension, value in enumerate(position):
            numeric_value = float(value)

            if not (
                config.lower_bound
                <= numeric_value
                <= config.upper_bound
            ):
                raise TraceExperimentError(
                    "initial population value outside bounds at "
                    f"index={index}, dimension={dimension}: "
                    f"{numeric_value} not in "
                    f"[{config.lower_bound}, {config.upper_bound}]"
                )