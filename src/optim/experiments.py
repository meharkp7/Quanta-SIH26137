from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from statistics import mean
from typing import Callable, Iterable, Sequence

from src.optim.common import (
    FitnessOracle,
    FitnessResult,
    OptimizationConfig,
    OptimizationResult,
    Vector,
)
from src.optim.pso import (
    MatchedPSO,
    PSOConfig,
)
from src.optim.qpso import (
    AdaptiveQPSO,
    AdaptiveQPSOConfig,
)
from src.optim.qpso_variants import (
    VARIANTS,
    config_for_variant,
)


# ============================================================================
# Experiment contracts
# ============================================================================


@dataclass(frozen=True)
class ExperimentSpec:
    """
    One algorithm/condition evaluated over a fixed set of seeds.
    """

    name: str
    algorithm: str
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentRun:
    """
    Complete persisted result for one condition/seed pair.

    The full histories are retained so convergence, diversity and
    optimization dynamics can be analyzed without rerunning the optimizer.
    """

    condition: str
    algorithm: str
    seed: int

    best_fitness: float
    feasible: bool

    evaluations: int
    iterations: int

    coordinate_diversity_final: float
    route_diversity_final: float

    best_position: Vector
    route_signature: object
    repair_distance: float

    history_best: tuple[float, ...]
    history_mean: tuple[float, ...]
    history_coordinate_diversity: tuple[float, ...]
    history_route_diversity: tuple[float, ...]

    @property
    def history_length(self) -> int:
        return len(self.history_best)


@dataclass(frozen=True)
class ExperimentSummary:
    """
    Aggregate statistics over multiple independent seeds.
    """

    condition: str
    algorithm: str
    runs: int

    best_fitness_mean: float
    best_fitness_std: float

    feasible_rate: float

    evaluations_mean: float
    evaluations_std: float

    iterations_mean: float
    iterations_std: float

    coordinate_diversity_final_mean: float
    coordinate_diversity_final_std: float

    route_diversity_final_mean: float
    route_diversity_final_std: float

    repair_distance_mean: float
    repair_distance_std: float


OracleFactory = Callable[[int], FitnessOracle]


DEFAULT_QPSO_CONDITIONS = tuple(VARIANTS.keys())


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_algorithm(algorithm: str) -> None:
    if algorithm not in {"qpso", "pso"}:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; "
            "expected 'qpso' or 'pso'."
        )


def _validate_seeds(
    seeds: Sequence[int],
) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)

    if not normalized:
        raise ValueError(
            "At least one optimizer seed is required."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Optimizer seeds must be unique."
        )

    return normalized


def _validate_conditions(
    conditions: Sequence[str],
) -> tuple[str, ...]:
    normalized = tuple(
        str(condition).strip()
        for condition in conditions
    )

    if not normalized:
        raise ValueError(
            "At least one QPSO condition is required."
        )

    unknown = [
        condition
        for condition in normalized
        if condition not in VARIANTS
    ]

    if unknown:
        available = ", ".join(
            sorted(VARIANTS)
        )

        raise ValueError(
            "unknown QPSO conditions: "
            f"{unknown}; available: {available}"
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "QPSO conditions must be unique."
        )

    return normalized


# ============================================================================
# Deterministic matched initialization
# ============================================================================


def _initial_population(
    optimizer_config: OptimizationConfig,
    seed: int,
) -> tuple[Vector, ...]:
    """
    Generate one deterministic initial population.

    The exact same population can then be supplied to QPSO and PSO for a
    matched comparison, preventing initialization from becoming a hidden
    experimental variable.
    """

    rng = random.Random(int(seed))

    return tuple(
        tuple(
            rng.uniform(
                optimizer_config.lower_bound,
                optimizer_config.upper_bound,
            )
            for _ in range(
                optimizer_config.dimensions
            )
        )
        for _ in range(
            optimizer_config.population_size
        )
    )


# ============================================================================
# Result normalization
# ============================================================================


def _history_tuple(
    values: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in values
    )


def _make_run(
    *,
    condition: str,
    algorithm: str,
    seed: int,
    result: OptimizationResult,
    oracle: FitnessOracle,
) -> ExperimentRun:
    """
    Convert an optimizer result into a complete immutable experiment record.

    The optimizer's `best_result` is authoritative. We deliberately do not
    inspect the oracle's last evaluated candidate because the last candidate
    is not necessarily the best candidate.
    """

    _validate_algorithm(algorithm)

    if result.evaluations <= 0:
        raise RuntimeError(
            f"{algorithm}/{condition}/seed={seed} "
            "produced no evaluations."
        )

    oracle_calls = getattr(
        oracle,
        "calls",
        None,
    )

    if (
        oracle_calls is not None
        and int(oracle_calls) != int(result.evaluations)
    ):
        raise RuntimeError(
            "Evaluation accounting mismatch for "
            f"{algorithm}/{condition}/seed={seed}: "
            f"optimizer={result.evaluations}, "
            f"oracle={oracle_calls}."
        )

    history_best = _history_tuple(
        result.history_best
    )

    history_mean = _history_tuple(
        result.history_mean
    )

    history_coordinate_diversity = _history_tuple(
        result.history_coordinate_diversity
    )

    history_route_diversity = _history_tuple(
        result.history_route_diversity
    )

    history_lengths = {
        len(history_best),
        len(history_mean),
        len(history_coordinate_diversity),
        len(history_route_diversity),
    }

    if len(history_lengths) != 1:
        raise RuntimeError(
            "Inconsistent optimizer history lengths for "
            f"{algorithm}/{condition}/seed={seed}: "
            f"{sorted(history_lengths)}."
        )

    if not history_best:
        raise RuntimeError(
            f"{algorithm}/{condition}/seed={seed} "
            "returned an empty optimization history."
        )

    best_result: FitnessResult = result.best_result

    if not isinstance(
        best_result,
        FitnessResult,
    ):
        raise RuntimeError(
            f"{algorithm}/{condition}/seed={seed} "
            "returned an invalid best_result."
        )

    if len(result.best_position) == 0:
        raise RuntimeError(
            f"{algorithm}/{condition}/seed={seed} "
            "returned an empty best_position."
        )

    route_signature = best_result.route_signature

    return ExperimentRun(
        condition=condition,
        algorithm=algorithm,
        seed=int(seed),
        best_fitness=float(
            result.best_fitness
        ),
        feasible=bool(
            best_result.feasible
        ),
        evaluations=int(
            result.evaluations
        ),
        iterations=int(
            result.iterations
        ),
        coordinate_diversity_final=float(
            history_coordinate_diversity[-1]
        ),
        route_diversity_final=float(
            history_route_diversity[-1]
        ),
        best_position=tuple(
            float(value)
            for value in result.best_position
        ),
        route_signature=route_signature,
        repair_distance=float(
            best_result.repair_distance
        ),
        history_best=history_best,
        history_mean=history_mean,
        history_coordinate_diversity=(
            history_coordinate_diversity
        ),
        history_route_diversity=(
            history_route_diversity
        ),
    )


# ============================================================================
# Individual optimizer execution
# ============================================================================


def _run_qpso(
    *,
    condition: str,
    seed: int,
    base_config: AdaptiveQPSOConfig,
    oracle: FitnessOracle,
    initial_population: Sequence[Sequence[float]] | None,
) -> ExperimentRun:
    config = config_for_variant(
        base_config,
        condition,
    )

    optimizer = AdaptiveQPSO(
        config,
        oracle,
        initial_population=initial_population,
    )

    result = optimizer.optimize()

    return _make_run(
        condition=condition,
        algorithm="qpso",
        seed=seed,
        result=result,
        oracle=oracle,
    )


def _run_pso(
    *,
    seed: int,
    base_config: AdaptiveQPSOConfig,
    oracle: FitnessOracle,
    initial_population: Sequence[Sequence[float]] | None,
) -> ExperimentRun:
    """
    Run the matched classical PSO baseline.

    Every parameter shared with QPSO is copied directly from the base QPSO
    configuration. PSO-specific update parameters remain at their declared
    baseline defaults.
    """

    config = PSOConfig(
        dimensions=base_config.dimensions,
        lower_bound=base_config.lower_bound,
        upper_bound=base_config.upper_bound,
        population_size=base_config.population_size,
        max_evaluations=base_config.max_evaluations,
        seed=seed,
        tolerance=base_config.tolerance,
        stagnation_patience=(
            base_config.stagnation_patience
        ),
    )

    optimizer = MatchedPSO(
        config,
        oracle,
        initial_population=initial_population,
    )

    result = optimizer.optimize()

    return _make_run(
        condition="matched_pso",
        algorithm="pso",
        seed=seed,
        result=result,
        oracle=oracle,
    )


# ============================================================================
# Public condition runner
# ============================================================================


def run_condition(
    *,
    condition: str,
    algorithm: str,
    seed: int,
    base_qpso_config: AdaptiveQPSOConfig,
    oracle_factory: OracleFactory,
    initial_population: Sequence[Sequence[float]] | None = None,
) -> ExperimentRun:
    """
    Run one algorithm/condition/seed combination.

    If `initial_population` is supplied, it is reused exactly. This is the
    mechanism used by the matched ablation suite.
    """

    _validate_algorithm(
        algorithm
    )

    seed = int(seed)

    if algorithm == "qpso":
        if condition not in VARIANTS:
            available = ", ".join(
                sorted(VARIANTS)
            )

            raise ValueError(
                f"Unknown QPSO condition "
                f"{condition!r}; "
                f"available: {available}"
            )

        oracle = oracle_factory(seed)

        return _run_qpso(
            condition=condition,
            seed=seed,
            base_config=base_qpso_config,
            oracle=oracle,
            initial_population=initial_population,
        )

    oracle = oracle_factory(seed)

    return _run_pso(
        seed=seed,
        base_config=base_qpso_config,
        oracle=oracle,
        initial_population=initial_population,
    )


# ============================================================================
# Aggregation
# ============================================================================


def _mean(
    values: Sequence[float],
) -> float:
    if not values:
        raise ValueError(
            "Cannot compute mean of an empty sequence."
        )

    return mean(values)


def _std(
    values: Sequence[float],
) -> float:
    """
    Population standard deviation.

    The complete requested seed set is treated as the experimental population,
    so ddof=0 semantics are used.
    """

    if not values:
        raise ValueError(
            "Cannot compute standard deviation "
            "of an empty sequence."
        )

    average = _mean(values)

    return (
        sum(
            (value - average) ** 2
            for value in values
        )
        / len(values)
    ) ** 0.5


def _summary(
    *,
    condition: str,
    algorithm: str,
    runs: Sequence[ExperimentRun],
) -> ExperimentSummary:
    if not runs:
        raise ValueError(
            "Cannot summarize "
            f"{algorithm}/{condition}: "
            "no runs were supplied."
        )

    values = [
        float(run.best_fitness)
        for run in runs
    ]

    evaluations = [
        float(run.evaluations)
        for run in runs
    ]

    iterations = [
        float(run.iterations)
        for run in runs
    ]

    coordinate_diversity = [
        float(
            run.coordinate_diversity_final
        )
        for run in runs
    ]

    route_diversity = [
        float(
            run.route_diversity_final
        )
        for run in runs
    ]

    repair_distance = [
        float(
            run.repair_distance
        )
        for run in runs
    ]

    return ExperimentSummary(
        condition=condition,
        algorithm=algorithm,
        runs=len(runs),

        best_fitness_mean=_mean(values),
        best_fitness_std=_std(values),

        feasible_rate=(
            sum(
                1
                for run in runs
                if run.feasible
            )
            / len(runs)
        ),

        evaluations_mean=_mean(
            evaluations
        ),
        evaluations_std=_std(
            evaluations
        ),

        iterations_mean=_mean(
            iterations
        ),
        iterations_std=_std(
            iterations
        ),

        coordinate_diversity_final_mean=_mean(
            coordinate_diversity
        ),
        coordinate_diversity_final_std=_std(
            coordinate_diversity
        ),

        route_diversity_final_mean=_mean(
            route_diversity
        ),
        route_diversity_final_std=_std(
            route_diversity
        ),

        repair_distance_mean=_mean(
            repair_distance
        ),
        repair_distance_std=_std(
            repair_distance
        ),
    )


def summarize_runs(
    runs: Iterable[ExperimentRun],
) -> tuple[ExperimentSummary, ...]:
    """
    Aggregate runs by `(condition, algorithm)`.

    No seed is silently dropped and no condition is merged with another.
    """

    grouped: dict[
        tuple[str, str],
        list[ExperimentRun],
    ] = {}

    for run in runs:
        if not isinstance(
            run,
            ExperimentRun,
        ):
            raise TypeError(
                "summarize_runs expects "
                "ExperimentRun objects"
            )

        grouped.setdefault(
            (
                run.condition,
                run.algorithm,
            ),
            [],
        ).append(run)

    summaries = [
        _summary(
            condition=condition,
            algorithm=algorithm,
            runs=group,
        )
        for (
            condition,
            algorithm,
        ), group in sorted(
            grouped.items()
        )
    ]

    return tuple(summaries)


# ============================================================================
# Matched ablation suite
# ============================================================================


def run_ablation_suite(
    *,
    base_qpso_config: AdaptiveQPSOConfig,
    oracle_factory: OracleFactory,
    seeds: Sequence[int],
    conditions: Sequence[str] = DEFAULT_QPSO_CONDITIONS,
    include_pso: bool = True,
) -> tuple[
    tuple[ExperimentRun, ...],
    tuple[ExperimentSummary, ...],
]:
    """
    Run the complete matched QPSO ablation suite.

    For every seed:

        1. generate exactly one initial population;
        2. reuse that population for every QPSO condition;
        3. reuse the same population for matched PSO.

    Every condition receives:

        - identical scenario/oracle factory semantics
        - identical population size
        - identical initial population
        - identical evaluation budget
        - identical optimizer seed

    Only the QPSO mechanism configuration or the optimizer update rule is
    allowed to differ.
    """

    normalized_seeds = _validate_seeds(
        seeds
    )

    normalized_conditions = _validate_conditions(
        conditions
    )

    runs: list[ExperimentRun] = []

    for seed in normalized_seeds:
        initial_population = _initial_population(
            base_qpso_config,
            seed,
        )

        for condition in normalized_conditions:
            runs.append(
                run_condition(
                    condition=condition,
                    algorithm="qpso",
                    seed=seed,
                    base_qpso_config=base_qpso_config,
                    oracle_factory=oracle_factory,
                    initial_population=initial_population,
                )
            )

        if include_pso:
            runs.append(
                run_condition(
                    condition="matched_pso",
                    algorithm="pso",
                    seed=seed,
                    base_qpso_config=base_qpso_config,
                    oracle_factory=oracle_factory,
                    initial_population=initial_population,
                )
            )

    return (
        tuple(runs),
        summarize_runs(runs),
    )


# ============================================================================
# Persistent experiment artifacts
# ============================================================================


def _run_to_dict(
    run: ExperimentRun,
) -> dict[str, object]:
    """
    Convert a run into a JSON/CSV-safe dictionary.

    Lists are used instead of tuples because JSON has no tuple type.
    """

    data = asdict(run)

    data["best_position"] = list(
        run.best_position
    )

    data["history_best"] = list(
        run.history_best
    )

    data["history_mean"] = list(
        run.history_mean
    )

    data["history_coordinate_diversity"] = list(
        run.history_coordinate_diversity
    )

    data["history_route_diversity"] = list(
        run.history_route_diversity
    )

    return data


def write_results(
    output_dir: str | Path,
    runs: Sequence[ExperimentRun],
    summaries: Sequence[ExperimentSummary],
) -> tuple[Path, Path, Path]:
    """
    Persist run-level and aggregate experiment artifacts.

    Files:

        runs.csv
        runs.json
        summary.json

    `runs.json` is the canonical lossless trajectory representation.
    `runs.csv` is provided for spreadsheet/statistical inspection.
    """

    output_path = Path(
        output_dir
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_runs = tuple(
        runs
    )

    normalized_summaries = tuple(
        summaries
    )

    for run in normalized_runs:
        if not isinstance(
            run,
            ExperimentRun,
        ):
            raise TypeError(
                "runs must contain only "
                "ExperimentRun objects"
            )

    for summary in normalized_summaries:
        if not isinstance(
            summary,
            ExperimentSummary,
        ):
            raise TypeError(
                "summaries must contain only "
                "ExperimentSummary objects"
            )

    run_dicts = [
        _run_to_dict(run)
        for run in normalized_runs
    ]

    # --------------------------------------------------------------
    # Run-level JSON
    # --------------------------------------------------------------

    runs_json = (
        output_path
        / "runs.json"
    )

    runs_json.write_text(
        json.dumps(
            run_dicts,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Run-level CSV
    # --------------------------------------------------------------

    runs_csv = (
        output_path
        / "runs.csv"
    )

    if run_dicts:
        fieldnames = list(
            run_dicts[0].keys()
        )

        with runs_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in run_dicts:
                csv_row = dict(row)

                for field in (
                    "best_position",
                    "route_signature",
                    "history_best",
                    "history_mean",
                    "history_coordinate_diversity",
                    "history_route_diversity",
                ):
                    csv_row[field] = json.dumps(
                        csv_row[field],
                        separators=(",", ":"),
                        default=str,
                    )

                writer.writerow(
                    csv_row
                )
    else:
        # Still create a valid empty CSV with the canonical schema.
        fieldnames = [
            "condition",
            "algorithm",
            "seed",
            "best_fitness",
            "feasible",
            "evaluations",
            "iterations",
            "coordinate_diversity_final",
            "route_diversity_final",
            "best_position",
            "route_signature",
            "repair_distance",
            "history_best",
            "history_mean",
            "history_coordinate_diversity",
            "history_route_diversity",
        ]

        with runs_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()

    # --------------------------------------------------------------
    # Aggregate summary JSON
    # --------------------------------------------------------------

    summary_json = (
        output_path
        / "summary.json"
    )

    summary_json.write_text(
        json.dumps(
            [
                asdict(summary)
                for summary in normalized_summaries
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    return (
        runs_csv,
        runs_json,
        summary_json,
    )


__all__ = [
    "DEFAULT_QPSO_CONDITIONS",
    "ExperimentRun",
    "ExperimentSpec",
    "ExperimentSummary",
    "run_ablation_suite",
    "run_condition",
    "summarize_runs",
    "write_results",
]