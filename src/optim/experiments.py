from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from src.optim.common import FitnessOracle, FitnessResult, OptimizationConfig, OptimizationResult, Vector
from src.optim.pso import MatchedPSOConfig, ParticleSwarmOptimizer
from src.optim.qpso import AdaptiveQPSO, AdaptiveQPSOConfig, RouteFitnessOracle
from src.optim.qpso_variants import VARIANTS, config_for_variant
from src.optim.diversity import repair_displacement

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

    The four history fields intentionally preserve the complete optimizer
    trajectory instead of only storing final-state metrics. This allows
    convergence, diversity, stagnation, and ablation analysis to be performed
    without rerunning the optimization.
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
    Aggregate statistics over multiple seeds for one condition.
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


DEFAULT_QPSO_CONDITIONS = tuple(VARIANTS.keys())


OracleFactory = Callable[[int], FitnessOracle]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of an empty sequence.")
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    """
    Population standard deviation.

    Experiments summarize the complete set of requested seeds, so the
    population definition is used consistently here.
    """
    if not values:
        raise ValueError("Cannot compute standard deviation of an empty sequence.")

    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def _history_tuple(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _run_to_dict(run: ExperimentRun) -> dict[str, object]:
    """
    Convert an ExperimentRun into a JSON/CSV-friendly dictionary.

    Histories are retained as JSON arrays rather than flattened so one run
    remains one atomic experimental record.
    """
    data = asdict(run)

    data["best_position"] = list(run.best_position)
    data["route_signature"] = run.route_signature

    data["history_best"] = list(run.history_best)
    data["history_mean"] = list(run.history_mean)
    data["history_coordinate_diversity"] = list(
        run.history_coordinate_diversity
    )
    data["history_route_diversity"] = list(run.history_route_diversity)

    return data


def _make_run(
    *,
    condition: str,
    algorithm: str,
    seed: int,
    result: OptimizationResult,
    oracle: FitnessOracle,
) -> ExperimentRun:
    if result.evaluations <= 0:
        raise RuntimeError(
            f"{algorithm}/{condition}/seed={seed} produced no evaluations."
        )

    if result.evaluations != oracle.calls:
        raise RuntimeError(
            f"Evaluation accounting mismatch for {algorithm}/{condition}/seed={seed}: "
            f"optimizer={result.evaluations}, oracle={oracle.calls}."
        )

    history_best = _history_tuple(result.history_best)
    history_mean = _history_tuple(result.history_mean)
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
            f"Inconsistent history lengths for {algorithm}/{condition}/seed={seed}: "
            f"{sorted(history_lengths)}."
        )

    if not history_best:
        raise RuntimeError(
            f"{algorithm}/{condition}/seed={seed} returned an empty history."
        )

    # RouteFitnessOracle exposes the final candidate evaluated by the oracle.
    route_signature: object = None
    repair_distance = 0.0
    feasible = False

    if isinstance(oracle, RouteFitnessOracle):
        candidate = oracle.last_candidate

        if candidate is not None:
            route_signature = tuple(
                (
                    vehicle_route.vehicle_id,
                    tuple(vehicle_route.customer_ids),
                )
                for vehicle_route in candidate.repaired_plan.vehicle_routes
            )

            repair_distance = float(
                repair_displacement(
                    candidate.repair_result.original_route_plan,
                    candidate.repair_result.repaired_route_plan,
                ).exact
            )

            feasible = bool(candidate.repaired_evaluation.feasible)

    # For non-route oracles, retain the optimizer's final fitness result if
    # available but do not invent route-specific metadata.
    if route_signature is None and result.best_result is not None:
        best_result: FitnessResult = result.best_result
        route_signature = best_result.route_signature
        feasible = bool(best_result.feasible)

    return ExperimentRun(
        condition=condition,
        algorithm=algorithm,
        seed=seed,
        best_fitness=float(result.best_fitness),
        feasible=feasible,
        evaluations=int(result.evaluations),
        iterations=int(result.iterations),
        coordinate_diversity_final=float(history_coordinate_diversity[-1]),
        route_diversity_final=float(history_route_diversity[-1]),
        best_position=tuple(float(value) for value in result.best_position),
        route_signature=route_signature,
        repair_distance=repair_distance,
        history_best=history_best,
        history_mean=history_mean,
        history_coordinate_diversity=history_coordinate_diversity,
        history_route_diversity=history_route_diversity,
    )


def _summary(
    condition: str,
    algorithm: str,
    runs: Sequence[ExperimentRun],
) -> ExperimentSummary:
    if not runs:
        raise ValueError(
            f"Cannot summarize condition {algorithm}/{condition}: no runs."
        )

    best_fitness = [run.best_fitness for run in runs]
    evaluations = [float(run.evaluations) for run in runs]
    iterations = [float(run.iterations) for run in runs]
    coordinate_diversity = [
        run.coordinate_diversity_final for run in runs
    ]
    route_diversity = [
        run.route_diversity_final for run in runs
    ]
    repair_distance = [run.repair_distance for run in runs]

    return ExperimentSummary(
        condition=condition,
        algorithm=algorithm,
        runs=len(runs),
        best_fitness_mean=_mean(best_fitness),
        best_fitness_std=_std(best_fitness),
        feasible_rate=sum(run.feasible for run in runs) / len(runs),
        evaluations_mean=_mean(evaluations),
        evaluations_std=_std(evaluations),
        iterations_mean=_mean(iterations),
        iterations_std=_std(iterations),
        coordinate_diversity_final_mean=_mean(coordinate_diversity),
        coordinate_diversity_final_std=_std(coordinate_diversity),
        route_diversity_final_mean=_mean(route_diversity),
        route_diversity_final_std=_std(route_diversity),
        repair_distance_mean=_mean(repair_distance),
        repair_distance_std=_std(repair_distance),
    )


def _initial_population(
    optimizer_config: OptimizationConfig,
    seed: int,
) -> tuple[Vector, ...]:
    """
    Generate one deterministic initial population.

    This function mirrors the optimizer's population initialization so
    matched QPSO/PSO comparisons can use exactly the same starting swarm.
    """
    import random

    rng = random.Random(seed)

    return tuple(
        tuple(
            rng.uniform(
                optimizer_config.lower_bound,
                optimizer_config.upper_bound,
            )
            for _ in range(optimizer_config.dimensions)
        )
        for _ in range(optimizer_config.population_size)
    )


def _run_qpso(
    *,
    condition: str,
    seed: int,
    base_config: AdaptiveQPSOConfig,
    oracle: FitnessOracle,
    initial_population: tuple[Vector, ...] | None = None,
) -> ExperimentRun:
    config = config_for_variant(base_config, condition)

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
    initial_population: tuple[Vector, ...] | None = None,
) -> ExperimentRun:
    config = MatchedPSOConfig(
        dimensions=base_config.dimensions,
        lower_bound=base_config.lower_bound,
        upper_bound=base_config.upper_bound,
        population_size=base_config.population_size,
        max_evaluations=base_config.max_evaluations,
        seed=seed,
        tolerance=base_config.tolerance,
        stagnation_patience=base_config.stagnation_patience,
    )

    optimizer = ParticleSwarmOptimizer(
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


def run_condition(
    *,
    condition: str,
    algorithm: str,
    seed: int,
    base_qpso_config: AdaptiveQPSOConfig,
    oracle_factory: OracleFactory,
    initial_population: tuple[Vector, ...] | None = None,
) -> ExperimentRun:
    """
    Run one condition/seed pair.

    `initial_population` is optional. When supplied, it is reused exactly,
    which is required for matched ablations.
    """
    if algorithm not in {"qpso", "pso"}:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; expected 'qpso' or 'pso'."
        )

    if algorithm == "qpso":
        if condition not in VARIANTS:
            raise ValueError(
                f"Unknown QPSO condition {condition!r}. "
                f"Available: {', '.join(VARIANTS)}"
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


def run_ablation_suite(
    *,
    base_qpso_config: AdaptiveQPSOConfig,
    oracle_factory: OracleFactory,
    seeds: Sequence[int],
    conditions: Sequence[str] = DEFAULT_QPSO_CONDITIONS,
    include_pso: bool = True,
) -> tuple[tuple[ExperimentRun, ...], tuple[ExperimentSummary, ...]]:
    """
    Run the matched QPSO ablation suite.

    For every seed:
      1. one initial population is generated;
      2. every QPSO condition receives that exact population;
      3. matched PSO, if enabled, receives the same population.

    This prevents initialization differences from contaminating the
    mechanism comparison.
    """
    seeds = tuple(int(seed) for seed in seeds)
    conditions = tuple(conditions)

    if not seeds:
        raise ValueError("At least one seed is required.")

    unknown_conditions = [
        condition for condition in conditions if condition not in VARIANTS
    ]
    if unknown_conditions:
        raise ValueError(
            "unknown QPSO conditions: "
            + ", ".join(unknown_conditions)
        )

    runs: list[ExperimentRun] = []

    for seed in seeds:
        initial_population = _initial_population(base_qpso_config, seed)

        for condition in conditions:
            run = run_condition(
                condition=condition,
                algorithm="qpso",
                seed=seed,
                base_qpso_config=base_qpso_config,
                oracle_factory=oracle_factory,
                initial_population=initial_population,
            )
            runs.append(run)

        if include_pso:
            run = run_condition(
                condition="matched_pso",
                algorithm="pso",
                seed=seed,
                base_qpso_config=base_qpso_config,
                oracle_factory=oracle_factory,
                initial_population=initial_population,
            )
            runs.append(run)

    grouped: dict[tuple[str, str], list[ExperimentRun]] = {}

    for run in runs:
        grouped.setdefault(
            (run.condition, run.algorithm),
            [],
        ).append(run)

    summaries = tuple(
        _summary(
            condition=condition,
            algorithm=algorithm,
            runs=grouped[(condition, algorithm)],
        )
        for condition, algorithm in sorted(
            grouped,
            key=lambda item: (item[0], item[1]),
        )
    )

    return tuple(runs), summaries


def write_results(
    output_dir: str | Path,
    runs: Sequence[ExperimentRun],
    summaries: Sequence[ExperimentSummary],
) -> None:
    """
    Persist both run-level trajectories and aggregate summaries.

    Files:
      runs.csv
      runs.json
      summary.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_dicts = [_run_to_dict(run) for run in runs]

    # CSV keeps scalar fields easy to inspect while storing trajectory arrays
    # as JSON strings in the corresponding cells.
    if run_dicts:
        csv_path = output_path / "runs.csv"

        fieldnames = list(run_dicts[0].keys())

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
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
                    )

                writer.writerow(csv_row)

    # JSON is the canonical trajectory-preserving representation.
    (output_path / "runs.json").write_text(
        json.dumps(
            run_dicts,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    summary_dicts = [asdict(summary) for summary in summaries]

    (output_path / "summary.json").write_text(
        json.dumps(
            summary_dicts,
            indent=2,
        ),
        encoding="utf-8",
    )