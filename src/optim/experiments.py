from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
from statistics import mean, median, stdev
from typing import Callable, Iterable, Sequence

from src.optim.common import FitnessOracle, OptimizationResult, Vector
from src.optim.pso import MatchedPSO, PSOConfig
from src.optim.qpso import AdaptiveQPSO, AdaptiveQPSOConfig
from src.optim.qpso_variants import VARIANTS, config_for_variant


@dataclass(frozen=True)
class ExperimentSpec:
    """One controlled algorithm condition."""

    name: str
    algorithm: str  # "qpso" or "pso"
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentRun:
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


@dataclass(frozen=True)
class ExperimentSummary:
    condition: str
    algorithm: str
    runs: int
    mean_best_fitness: float
    median_best_fitness: float
    std_best_fitness: float
    success_rate: float
    mean_evaluations: float
    mean_iterations: float
    mean_final_coordinate_diversity: float
    mean_final_route_diversity: float
    best_seed: int
    best_fitness: float


OracleFactory = Callable[[int], FitnessOracle]


DEFAULT_QPSO_CONDITIONS = tuple(VARIANTS.keys())


def _initial_population(
    *,
    dimensions: int,
    population_size: int,
    lower_bound: float,
    upper_bound: float,
    seed: int,
) -> tuple[Vector, ...]:
    rng = random.Random(seed)
    return tuple(
        tuple(rng.uniform(lower_bound, upper_bound) for _ in range(dimensions))
        for _ in range(population_size)
    )


def _result_row(condition: str, algorithm: str, seed: int, result: OptimizationResult) -> ExperimentRun:
    return ExperimentRun(
        condition=condition,
        algorithm=algorithm,
        seed=seed,
        best_fitness=float(result.best_fitness),
        feasible=bool(result.best_result.feasible),
        evaluations=int(result.evaluations),
        iterations=int(result.iterations),
        coordinate_diversity_final=float(result.history_coordinate_diversity[-1]),
        route_diversity_final=float(result.history_route_diversity[-1]),
        best_position=tuple(result.best_position),
        route_signature=result.best_result.route_signature,
        repair_distance=float(result.best_result.repair_distance),
    )


def run_condition(
    *,
    condition: str,
    algorithm: str,
    base_qpso_config: AdaptiveQPSOConfig,
    oracle_factory: OracleFactory,
    seed: int,
    initial_population: Sequence[Sequence[float]],
) -> ExperimentRun:
    """Run exactly one condition with an isolated oracle and matched seed/population."""
    oracle = oracle_factory(seed)

    if algorithm == "qpso":
        config = config_for_variant(base_qpso_config, condition)
        optimizer = AdaptiveQPSO(
            config,
            oracle,
            initial_population=initial_population,
        )
    elif algorithm == "pso":
        config = PSOConfig(
            dimensions=base_qpso_config.dimensions,
            lower_bound=base_qpso_config.lower_bound,
            upper_bound=base_qpso_config.upper_bound,
            population_size=base_qpso_config.population_size,
            max_evaluations=base_qpso_config.max_evaluations,
            seed=seed,
            tolerance=base_qpso_config.tolerance,
            stagnation_patience=base_qpso_config.stagnation_patience,
        )
        optimizer = MatchedPSO(
            config,
            oracle,
            initial_population=initial_population,
        )
    else:
        raise ValueError(f"unsupported algorithm: {algorithm!r}")

    result = optimizer.optimize()
    if oracle.calls != result.evaluations:
        raise RuntimeError(
            f"oracle accounting mismatch for {condition}/{seed}: "
            f"calls={oracle.calls}, evaluations={result.evaluations}"
        )
    return _result_row(condition, algorithm, seed, result)


def summarize_runs(runs: Iterable[ExperimentRun]) -> tuple[ExperimentSummary, ...]:
    grouped: dict[tuple[str, str], list[ExperimentRun]] = {}
    for run in runs:
        grouped.setdefault((run.condition, run.algorithm), []).append(run)

    summaries: list[ExperimentSummary] = []
    for (condition, algorithm), group in sorted(grouped.items()):
        values = [r.best_fitness for r in group]
        best = min(group, key=lambda r: (r.best_fitness, r.seed))
        summaries.append(
            ExperimentSummary(
                condition=condition,
                algorithm=algorithm,
                runs=len(group),
                mean_best_fitness=mean(values),
                median_best_fitness=median(values),
                std_best_fitness=stdev(values) if len(values) > 1 else 0.0,
                success_rate=sum(r.feasible for r in group) / len(group),
                mean_evaluations=mean(r.evaluations for r in group),
                mean_iterations=mean(r.iterations for r in group),
                mean_final_coordinate_diversity=mean(r.coordinate_diversity_final for r in group),
                mean_final_route_diversity=mean(r.route_diversity_final for r in group),
                best_seed=best.seed,
                best_fitness=best.best_fitness,
            )
        )
    return tuple(summaries)


def run_ablation_suite(
    *,
    base_qpso_config: AdaptiveQPSOConfig,
    oracle_factory: OracleFactory,
    seeds: Sequence[int],
    conditions: Sequence[str] = DEFAULT_QPSO_CONDITIONS,
    include_pso: bool = True,
) -> tuple[tuple[ExperimentRun, ...], tuple[ExperimentSummary, ...]]:
    """Run a paired ablation study under identical budgets, seeds and initial populations."""
    seeds = tuple(int(seed) for seed in seeds)
    conditions = tuple(conditions)
    unknown = [name for name in conditions if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown QPSO conditions: {unknown}")

    runs: list[ExperimentRun] = []
    for seed in seeds:
        population = _initial_population(
            dimensions=base_qpso_config.dimensions,
            population_size=base_qpso_config.population_size,
            lower_bound=base_qpso_config.lower_bound,
            upper_bound=base_qpso_config.upper_bound,
            seed=seed,
        )
        for condition in conditions:
            runs.append(
                run_condition(
                    condition=condition,
                    algorithm="qpso",
                    base_qpso_config=base_qpso_config,
                    oracle_factory=oracle_factory,
                    seed=seed,
                    initial_population=population,
                )
            )
        if include_pso:
            runs.append(
                run_condition(
                    condition="matched_pso",
                    algorithm="pso",
                    base_qpso_config=base_qpso_config,
                    oracle_factory=oracle_factory,
                    seed=seed,
                    initial_population=population,
                )
            )

    return tuple(runs), summarize_runs(runs)


def write_results(
    output_dir: str | Path,
    runs: Sequence[ExperimentRun],
    summaries: Sequence[ExperimentSummary],
) -> tuple[Path, Path]:
    """Write machine-readable run-level CSV and JSON summary artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    runs_csv = output / "runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "condition", "algorithm", "seed", "best_fitness", "feasible",
                "evaluations", "iterations", "coordinate_diversity_final",
                "route_diversity_final", "best_position", "route_signature", "repair_distance",
            ),
        )
        writer.writeheader()
        for run in runs:
            row = asdict(run)
            row["best_position"] = json.dumps(row["best_position"])
            row["route_signature"] = json.dumps(row["route_signature"], default=str)
            writer.writerow(row)

    summary_json = output / "summary.json"
    summary_json.write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2),
        encoding="utf-8",
    )
    return runs_csv, summary_json