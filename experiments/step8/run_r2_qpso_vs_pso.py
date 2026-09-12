
"""
Step 8 R2 — diversified feasible initialization + matched QPSO/PSO.

This runner is written against the repository contracts actually used by
Step 4, Step 7 and Step 8:

    Scenario.nodes / Scenario.edges / Scenario.requests / Scenario.fleet
    Step7RouteEngine(scenario, evaluator)
    ContinuousRouteEncoder.encode(...)
    RouteCandidate.repaired_plan / repair_result.complete
    RouteFitnessOracle(route_engine)
    run_ablation_suite(..., initial_population_factory=...)

The experiment deliberately separates:
    - scenario generation
    - feasible population construction
    - population diversity validation
    - matched optimizer execution

No duplicate padding is allowed. If the requested number of unique feasible
route phenotypes cannot be produced, the run stops rather than manufacturing
diversity.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.contracts.scenario import Scenario
from src.data.dataset_generator import (
    DynamicDatasetConfig,
    generate_dynamic_scenario,
)
from src.optim.common import FitnessOracle
from src.optim.experiments import (
    run_ablation_suite,
    write_results,
)
from src.optim.qpso import (
    AdaptiveQPSOConfig,
    RouteFitnessOracle,
)
from src.optim.diversity import route_distance
from src.routing.initial_solution import (
    InitialSolutionBuilder,
    InitialSolutionConfig,
)
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_encoding import Step7RouteEngine
from src.routing.route_plan import RoutePlan, VehicleRoute


ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "artifacts" / "step8_research"

SCENARIO_SEED = 26157
DEFAULT_CUSTOMERS = 20
DEFAULT_VEHICLES = 4
DEFAULT_ROAD_JUNCTIONS = 196
DEFAULT_POPULATION = 20
DEFAULT_EVALUATIONS = 30
DEFAULT_SEEDS = (1,)

HEURISTICS = (
    "earliest_deadline",
    "earliest_release",
    "largest_demand",
    "customer_id",
    "nearest_feasible",
)

PERTURBATION_OPERATORS = (
    "relocate",
    "swap",
    "reverse",
)


@dataclass(frozen=True)
class PopulationMember:
    optimizer_seed: int
    heuristic: str
    construction_variant: int
    operation: str
    parent_index: int | None
    route_signature: str
    keys: tuple[float, ...]


def _scenario_config(args: argparse.Namespace) -> DynamicDatasetConfig:
    return DynamicDatasetConfig(
        customer_count=args.customers,
        road_junction_count=args.road_junctions,
        extent_m=5000.0,
        topology_family="grid",
        grid_rows=14,
        grid_cols=14,
        jitter_fraction=0.08,
        one_way_fraction=0.18,
        max_customer_snap_m=350.0,
        window_profile="medium",
        window_slack_s=900.0,
        service_duration_s=120.0,
        vehicle_capacity=30.0,
        fleet_size=args.vehicles,
        customer_spread=0.85,
        cluster_strength=0.20,
        minimum_road_segment_m=80.0,
        demand_min=1,
        demand_max=8,
        seed=SCENARIO_SEED,
        dataset_split="test",
    )


def _fresh_engine(scenario: Scenario) -> Step7RouteEngine:
    evaluator = RouteEvaluator(scenario)
    return Step7RouteEngine(scenario, evaluator=evaluator)


def _constructive_config(heuristic: str) -> InitialSolutionConfig:
    return InitialSolutionConfig(
        heuristic=heuristic,
        allow_unassigned=False,
        prioritize_time_windows=True,
        prioritize_distance=True,
        prioritize_travel_time=True,
        prioritize_waiting_time=True,
        require_capacity_feasibility=True,
        require_connectivity=True,
        require_time_window_feasibility=True,
        max_candidates_per_insertion=64,
        planning_time_s=0.0,
        include_future_released_requests=True,
    )


def _route_signature(route_plan: RoutePlan) -> str:
    payload = [
        {
            "vehicle_id": str(route.vehicle_id),
            "customer_ids": [str(cid) for cid in route.customer_ids],
        }
        for route in route_plan.vehicle_routes
    ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _keys(engine: Step7RouteEngine, route_plan: RoutePlan) -> tuple[float, ...]:
    encoded = engine.encoder.encode(route_plan)
    return tuple(float(value) for value in encoded.keys)


def _validate_candidate(
    engine: Step7RouteEngine,
    route_plan: RoutePlan,
):
    candidate = engine.evaluate_keys(_keys(engine, route_plan), repair=True)

    if not candidate.feasible:
        return None

    if (
        candidate.repair_result is not None
        and not candidate.repair_result.complete
    ):
        return None

    return candidate.repaired_plan


def _rebuild_plan(
    route_plan: RoutePlan,
    sequences: Sequence[Sequence[object]],
) -> RoutePlan:
    routes = tuple(
        VehicleRoute.from_sequence(
            route.vehicle_id,
            tuple(sequence),
        )
        for route, sequence in zip(route_plan.vehicle_routes, sequences)
    )
    return RoutePlan.from_routes(routes)


def _perturb(
    route_plan: RoutePlan,
    rng: random.Random,
    operator: str,
) -> RoutePlan | None:
    sequences = [list(route.customer_ids) for route in route_plan.vehicle_routes]

    nonempty = [i for i, sequence in enumerate(sequences) if sequence]
    if not nonempty:
        return None

    if operator == "relocate":
        source = rng.choice(nonempty)
        destination = rng.randrange(len(sequences))
        position = rng.randrange(len(sequences[source]))
        customer = sequences[source].pop(position)

        destination_position = rng.randrange(len(sequences[destination]) + 1)
        sequences[destination].insert(destination_position, customer)

    elif operator == "swap":
        first = rng.choice(nonempty)
        second = rng.choice(nonempty)
        first_pos = rng.randrange(len(sequences[first]))
        second_pos = rng.randrange(len(sequences[second]))

        sequences[first][first_pos], sequences[second][second_pos] = (
            sequences[second][second_pos],
            sequences[first][first_pos],
        )

    elif operator == "reverse":
        vehicle = rng.choice(nonempty)
        sequence = sequences[vehicle]

        if len(sequence) < 3:
            return None

        first, second = sorted(rng.sample(range(len(sequence)), 2))
        if first == second:
            return None

        sequence[first : second + 1] = reversed(sequence[first : second + 1])

    else:
        raise ValueError(f"Unknown perturbation operator: {operator}")

    try:
        return _rebuild_plan(route_plan, sequences)
    except Exception:
        return None


def _constructive_solutions(
    scenario: Scenario,
    optimizer_seed: int,
) -> list[tuple[RoutePlan, str, int]]:
    """
    Generate independent feasible seeds.

    The solver is deterministic for a fixed request order. We therefore vary
    request order in three documented ways for each constructive heuristic.
    """
    rng = random.Random(
        int(optimizer_seed) * 1_000_003 + int(scenario.seeds.scenario_seed)
    )

    request_ids = [request.request_id for request in scenario.requests]
    solutions = []

    for heuristic_index, heuristic in enumerate(HEURISTICS):
        for variant in range(3):
            ordering = list(request_ids)

            if variant == 0:
                rng.shuffle(ordering)
            elif variant == 1:
                ordering.sort(key=str, reverse=True)
            else:
                ordering.sort(
                    key=lambda cid: (
                        float(next(
                            request.demand
                            for request in scenario.requests
                            if request.request_id == cid
                        )),
                        str(cid),
                    ),
                    reverse=True,
                )

            engine = _fresh_engine(scenario)
            builder = InitialSolutionBuilder(
                scenario,
                engine.evaluator,
                config=_constructive_config(heuristic),
            )

            result = builder.build(request_ids=ordering)

            if (
                result.feasible
                and result.complete
                and not result.unassigned_customer_ids
            ):
                solutions.append(
                    (
                        result.route_plan,
                        heuristic,
                        variant + heuristic_index * 10,
                    )
                )

    return solutions


def build_population(
    scenario: Scenario,
    population_size: int,
    optimizer_seed: int,
) -> tuple[tuple[tuple[float, ...], ...], dict[str, object]]:
    """
    Build population by phenotype-first selection.

    We preserve independent constructive solutions first, then generate
    route-level mutations and pass every mutation through the actual Step-7
    repair/evaluation path. Duplicate repaired phenotypes are discarded.
    """
    seen: set[str] = set()
    members: list[PopulationMember] = []
    route_plans: list[RoutePlan] = []

    constructive = _constructive_solutions(
        scenario,
        optimizer_seed,
    )

    def consider(
        plan: RoutePlan,
        heuristic: str,
        variant: int,
        operation: str,
        parent_index: int | None,
    ) -> bool:
        engine = _fresh_engine(scenario)
        repaired = _validate_candidate(engine, plan)

        if repaired is None:
            return False

        signature = _route_signature(repaired)
        if signature in seen:
            return False

        seen.add(signature)
        route_plans.append(repaired)

        members.append(
            PopulationMember(
                optimizer_seed=int(optimizer_seed),
                heuristic=heuristic,
                construction_variant=int(variant),
                operation=operation,
                parent_index=parent_index,
                route_signature=signature,
                keys=_keys(engine, repaired),
            )
        )
        return True

    # Phase A: independent constructive phenotypes.
    for plan, heuristic, variant in constructive:
        consider(
            plan,
            heuristic,
            variant,
            "constructive",
            None,
        )

        if len(members) >= population_size:
            break

    # Phase B: route-level mutation until the population is full.
    #
    # Parent order is shuffled every round, but the resulting population is
    # still deterministic because the RNG seed is explicit.
    for round_index in range(60):
        if len(members) >= population_size:
            break

        parent_indices = list(range(len(route_plans)))
        rng = random.Random(
            int(optimizer_seed) * 7_919
            + round_index * 104_729
            + int(scenario.seeds.scenario_seed)
        )
        rng.shuffle(parent_indices)

        for parent_index in parent_indices:
            if len(members) >= population_size:
                break

            parent = route_plans[parent_index]

            for operator in PERTURBATION_OPERATORS:
                attempts = max(
                    8,
                    min(40, (population_size - len(members)) * 4),
                )

                for _ in range(attempts):
                    if len(members) >= population_size:
                        break

                    child = _perturb(parent, rng, operator)
                    if child is None:
                        continue

                    parent_meta = members[parent_index]

                    consider(
                        child,
                        parent_meta.heuristic,
                        parent_meta.construction_variant,
                        operator,
                        parent_index,
                    )

    if len(members) < population_size:
        raise RuntimeError(
            "R2 initialization failed scientifically: only "
            f"{len(members)}/{population_size} UNIQUE feasible route "
            "phenotypes were found. Duplicate padding is intentionally "
            "disabled."
        )

    selected = members[:population_size]
    selected_plans = route_plans[:population_size]

    distances = []
    for i in range(len(selected_plans)):
        for j in range(i + 1, len(selected_plans)):
            distances.append(
                float(
                    route_distance(
                        selected_plans[i],
                        selected_plans[j],
                    ).exact
                )
            )

    diversity = (
        sum(distances) / len(distances)
        if distances
        else 0.0
    )

    diagnostics = {
        "optimizer_seed": int(optimizer_seed),
        "requested_population": int(population_size),
        "unique_feasible_population": len(selected),
        "feasible_fraction": 1.0,
        "route_diversity_mean_pairwise_exact": diversity,
        "heuristic_counts": {
            heuristic: sum(
                member.heuristic == heuristic
                for member in selected
            )
            for heuristic in HEURISTICS
        },
        "operation_counts": {
            operation: sum(
                member.operation == operation
                for member in selected
            )
            for operation in ("constructive", *PERTURBATION_OPERATORS)
        },
    }

    return (
        tuple(member.keys for member in selected),
        diagnostics,
    )


def _write_population_manifest(
    artifact_dir: Path,
    seed: int,
    population: Sequence[Sequence[float]],
    diagnostics: dict[str, object],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "protocol": "step8_r2_diversified_feasible_initialization",
        "scenario_seed": SCENARIO_SEED,
        "optimizer_seed": int(seed),
        "diagnostics": diagnostics,
        "population_keys": [
            [float(value) for value in vector]
            for vector in population
        ],
    }

    (artifact_dir / f"initial_population_seed_{seed}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def make_population_factory(
    scenario: Scenario,
    population_size: int,
    artifact_dir: Path,
):
    cache: dict[int, tuple[tuple[float, ...], ...]] = {}

    def factory(seed: int):
        seed = int(seed)

        if seed not in cache:
            population, diagnostics = build_population(
                scenario,
                population_size,
                seed,
            )

            cache[seed] = population

            _write_population_manifest(
                artifact_dir,
                seed,
                population,
                diagnostics,
            )

            print(
                f"  seed={seed}: "
                f"{diagnostics['unique_feasible_population']}/"
                f"{population_size} unique feasible | "
                f"route_diversity="
                f"{diagnostics['route_diversity_mean_pairwise_exact']:.4f}"
            )

        return cache[seed]

    return factory


def make_oracle_factory(scenario: Scenario):
    """
    Create a fresh RouteFitnessOracle for every optimizer condition/seed.

    A fresh engine/evaluator keeps evaluation accounting isolated between
    matched runs.
    """
    def factory(seed: int) -> FitnessOracle:
        del seed
        engine = _fresh_engine(scenario)
        return RouteFitnessOracle(
            engine,
            repair=True,
            repair_penalty=0.0,
        )

    return factory


def _base_qpso_config(
    scenario: Scenario,
    population_size: int,
    evaluations: int,
) -> AdaptiveQPSOConfig:
    return AdaptiveQPSOConfig(
        dimensions=2 * len(scenario.requests),
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=population_size,
        max_evaluations=evaluations,
        seed=1,
        tolerance=0.0,
        stagnation_patience=50,
    )


def run(args: argparse.Namespace) -> None:
    scenario = generate_dynamic_scenario(
        _scenario_config(args)
    )

    smoke = args.evaluations < 100 or len(args.seeds) < 3
    artifact_dir = ARTIFACT_ROOT / (
        "r2_qpso_vs_pso_smoke" if smoke else "r2_qpso_vs_pso"
    )

    print()
    print("Step 8 R2 — QPSO vs matched PSO")
    print("=" * 78)
    print(
        f"Scenario: {scenario.scenario_id} | "
        f"{len(scenario.requests)} customers | "
        f"{len(scenario.fleet)} vehicles | "
        f"{len(scenario.nodes)} nodes | "
        f"{len(scenario.edges)} edges"
    )
    print(
        f"Population: {args.population} | "
        f"Evaluation budget: {args.evaluations} | "
        f"Seeds: {len(args.seeds)}"
    )
    print("QPSO condition: canonical")
    print("Initialization: independent constructive + route-level diversity")
    print(
        "Stage: SMOKE — not final research evidence"
        if smoke
        else "Stage: RESEARCH RUN"
    )

    print()
    print("Initial-population diagnostics")
    print("-" * 78)

    population_factory = make_population_factory(
        scenario,
        args.population,
        artifact_dir,
    )

    # Force the initialization gate before any optimizer evaluation.
    for seed in args.seeds:
        population_factory(seed)

    base_config = _base_qpso_config(
        scenario,
        args.population,
        args.evaluations,
    )

    runs, summaries = run_ablation_suite(
        base_qpso_config=base_config,
        oracle_factory=make_oracle_factory(scenario),
        seeds=args.seeds,
        conditions=("canonical",),
        include_pso=True,
        initial_population_factory=population_factory,
    )

    write_results(
        artifact_dir,
        runs,
        summaries,
    )

    print()
    print("Results")
    print("-" * 78)

    for summary in summaries:
        print(
            f"{summary.algorithm}/{summary.condition}: "
            f"runs={summary.runs} | "
            f"best_mean={summary.best_fitness_mean:.6f} | "
            f"best_std={summary.best_fitness_std:.6f} | "
            f"feasible_rate={summary.feasible_rate:.3f} | "
            f"eval_mean={summary.evaluations_mean:.1f} | "
            f"iterations_mean={summary.iterations_mean:.2f}"
        )

    print()
    print(f"Artifacts: {artifact_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--customers",
        type=int,
        default=DEFAULT_CUSTOMERS,
    )
    parser.add_argument(
        "--vehicles",
        type=int,
        default=DEFAULT_VEHICLES,
    )
    parser.add_argument(
        "--road-junctions",
        type=int,
        default=DEFAULT_ROAD_JUNCTIONS,
    )
    parser.add_argument(
        "--population",
        type=int,
        default=DEFAULT_POPULATION,
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=DEFAULT_EVALUATIONS,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
