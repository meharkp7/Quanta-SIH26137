from __future__ import annotations

import pytest

from src.optim.common import FitnessResult
from src.optim.instrumentation import (
    CandidateObservation,
    InstrumentationError,
    OptimizationTraceRecorder,
    candidate_observation_from_fitness,
    observation_from_candidates,
)
from src.routing.route_plan import RoutePlan, VehicleRoute


def plan(*routes):
    return RoutePlan.from_routes(
        VehicleRoute.from_sequence(vehicle_id, customers)
        for vehicle_id, customers in routes
    )


def candidate(
    *,
    position,
    decoded,
    repaired,
    fitness,
    feasible=True,
    repair_distance=0.0,
):
    return CandidateObservation(
        position=tuple(position),
        decoded_plan=decoded,
        repaired_plan=repaired,
        fitness=fitness,
        feasible=feasible,
        repair_distance=repair_distance,
    )


def test_candidate_observation_is_immutable():
    observation = candidate(
        position=(0.1, 0.2),
        decoded=plan((1, (101,)), (2, (102,))),
        repaired=plan((1, (101,)), (2, (102,))),
        fitness=10.0,
    )

    with pytest.raises(AttributeError):
        observation.fitness = 5.0


def test_observation_computes_population_statistics():
    route_a = plan(
        (1, (101, 102)),
        (2, (103,)),
    )

    route_b = plan(
        (1, (101,)),
        (2, (102, 103)),
    )

    observation = observation_from_candidates(
        iteration=3,
        evaluations=20,
        candidates=(
            candidate(
                position=(0.1, 0.2),
                decoded=route_a,
                repaired=route_a,
                fitness=20.0,
                feasible=True,
            ),
            candidate(
                position=(0.9, 0.8),
                decoded=route_b,
                repaired=route_b,
                fitness=10.0,
                feasible=False,
            ),
        ),
    )

    assert observation.iteration == 3
    assert observation.evaluations == 20
    assert observation.best_fitness == 10.0
    assert observation.mean_fitness == 15.0
    assert observation.best_index == 1
    assert observation.feasible_rate == pytest.approx(0.5)

    assert observation.diversity.genotype_coordinate > 0.0
    assert observation.diversity.decoded_assignment > 0.0


def test_repair_pressure_is_measured_from_plans():
    decoded_a = plan(
        (1, (101, 102)),
        (2, (103,)),
    )

    decoded_b = plan(
        (1, (101,)),
        (2, (102, 103)),
    )

    repaired_a = decoded_a

    repaired_b = plan(
        (1, (101, 102)),
        (2, (103,)),
    )

    observation = observation_from_candidates(
        iteration=1,
        evaluations=2,
        candidates=(
            candidate(
                position=(0.1, 0.2),
                decoded=decoded_a,
                repaired=repaired_a,
                fitness=10.0,
            ),
            candidate(
                position=(0.8, 0.9),
                decoded=decoded_b,
                repaired=repaired_b,
                fitness=20.0,
            ),
        ),
    )

    assert observation.repair_pressure.assignment > 0.0
    assert observation.repair_pressure.structure > 0.0


def test_recorder_requires_monotonic_iterations():
    recorder = OptimizationTraceRecorder()

    route = plan(
        (1, (101,)),
        (2, (102,)),
    )

    observation = candidate(
        position=(0.1, 0.2),
        decoded=route,
        repaired=route,
        fitness=10.0,
    )

    recorder.record(
        iteration=1,
        evaluations=2,
        candidates=(observation,),
    )

    with pytest.raises(InstrumentationError):
        recorder.record(
            iteration=1,
            evaluations=3,
            candidates=(observation,),
        )


def test_recorder_requires_non_decreasing_evaluations():
    recorder = OptimizationTraceRecorder()

    route = plan(
        (1, (101,)),
        (2, (102,)),
    )

    observation = candidate(
        position=(0.1, 0.2),
        decoded=route,
        repaired=route,
        fitness=10.0,
    )

    recorder.record(
        iteration=1,
        evaluations=5,
        candidates=(observation,),
    )

    with pytest.raises(InstrumentationError):
        recorder.record(
            iteration=2,
            evaluations=4,
            candidates=(observation,),
        )


def test_trace_snapshot_is_immutable():
    recorder = OptimizationTraceRecorder()

    route = plan(
        (1, (101,)),
        (2, (102,)),
    )

    observation = candidate(
        position=(0.1, 0.2),
        decoded=route,
        repaired=route,
        fitness=10.0,
    )

    recorder.record(
        iteration=1,
        evaluations=2,
        candidates=(observation,),
    )

    trace = recorder.snapshot()

    assert trace.iterations == (1,)
    assert trace.best_fitness == (10.0,)
    assert trace.mean_fitness == (10.0,)


def test_fitness_result_adapter_does_not_re_evaluate():
    route = plan(
        (1, (101,)),
        (2, (102,)),
    )

    result = FitnessResult(
        fitness=42.0,
        feasible=True,
        repair_distance=0.25,
    )

    observation = candidate_observation_from_fitness(
        position=(0.2, 0.4),
        decoded_plan=route,
        repaired_plan=route,
        result=result,
    )

    assert observation.position == (0.2, 0.4)
    assert observation.fitness == 42.0
    assert observation.feasible is True
    assert observation.repair_distance == 0.25


def test_invalid_candidate_fitness_is_rejected():
    route = plan(
        (1, (101,)),
        (2, (102,)),
    )

    with pytest.raises(InstrumentationError):
        candidate(
            position=(0.1, 0.2),
            decoded=route,
            repaired=route,
            fitness=float("nan"),
        )


def test_invalid_repair_distance_is_rejected():
    route = plan(
        (1, (101,)),
        (2, (102,)),
    )

    with pytest.raises(InstrumentationError):
        candidate(
            position=(0.1, 0.2),
            decoded=route,
            repaired=route,
            fitness=10.0,
            repair_distance=-1.0,
        )


def test_clear_does_not_change_existing_snapshot():
    recorder = OptimizationTraceRecorder()

    route = plan(
        (1, (101,)),
        (2, (102,)),
    )

    observation = candidate(
        position=(0.1, 0.2),
        decoded=route,
        repaired=route,
        fitness=10.0,
    )

    recorder.record(
        iteration=1,
        evaluations=2,
        candidates=(observation,),
    )

    before_clear = recorder.snapshot()

    recorder.clear()

    after_clear = recorder.snapshot()

    assert before_clear.iterations == (1,)
    assert after_clear.iterations == ()