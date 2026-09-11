from src.routing.route_plan import RoutePlan, VehicleRoute

import pytest

from src.optim.instrumentation import (
    CandidateObservation,
    OptimizationTraceRecorder,
)
from src.optim.predictive_analysis import SearchState
from src.optim.trace_bridge import (
    TraceBridgeError,
    trace_to_search_states,
    trace_to_search_states_strict,
)


def make_candidate(
    position,
    fitness,
    *,
    feasible=True,
    repair_distance=0.1,
):
    plan = RoutePlan.from_routes(
        (
            VehicleRoute.from_sequence(
                vehicle_id=1,
                customer_ids=(101, 102),
            ),
        )
    )

    return CandidateObservation(
        position=tuple(position),
        decoded_plan=plan,
        repaired_plan=plan,
        fitness=fitness,
        feasible=feasible,
        repair_distance=repair_distance,
    )


def make_trace():
    recorder = OptimizationTraceRecorder(
        lower_bound=0.0,
        upper_bound=1.0,
    )

    recorder.record(
        iteration=0,
        evaluations=2,
        candidates=(
            make_candidate(
                (0.1, 0.2),
                100.0,
                repair_distance=0.0,
            ),
            make_candidate(
                (0.8, 0.9),
                110.0,
                feasible=False,
                repair_distance=0.2,
            ),
        ),
    )

    recorder.record(
        iteration=1,
        evaluations=4,
        candidates=(
            make_candidate(
                (0.2, 0.3),
                90.0,
                repair_distance=0.1,
            ),
            make_candidate(
                (0.7, 0.8),
                95.0,
                repair_distance=0.3,
            ),
        ),
    )

    return recorder.snapshot()


def test_trace_is_projected_into_search_states():
    trace = make_trace()

    states = trace_to_search_states(trace)

    assert len(states) == 2
    assert all(isinstance(state, SearchState) for state in states)


def test_trace_projection_preserves_iteration_and_evaluations():
    trace = make_trace()

    states = trace_to_search_states(trace)

    assert tuple(
        state.iteration
        for state in states
    ) == (0, 1)

    assert tuple(
        state.evaluations
        for state in states
    ) == (2, 4)


def test_trace_projection_preserves_best_fitness():
    trace = make_trace()

    states = trace_to_search_states(trace)

    assert tuple(
        state.best_fitness
        for state in states
    ) == (100.0, 90.0)


def test_trace_projection_preserves_diversity_signals():
    trace = make_trace()

    states = trace_to_search_states(trace)

    first = states[0]

    assert first.coordinate_diversity == pytest.approx(
        trace.observations[0].diversity.genotype_coordinate
    )

    assert first.decoded_assignment_diversity == pytest.approx(
        trace.observations[0].diversity.decoded_assignment
    )

    assert first.decoded_precedence_diversity == pytest.approx(
        trace.observations[0].diversity.decoded_precedence
    )

    assert first.decoded_structure_diversity == pytest.approx(
        trace.observations[0].diversity.decoded_structure
    )


def test_trace_projection_uses_repaired_diversity():
    trace = make_trace()

    states = trace_to_search_states(trace)

    first = states[0]

    assert first.repaired_assignment_diversity == pytest.approx(
        trace.observations[0].diversity.repaired_assignment
    )

    assert first.repaired_precedence_diversity == pytest.approx(
        trace.observations[0].diversity.repaired_precedence
    )

    assert first.repaired_structure_diversity == pytest.approx(
        trace.observations[0].diversity.repaired_structure
    )


def test_trace_projection_uses_structural_repair_pressure():
    trace = make_trace()

    states = trace_to_search_states(trace)

    for state, observation in zip(
        states,
        trace.observations,
    ):
        assert state.repair_pressure == pytest.approx(
            observation.repair_pressure.structure
        )


def test_trace_projection_preserves_feasible_rate():
    trace = make_trace()

    states = trace_to_search_states(trace)

    assert states[0].feasible_rate == pytest.approx(
        trace.observations[0].feasible_rate
    )

    assert states[1].feasible_rate == pytest.approx(
        trace.observations[1].feasible_rate
    )


def test_trace_projection_does_not_modify_trace():
    trace = make_trace()

    before = trace

    states = trace_to_search_states(trace)

    assert states
    assert trace == before


def test_empty_trace_returns_empty_states():
    recorder = OptimizationTraceRecorder(
        lower_bound=0.0,
        upper_bound=1.0,
    )

    trace = recorder.snapshot()

    assert trace_to_search_states(trace) == ()


def test_strict_bridge_rejects_empty_trace():
    recorder = OptimizationTraceRecorder(
        lower_bound=0.0,
        upper_bound=1.0,
    )

    trace = recorder.snapshot()

    with pytest.raises(TraceBridgeError):
        trace_to_search_states_strict(trace)