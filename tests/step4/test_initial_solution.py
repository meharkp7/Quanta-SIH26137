"""Tests for deterministic initial route construction."""

from __future__ import annotations

import pytest

from src.contracts.scenario import Scenario
from src.routing.initial_solution import (
    InitialSolutionBuilder,
    InitialSolutionConfig,
)
from src.routing.route_evaluator import RouteEvaluator


def test_initial_solution_assigns_all_requests(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
    )

    result = builder.build()

    assert result.complete
    assert result.feasible

    assigned = set(result.assigned_request_ids)

    assert assigned == {
        request.request_id
        for request in scenario.requests
    }

    assert result.unassigned_request_ids == ()


def test_initial_solution_creates_one_route_per_vehicle(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
    )

    result = builder.build()

    route_vehicle_ids = {
        route.vehicle_id
        for route in result.route_plan.vehicle_routes
    }

    scenario_vehicle_ids = {
        vehicle.vehicle_id
        for vehicle in scenario.fleet
    }

    assert route_vehicle_ids == scenario_vehicle_ids


def test_initial_solution_contains_no_duplicate_requests(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
    )

    result = builder.build()

    all_requests = [
        customer_id
        for route in result.route_plan.vehicle_routes
        for customer_id in route.customer_ids
    ]

    assert len(all_requests) == len(set(all_requests))


def test_initial_solution_is_capacity_feasible(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    config = InitialSolutionConfig(
        require_capacity_feasibility=True,
    )

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
        config=config,
    )

    result = builder.build()

    assert result.evaluation.capacity_feasible


def test_initial_solution_is_connectivity_feasible(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    config = InitialSolutionConfig(
        require_connectivity=True,
    )

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
        config=config,
    )

    result = builder.build()

    assert result.evaluation.connectivity_feasible


def test_initial_solution_is_time_window_feasible(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    config = InitialSolutionConfig(
        require_time_window_feasibility=True,
    )

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
        config=config,
    )

    result = builder.build()

    assert result.evaluation.time_window_feasible


def test_initial_solution_is_deterministic(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    config = InitialSolutionConfig(
        customer_ordering="customer_id",
    )

    first = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
        config=config,
    ).build()

    second = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
        config=config,
    ).build()

    assert first.route_plan == second.route_plan
    assert first.evaluation == second.evaluation
    assert first.assigned_request_ids == second.assigned_request_ids
    assert first.unassigned_request_ids == second.unassigned_request_ids


def test_request_subset_can_be_solved(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
    )

    result = builder.build(
        request_ids=("J1", "J3"),
    )

    assert set(result.assigned_request_ids) == {
        "J1",
        "J3",
    }

    assert result.unassigned_request_ids == ()


def test_planning_time_is_forwarded(
    scenario: Scenario,
):
    observed: list[float] = []

    def provider(edge, departure_time_s):
        observed.append(departure_time_s)
        return edge.free_flow_time_s

    evaluator = RouteEvaluator(
        scenario,
        travel_time_provider=provider,
    )

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
    )

    builder.build(
        planning_time_s=100.0,
    )

    assert observed
    assert min(observed) >= 100.0


def test_unassigned_requests_are_reported_when_allowed(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    config = InitialSolutionConfig(
        allow_unassigned=True,
    )

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
        config=config,
    )

    result = builder.build(
        request_ids=("J1", "J2", "J3", "J4", "J5"),
    )

    assert set(result.assigned_request_ids).issubset(
        {"J1", "J2", "J3", "J4", "J5"}
    )

    assert set(result.assigned_request_ids).isdisjoint(
        set(result.unassigned_request_ids)
    )


def test_invalid_request_id_is_rejected(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
    )

    with pytest.raises(ValueError):
        builder.build(
            request_ids=("J1", "DOES_NOT_EXIST"),
        )


def test_scenario_is_not_mutated(
    scenario: Scenario,
):
    before = scenario.model_dump(mode="python")

    evaluator = RouteEvaluator(scenario)

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
    )

    builder.build()

    after = scenario.model_dump(mode="python")

    assert before == after


def test_trace_is_available(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    builder = InitialSolutionBuilder(
        scenario,
        evaluator=evaluator,
    )

    result = builder.build()

    assert result.construction_trace
    assert result.insertion_count >= 0
    assert result.attempt_count >= result.insertion_count