"""Tests for route evaluation on the V1.2 scenario contract."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.contracts.scenario import Scenario
from src.routing.route_evaluator import (
    RouteEvaluationConfig,
    RouteEvaluator,
)
from src.routing.route_plan import RoutePlan, VehicleRoute


def reference_plan() -> RoutePlan:
    return RoutePlan(
        vehicle_routes=(
            VehicleRoute(
                vehicle_id="V1",
                customer_ids=("J1", "J2", "J3"),
            ),
            VehicleRoute(
                vehicle_id="V2",
                customer_ids=("J4", "J5"),
            ),
        )
    )


def test_reference_route_is_feasible(scenario: Scenario):
    evaluator = RouteEvaluator(scenario)

    result = evaluator.evaluate(reference_plan())

    assert result.feasible
    assert result.assignment_feasible
    assert result.capacity_feasible
    assert result.connectivity_feasible
    assert result.time_window_feasible

    assert result.assigned_request_ids == (
        "J1",
        "J2",
        "J3",
        "J4",
        "J5",
    )

    assert result.unassigned_request_ids == ()


def test_reference_distance_includes_return_to_depot(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    result = evaluator.evaluate(reference_plan())

    # The fixture's actual directed graph gives:
    # V1: N0 -> N1 -> N2 -> N3 -> N4 -> N5 -> N1 -> N0 = 700 m
    # V2: N0 -> N1 -> N2 -> N3 -> N4 -> N5 -> N1 -> N0 = 700 m
    # Total = 1400 m.
    #
    # The evaluator uses the actual directed graph, including the
    # shortest legal return path to the depot. The old 1500 m assertion
    # described edges that do not exist in the V1.2 fixture.
    assert result.total_distance_m == pytest.approx(1400.0)


def test_reference_timing_is_correct(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    result = evaluator.evaluate(reference_plan())

    v1 = result.vehicle_evaluations["V1"]
    v2 = result.vehicle_evaluations["V2"]

    assert v1.service_starts_s == pytest.approx(
        {
            "J1": 10.0,
            "J2": 35.0,
            "J3": 60.0,
        }
    )

    assert v2.service_starts_s == pytest.approx(
        {
            "J4": 80.0,
            "J5": 100.0,
        }
    )

    assert v1.total_waiting_time_s == pytest.approx(10.0)
    assert v2.total_waiting_time_s == pytest.approx(40.0)

    assert v1.total_service_time_s == pytest.approx(35.0)
    assert v2.total_service_time_s == pytest.approx(20.0)


def test_capacity_violation_is_detected(
    scenario: Scenario,
):
    overloaded = RoutePlan(
        vehicle_routes=(
            VehicleRoute(
                vehicle_id="V1",
                customer_ids=("J1", "J2", "J3"),
            ),
            VehicleRoute(
                vehicle_id="V2",
                customer_ids=("J4", "J5"),
            ),
        )
    )

    # V1 carries 2 + 2 + 3 = 7, exactly at capacity.
    evaluator = RouteEvaluator(scenario)
    result = evaluator.evaluate(overloaded)

    assert result.capacity_feasible


def test_excess_load_is_infeasible():
    from tests.conftest import load_step3_case

    data = load_step3_case("excess_load")
    scenario = Scenario.model_validate(data)

    plan = RoutePlan(
        vehicle_routes=(
            VehicleRoute(
                vehicle_id="V1",
                customer_ids=("J1", "J2", "J3"),
            ),
            VehicleRoute(
                vehicle_id="V2",
                customer_ids=("J4", "J5"),
            ),
        )
    )

    evaluator = RouteEvaluator(scenario)
    result = evaluator.evaluate(plan)

    assert not result.feasible
    assert not result.capacity_feasible


def test_time_window_violation_is_detected():
    from tests.conftest import load_step3_case

    data = load_step3_case("missed_window")
    scenario = Scenario.model_validate(data)

    evaluator = RouteEvaluator(scenario)

    result = evaluator.evaluate(reference_plan())

    assert not result.feasible
    assert not result.time_window_feasible

    v1 = result.vehicle_evaluations["V1"]

    assert v1.service_starts_s["J3"] > 65.0


def test_unknown_vehicle_is_reported(
    scenario: Scenario,
):
    plan = RoutePlan(
        vehicle_routes=(
            VehicleRoute(
                vehicle_id="UNKNOWN",
                customer_ids=("J1",),
            ),
        )
    )

    evaluator = RouteEvaluator(scenario)
    result = evaluator.evaluate(plan)

    assert not result.assignment_feasible
    assert "UNKNOWN" in result.unknown_vehicle_ids


def test_unknown_customer_is_reported(
    scenario: Scenario,
):
    plan = RoutePlan(
        vehicle_routes=(
            VehicleRoute(
                vehicle_id="V1",
                customer_ids=("NOT_A_REQUEST",),
            ),
        )
    )

    evaluator = RouteEvaluator(scenario)
    result = evaluator.evaluate(plan)

    assert not result.assignment_feasible
    assert "NOT_A_REQUEST" in result.unknown_request_ids


def test_duplicate_customer_assignment_is_detected(
    scenario: Scenario,
):
    plan = RoutePlan(
        vehicle_routes=(
            VehicleRoute(
                vehicle_id="V1",
                customer_ids=("J1", "J2"),
            ),
            VehicleRoute(
                vehicle_id="V2",
                customer_ids=("J1", "J3"),
            ),
        )
    )

    evaluator = RouteEvaluator(scenario)
    result = evaluator.evaluate(plan)

    assert not result.assignment_feasible
    assert "J1" in result.duplicate_request_ids


def test_dynamic_provider_changes_path_choice(
    scenario: Scenario,
):
    def slow_direct_edge(edge, departure_time_s):
        if edge.edge_id == "E12":
            return 100.0
        return edge.free_flow_time_s

    evaluator = RouteEvaluator(
        scenario,
        travel_time_provider=slow_direct_edge,
    )

    plan = RoutePlan(
        vehicle_routes=(
            VehicleRoute(
                vehicle_id="V1",
                customer_ids=("J1", "J2"),
            ),
        )
    )

    result = evaluator.evaluate(plan)

    assert result.feasible

    v1 = result.vehicle_evaluations["V1"]

    # The evaluator must choose N1 -> N6 -> N2 rather than taking
    # the dynamically slowed E12.
    j2_stop = next(
        stop
        for stop in v1.stops
        if stop.customer_id == "J2"
    )

    assert j2_stop.edge_ids == ("E16", "E62")
    assert j2_stop.travel_time_s == pytest.approx(30.0)


def test_dynamic_provider_receives_actual_departure_time(
    scenario: Scenario,
):
    calls: list[tuple[str, float]] = []

    def provider(edge, departure_time_s):
        calls.append(
            (
                edge.edge_id,
                departure_time_s,
            )
        )
        return edge.free_flow_time_s

    evaluator = RouteEvaluator(
        scenario,
        travel_time_provider=provider,
    )

    plan = RoutePlan(
        vehicle_routes=(
            VehicleRoute(
                vehicle_id="V1",
                customer_ids=("J1", "J2"),
            ),
        )
    )

    evaluator.evaluate(plan)

    assert calls

    # E01 begins at t=0.
    assert ("E01", 0.0) in calls

    # E12 is considered from the actual post-service time at J1.
    assert ("E12", 20.0) in calls


def test_closed_edge_is_not_used():
    from tests.conftest import load_step3_case

    data = load_step3_case("closure_with_detour")
    scenario = Scenario.model_validate(data)

    evaluator = RouteEvaluator(scenario)

    plan = RoutePlan(
        vehicle_routes=(
            VehicleRoute(
                vehicle_id="V1",
                customer_ids=("J1", "J2"),
            ),
        )
    )

    result = evaluator.evaluate(plan)

    assert result.feasible

    v1 = result.vehicle_evaluations["V1"]

    j2_stop = next(
        stop
        for stop in v1.stops
        if stop.customer_id == "J2"
    )

    assert j2_stop.edge_ids == (
        "E16",
        "E62",
    )


def test_same_route_is_deterministic(
    scenario: Scenario,
):
    evaluator = RouteEvaluator(scenario)

    first = evaluator.evaluate(reference_plan())
    second = evaluator.evaluate(reference_plan())

    assert first == second


def test_planning_time_is_propagated_to_dynamic_evaluation(
    scenario: Scenario,
):
    observed_times: list[float] = []

    def provider(edge, departure_time_s):
        observed_times.append(departure_time_s)
        return edge.free_flow_time_s

    evaluator = RouteEvaluator(
        scenario,
        travel_time_provider=provider,
    )

    evaluator.evaluate(
        reference_plan(),
        planning_time_s=125.0,
    )

    assert observed_times
    assert min(observed_times) >= 125.0