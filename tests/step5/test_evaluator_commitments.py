from __future__ import annotations

import pytest

from src.routing.evaluator_state import (
    CommitmentSnapshot,
    EvaluationConstraints,
    VehicleCommitment,
)
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_plan import RoutePlan, VehicleRoute


def _plan(vehicle_id: str, *customer_ids: str) -> RoutePlan:
    return RoutePlan.from_routes(
        (
            VehicleRoute.from_sequence(
                vehicle_id=vehicle_id,
                customer_ids=customer_ids,
            ),
        )
    )


def _commitment(
    vehicle_id: str,
    *,
    current_node_id: str,
    current_time_s: float,
    current_load_units: float = 0.0,
    onboard_request_ids: tuple[str, ...] = (),
    frozen_prefix_edge_ids: tuple[str, ...] = (),
    committed_customer_ids: tuple[str, ...] = (),
) -> CommitmentSnapshot:
    return CommitmentSnapshot(
        vehicles=(
            VehicleCommitment(
                vehicle_id=vehicle_id,
                current_node_id=current_node_id,
                current_time_s=current_time_s,
                current_load_units=current_load_units,
                onboard_request_ids=onboard_request_ids,
                frozen_prefix_edge_ids=frozen_prefix_edge_ids,
                committed_customer_ids=committed_customer_ids,
            ),
        )
    )


def test_evaluation_constraints_are_immutable_and_validate_tolerances():
    constraints = EvaluationConstraints(
        enforce_capacity=True,
        enforce_time_windows=True,
        enforce_releases=True,
        require_depot_return=True,
        require_commitments=True,
        allow_waiting=True,
        require_all_requests_served=False,
    )

    assert constraints.enforce_capacity is True
    assert constraints.require_commitments is True
    assert constraints.allow_waiting is True

    with pytest.raises(ValueError):
        EvaluationConstraints(load_tolerance=-1.0)

    with pytest.raises(ValueError):
        EvaluationConstraints(time_tolerance=float("inf"))


def test_vehicle_commitment_rejects_invalid_state():
    with pytest.raises(ValueError):
        VehicleCommitment(
            vehicle_id="V1",
            current_node_id="N0",
            current_time_s=-1.0,
            current_load_units=0.0,
        )

    with pytest.raises(ValueError):
        VehicleCommitment(
            vehicle_id="V1",
            current_node_id="N0",
            current_time_s=0.0,
            current_load_units=-1.0,
        )


def test_vehicle_commitment_rejects_duplicate_state_members():
    with pytest.raises(ValueError):
        VehicleCommitment(
            vehicle_id="V1",
            current_node_id="N0",
            current_time_s=0.0,
            current_load_units=0.0,
            onboard_request_ids=("J1", "J1"),
        )

    with pytest.raises(ValueError):
        VehicleCommitment(
            vehicle_id="V1",
            current_node_id="N0",
            current_time_s=0.0,
            current_load_units=0.0,
            frozen_prefix_edge_ids=("E1", "E1"),
        )

    with pytest.raises(ValueError):
        VehicleCommitment(
            vehicle_id="V1",
            current_node_id="N0",
            current_time_s=0.0,
            current_load_units=0.0,
            committed_customer_ids=("J1", "J1"),
        )


def test_vehicle_commitment_rejects_onboard_future_commitment_overlap():
    with pytest.raises(ValueError):
        VehicleCommitment(
            vehicle_id="V1",
            current_node_id="N0",
            current_time_s=0.0,
            current_load_units=1.0,
            onboard_request_ids=("J1",),
            committed_customer_ids=("J1",),
        )


def test_commitment_snapshot_rejects_duplicate_vehicle_commitments():
    commitment = VehicleCommitment(
        vehicle_id="V1",
        current_node_id="N0",
        current_time_s=0.0,
        current_load_units=0.0,
    )

    with pytest.raises(ValueError):
        CommitmentSnapshot(
            vehicles=(commitment, commitment)
        )


def test_commitment_snapshot_lookup_is_deterministic():
    first = VehicleCommitment(
        vehicle_id="V1",
        current_node_id="N1",
        current_time_s=10.0,
        current_load_units=2.0,
    )

    second = VehicleCommitment(
        vehicle_id="V2",
        current_node_id="N2",
        current_time_s=20.0,
        current_load_units=3.0,
    )

    snapshot = CommitmentSnapshot(
        vehicles=(first, second)
    )

    assert snapshot.for_vehicle("V1") == first
    assert snapshot.for_vehicle("V2") == second
    assert snapshot.for_vehicle("UNKNOWN") is None


def test_commitment_snapshot_from_scenario_preserves_evaluation_time(scenario):
    snapshot = CommitmentSnapshot.from_scenario(
        scenario.fleet,
        default_time_s=137.5,
    )

    assert snapshot.vehicles

    for commitment in snapshot.vehicles:
        assert commitment.current_time_s == pytest.approx(137.5)


def test_evaluator_accepts_explicit_constraints_and_commitments(scenario):
    evaluator = RouteEvaluator(scenario)

    vehicle = scenario.fleet[0]
    vehicle_id = vehicle.vehicle_id
    start_node = vehicle.start_node_id

    snapshot = _commitment(
        vehicle_id,
        current_node_id=start_node,
        current_time_s=25.0,
        current_load_units=0.0,
    )

    result = evaluator.evaluate(
        _plan(vehicle_id),
        planning_time_s=25.0,
        constraints=EvaluationConstraints(
            require_all_requests_served=False,
        ),
        commitments=snapshot,
    )

    assert result is not None
    assert result.commitment_feasible is True


def test_evaluator_uses_commitment_current_time(scenario):
    evaluator = RouteEvaluator(scenario)

    vehicle = scenario.fleet[0]
    vehicle_id = vehicle.vehicle_id
    start_node = vehicle.start_node_id

    snapshot = _commitment(
        vehicle_id,
        current_node_id=start_node,
        current_time_s=500.0,
    )

    result = evaluator.evaluate(
        _plan(vehicle_id),
        planning_time_s=500.0,
        commitments=snapshot,
    )

    vehicle_result = next(
        item
        for item in result.vehicle_evaluations
        if item.vehicle_id == vehicle_id
    )

    assert vehicle_result is not None

    if vehicle_result.stops:
        first_stop = vehicle_result.stops[0]
        assert first_stop.arrival_time_s >= 500.0


def test_evaluator_uses_commitment_current_load(scenario):
    evaluator = RouteEvaluator(scenario)

    vehicle = scenario.fleet[0]
    vehicle_id = vehicle.vehicle_id
    start_node = vehicle.start_node_id

    snapshot = _commitment(
        vehicle_id,
        current_node_id=start_node,
        current_time_s=0.0,
        current_load_units=1.0,
    )

    result = evaluator.evaluate(
        _plan(vehicle_id),
        planning_time_s=0.0,
        commitments=snapshot,
    )

    vehicle_result = next(
        item
        for item in result.vehicle_evaluations
        if item.vehicle_id == vehicle_id
    )

    assert vehicle_result is not None
    assert vehicle_result.maximum_load_units >= 1.0


def test_committed_customer_prefix_is_enforced(scenario):
    evaluator = RouteEvaluator(scenario)

    vehicle = scenario.fleet[0]
    vehicle_id = vehicle.vehicle_id
    start_node = vehicle.start_node_id

    request_ids = [
        request.request_id
        for request in scenario.requests
    ]

    if len(request_ids) < 2:
        pytest.skip("Fixture needs at least two requests")

    committed_first = request_ids[0]
    wrong_first = request_ids[1]

    snapshot = _commitment(
        vehicle_id,
        current_node_id=start_node,
        current_time_s=0.0,
        committed_customer_ids=(committed_first,),
    )

    result = evaluator.evaluate(
        _plan(
            vehicle_id,
            wrong_first,
        ),
        planning_time_s=0.0,
        commitments=snapshot,
    )

    assert result.commitment_feasible is False

    names = {
        violation.name
        for violation in result.violations
    }

    assert "committed_customer_prefix" in names


def test_matching_committed_customer_prefix_is_accepted(scenario):
    evaluator = RouteEvaluator(scenario)

    vehicle = scenario.fleet[0]
    vehicle_id = vehicle.vehicle_id
    start_node = vehicle.start_node_id

    request_ids = [
        request.request_id
        for request in scenario.requests
    ]

    if not request_ids:
        pytest.skip("Fixture has no requests")

    committed_first = request_ids[0]

    snapshot = _commitment(
        vehicle_id,
        current_node_id=start_node,
        current_time_s=0.0,
        committed_customer_ids=(committed_first,),
    )

    result = evaluator.evaluate(
        _plan(
            vehicle_id,
            committed_first,
        ),
        planning_time_s=0.0,
        commitments=snapshot,
    )

    assert result.commitment_feasible is True

def test_structured_violations_are_exposed(scenario):
    evaluator = RouteEvaluator(scenario)

    vehicle = scenario.fleet[0]

    result = evaluator.evaluate(
        _plan(
            vehicle.vehicle_id,
            "NOT_A_REAL_REQUEST",
        ),
        planning_time_s=0.0,
        constraints=EvaluationConstraints(
            require_all_requests_served=False,
        ),
    )

    assert result.feasible is False
    assert result.violations

    assert all(
        violation.name
        for violation in result.violations
    )

    assert any(
        violation.name == "unknown_request"
        for violation in result.violations
    )


def test_vehicle_evaluation_exposes_structured_violations(scenario):
    evaluator = RouteEvaluator(scenario)

    vehicle = scenario.fleet[0]

    result = evaluator.evaluate_vehicle_route(
        _plan(
            vehicle.vehicle_id,
            "NOT_A_REAL_REQUEST",
        ).route_for(vehicle.vehicle_id),
        planning_time_s=0.0,
    )

    assert result.feasible is False
    assert result.violations
    assert any(
        violation.name == "unknown_request"
        for violation in result.violations
    )


def test_evaluator_does_not_require_all_requests_during_partial_construction(
    scenario,
):
    evaluator = RouteEvaluator(scenario)

    vehicle = scenario.fleet[0]

    result = evaluator.evaluate(
        _plan(vehicle.vehicle_id),
        planning_time_s=0.0,
        constraints=EvaluationConstraints(
            require_all_requests_served=False,
        ),
    )

    assert result.all_requests_served is False
    assert result.assignment_feasible is False


def test_require_all_requests_served_is_explicit(scenario):
    evaluator = RouteEvaluator(scenario)

    vehicle = scenario.fleet[0]

    result = evaluator.evaluate(
        _plan(vehicle.vehicle_id),
        planning_time_s=0.0,
        constraints=EvaluationConstraints(
            require_all_requests_served=True,
        ),
    )

    assert result.all_requests_served is False
    assert result.assignment_feasible is False