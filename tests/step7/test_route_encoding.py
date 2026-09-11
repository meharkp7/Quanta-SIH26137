from __future__ import annotations

import pytest

from src.routing.evaluator_state import CommitmentSnapshot, VehicleCommitment
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_plan import RoutePlan, VehicleRoute
from src.routing.route_encoding import (
    BoundedRouteRepairer,
    ContinuousRouteEncoder,
    RepairConfig,
    RouteEncodingError,
    RouteEncodingConfig,
    Step7RouteEngine,
)


def _keys(encoder, assignments, priorities):
    return tuple(assignments) + tuple(priorities)


def test_dimension_is_exactly_2n(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    assert encoder.dimension == 2 * len(scenario.requests)


def test_assignment_boundaries_are_deterministic(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    n = len(scenario.requests)
    keys = _keys(encoder, [0.0] * n, [0.5] * n)
    decoded = encoder.decode(keys)
    assert decoded.route_plan.route_for("V1").customer_ids == tuple(
        r.request_id for r in scenario.requests
    )

    keys = _keys(encoder, [1.0] * n, [0.5] * n)
    decoded = encoder.decode(keys)
    assert decoded.route_plan.route_for("V2").stop_count == n


def test_key_changes_produce_distinct_route_candidates(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    n = len(scenario.requests)
    low = encoder.decode(_keys(encoder, [0.0] * n, [0.0, 0.1, 0.2, 0.3, 0.4]))
    high = encoder.decode(_keys(encoder, [1.0] * n, [0.0, 0.1, 0.2, 0.3, 0.4]))
    assert low.route_plan != high.route_plan


def test_order_keys_change_order_without_changing_assignment(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    n = len(scenario.requests)
    assignment = [0.0] * n
    a = encoder.decode(_keys(encoder, assignment, [0.0, 0.1, 0.2, 0.3, 0.4]))
    b = encoder.decode(_keys(encoder, assignment, [0.4, 0.3, 0.2, 0.1, 0.0]))
    assert a.route_plan.route_for("V1").customer_ids != b.route_plan.route_for("V1").customer_ids
    assert set(a.route_plan.route_for("V1").customer_ids) == set(b.route_plan.route_for("V1").customer_ids)


def test_nonfinite_keys_are_rejected(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    with pytest.raises(RouteEncodingError):
        encoder.decode([float("nan")] * encoder.dimension)
    with pytest.raises(RouteEncodingError):
        encoder.decode([float("inf")] * encoder.dimension)


def test_wrong_dimension_is_rejected(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    with pytest.raises(RouteEncodingError):
        encoder.decode([0.0] * (encoder.dimension - 1))


def test_encode_decode_round_trip_is_canonical(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence("V1", ["J3", "J1", "J2"]),
            VehicleRoute.from_sequence("V2", ["J5", "J4"]),
        ]
    )
    encoded = encoder.encode(plan)
    decoded = encoder.decode(encoded.keys)
    assert decoded.route_plan == plan


def test_locked_customer_ignores_assignment_key_and_preserves_prefix(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    commitment = CommitmentSnapshot(
        (
            VehicleCommitment(
                vehicle_id="V1",
                current_node_id="N0",
                current_time_s=0.0,
                current_load_units=0.0,
                committed_customer_ids=("J2", "J1"),
            ),
        )
    )
    n = len(scenario.requests)
    keys = _keys(encoder, [1.0] * n, [0.9, 0.8, 0.7, 0.6, 0.5])
    decoded = encoder.decode(keys, commitments=commitment)
    assert decoded.route_plan.route_for("V1").customer_ids[:2] == ("J2", "J1")
    assert decoded.assignment_for("J2") == "V1"
    assert decoded.assignment_for("J1") == "V1"


def test_locked_customer_cannot_be_encoded_on_another_vehicle(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    commitment = CommitmentSnapshot(
        (
            VehicleCommitment(
                vehicle_id="V1",
                current_node_id="N0",
                current_time_s=0.0,
                current_load_units=0.0,
                committed_customer_ids=("J1",),
            ),
        )
    )
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence("V1", ["J2"]),
            VehicleRoute.from_sequence("V2", ["J1", "J3", "J4", "J5"]),
        ]
    )
    with pytest.raises(RouteEncodingError):
        encoder.encode(plan, commitments=commitment)


def test_bounded_repair_has_hard_attempt_cap(scenario):
    evaluator = RouteEvaluator(scenario)
    repairer = BoundedRouteRepairer(
        scenario,
        evaluator,
        config=RepairConfig(max_attempts=1, max_moves=10),
    )
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence("V1", ["J1", "J2", "J3", "J4", "J5"]),
            VehicleRoute.from_sequence("V2", []),
        ]
    )
    result = repairer.repair(plan)
    assert result.attempts_used <= 1


def test_repair_finds_feasible_relocation_on_fixture(scenario):
    evaluator = RouteEvaluator(scenario)
    repairer = BoundedRouteRepairer(
        scenario,
        evaluator,
        config=RepairConfig(max_attempts=500, max_moves=20),
    )
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence("V1", ["J1", "J2", "J3", "J4", "J5"]),
            VehicleRoute.from_sequence("V2", []),
        ]
    )
    result = repairer.repair(plan)
    assert result.feasible
    assert result.complete
    assert set(result.repaired_route_plan.all_customer_ids()) == {"J1", "J2", "J3", "J4", "J5"}
    assert len(result.repaired_route_plan.duplicate_customer_ids()) == 0


def test_repair_does_not_move_locked_prefix(scenario):
    evaluator = RouteEvaluator(scenario)
    commitment = CommitmentSnapshot(
        (
            VehicleCommitment(
                vehicle_id="V1",
                current_node_id="N0",
                current_time_s=0.0,
                current_load_units=0.0,
                committed_customer_ids=("J1",),
            ),
        )
    )
    repairer = BoundedRouteRepairer(
        scenario,
        evaluator,
        config=RepairConfig(max_attempts=500, max_moves=20),
    )
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence("V1", ["J1", "J2", "J3", "J4", "J5"]),
            VehicleRoute.from_sequence("V2", []),
        ]
    )
    result = repairer.repair(plan, commitments=commitment)
    assert result.repaired_route_plan.route_for("V1").customer_ids[0] == "J1"
    assert all(
        action.customer_id != "J1"
        for action in result.actions
    )


def test_repaired_candidate_carries_canonical_keys(scenario):
    evaluator = RouteEvaluator(scenario)
    engine = Step7RouteEngine(
        scenario,
        evaluator,
        repair_config=RepairConfig(max_attempts=500, max_moves=20),
    )
    n = len(scenario.requests)
    candidate = engine.evaluate_keys(
        [0.0] * n + [0.1, 0.2, 0.3, 0.4, 0.5],
    )
    recoded = engine.encoder.decode(candidate.stored_keys)
    assert recoded.route_plan == candidate.repaired_plan


def test_encode_rejects_wrong_committed_prefix_order(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    commitment = CommitmentSnapshot(
        (
            VehicleCommitment(
                vehicle_id="V1",
                current_node_id="N0",
                current_time_s=0.0,
                current_load_units=0.0,
                committed_customer_ids=("J2", "J1"),
            ),
        )
    )
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence("V1", ["J1", "J2", "J3"]),
            VehicleRoute.from_sequence("V2", ["J4", "J5"]),
        ]
    )
    with pytest.raises(RouteEncodingError, match="committed customer prefix"):
        encoder.encode(plan, commitments=commitment)


def test_encode_decode_invariant_holds_under_commitments(scenario):
    encoder = ContinuousRouteEncoder(scenario)
    commitment = CommitmentSnapshot(
        (
            VehicleCommitment(
                vehicle_id="V1",
                current_node_id="N0",
                current_time_s=12.0,
                current_load_units=0.0,
                committed_customer_ids=("J2", "J1"),
            ),
        )
    )
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence("V1", ["J2", "J1", "J3"]),
            VehicleRoute.from_sequence("V2", ["J5", "J4"]),
        ]
    )
    encoded = encoder.encode(plan, commitments=commitment)
    assert encoder.decode(encoded.keys, commitments=commitment).route_plan == plan


def test_tie_tolerance_is_deterministic_and_customer_id_breaks_ties(scenario):
    encoder = ContinuousRouteEncoder(
        scenario,
        config=RouteEncodingConfig(tie_tolerance=0.1),
    )
    n = len(scenario.requests)
    # J1 and J2 fall in the same quantised priority bucket; customer ID is
    # therefore the deterministic tie breaker.
    priorities = [0.21, 0.24, 0.51, 0.71, 0.91]
    decoded_a = encoder.decode(_keys(encoder, [0.0] * n, priorities))
    decoded_b = encoder.decode(_keys(encoder, [0.0] * n, priorities))
    assert decoded_a.route_plan == decoded_b.route_plan
    assert decoded_a.route_plan.route_for("V1").customer_ids[:2] == ("J1", "J2")


def test_engine_canonical_keys_reproduce_repaired_plan_under_commitment(scenario):
    evaluator = RouteEvaluator(scenario)
    commitment = CommitmentSnapshot(
        (
            VehicleCommitment(
                vehicle_id="V1",
                current_node_id="N0",
                current_time_s=0.0,
                current_load_units=0.0,
                committed_customer_ids=("J1",),
            ),
        )
    )
    engine = Step7RouteEngine(
        scenario,
        evaluator,
        repair_config=RepairConfig(max_attempts=500, max_moves=20),
    )
    n = len(scenario.requests)
    candidate = engine.evaluate_keys(
        [1.0] * n + [0.9, 0.8, 0.7, 0.6, 0.5],
        commitments=commitment,
    )
    recoded = engine.encoder.decode(
        candidate.stored_keys,
        commitments=commitment,
    )
    assert recoded.route_plan == candidate.repaired_plan
    assert candidate.repaired_plan.route_for("V1").customer_ids[0] == "J1"


def test_repair_preserves_customer_cardinality_and_locked_prefix(scenario):
    evaluator = RouteEvaluator(scenario)
    commitment = CommitmentSnapshot(
        (
            VehicleCommitment(
                vehicle_id="V1",
                current_node_id="N0",
                current_time_s=0.0,
                current_load_units=0.0,
                committed_customer_ids=("J2", "J1"),
            ),
        )
    )
    repairer = BoundedRouteRepairer(
        scenario,
        evaluator,
        config=RepairConfig(max_attempts=500, max_moves=20),
    )
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence("V1", ["J2", "J1", "J3", "J4", "J5"]),
            VehicleRoute.from_sequence("V2", []),
        ]
    )
    result = repairer.repair(plan, commitments=commitment)
    repaired = result.repaired_route_plan
    assert repaired.route_for("V1").customer_ids[:2] == ("J2", "J1")
    assert tuple(sorted(repaired.all_customer_ids(), key=str)) == tuple(
        sorted((r.request_id for r in scenario.requests), key=str)
    )
