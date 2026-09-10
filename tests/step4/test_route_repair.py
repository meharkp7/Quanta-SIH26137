"""Tests for deterministic route-plan repair.

These tests verify that the repair layer:
- removes unknown customers;
- removes duplicate assignments;
- repairs capacity violations when possible;
- preserves valid assignments;
- produces deterministic results;
- reports unresolved violations instead of hiding them;
- works independently of any optimizer.
"""

from __future__ import annotations

from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_plan import RoutePlan, VehicleRoute
from src.routing.route_repair import (
    RouteRepairConfig,
    RouteRepairer,
)


def make_plan(*routes: tuple[str, tuple[str, ...]]) -> RoutePlan:
    """Build a deterministic RoutePlan from compact test data."""
    return RoutePlan.from_routes(
        VehicleRoute.from_sequence(
            vehicle_id=vehicle_id,
            customer_ids=customer_ids,
        )
        for vehicle_id, customer_ids in routes
    )


def test_repair_removes_unknown_customer(scenario) -> None:
    """Unknown request IDs must not survive repair."""
    evaluator = RouteEvaluator(scenario)

    original = make_plan(
        ("V1", ("J1", "UNKNOWN")),
        ("V2", ("J4", "J5")),
    )

    repairer = RouteRepairer(
        scenario,
        evaluator,
    )

    result = repairer.repair(original)

    assert result.changed is True

    assert "UNKNOWN" not in result.repaired_route_plan.all_customer_ids()

    assert result.complete is False or result.feasible is False

    assert any(
        action.action == "remove_unknown_customer"
        and action.customer_id == "UNKNOWN"
        for action in result.actions
    )


def test_repair_removes_duplicate_customer_assignment(scenario) -> None:
    """A customer assigned more than once must be reduced to one assignment."""
    evaluator = RouteEvaluator(scenario)

    original = make_plan(
        ("V1", ("J1", "J2", "J3")),
        ("V2", ("J3", "J4", "J5")),
    )

    assert original.is_customer_unique() is False

    repairer = RouteRepairer(
        scenario,
        evaluator,
    )

    result = repairer.repair(original)

    repaired_ids = result.repaired_route_plan.all_customer_ids()

    assert repaired_ids.count("J3") == 1
    assert result.repaired_route_plan.is_customer_unique() is True

    assert any(
        action.action == "remove_duplicate_customer"
        and action.customer_id == "J3"
        for action in result.actions
    )


def test_repair_can_move_customer_between_vehicles_for_capacity(
    scenario,
) -> None:
    """Capacity repair should relocate a customer when another vehicle can take it."""
    evaluator = RouteEvaluator(scenario)

    # V1 capacity is 7. J1+J2+J3+J4 = 4+4+3+2 = 13.
    # V2 has enough remaining capacity for J4.
    original = make_plan(
        ("V1", ("J1", "J2", "J3", "J4")),
        ("V2", ("J5",)),
    )

    original_evaluation = evaluator.evaluate(original)

    assert original_evaluation.capacity_feasible is False

    repairer = RouteRepairer(
        scenario,
        evaluator,
        config=RouteRepairConfig(
            allow_cross_vehicle_relocation=True,
        ),
    )

    result = repairer.repair(original)

    assert result.changed is True

    repaired_evaluation = result.repaired_evaluation

    assert repaired_evaluation.capacity_feasible is True

    # Every customer remains assigned exactly once.
    assert result.repaired_route_plan.is_customer_unique() is True

    assert set(
        result.repaired_route_plan.all_customer_ids()
    ) == {
        "J1",
        "J2",
        "J3",
        "J4",
        "J5",
    }

    assert any(
        action.action in {
            "relocate_customer",
            "move_customer",
        }
        and action.customer_id == "J4"
        for action in result.actions
    )


def test_repair_preserves_already_feasible_plan(scenario) -> None:
    """A feasible reference plan should not be modified."""
    evaluator = RouteEvaluator(scenario)

    original = make_plan(
        ("V1", ("J1", "J2", "J3")),
        ("V2", ("J4", "J5")),
    )

    original_evaluation = evaluator.evaluate(original)

    assert original_evaluation.feasible is True

    repairer = RouteRepairer(
        scenario,
        evaluator,
    )

    result = repairer.repair(original)

    assert result.repaired_route_plan == original
    assert result.changed is False
    assert result.feasible is True
    assert result.complete is True
    assert result.actions == ()

    assert (
        result.repaired_evaluation.objective_value
        == original_evaluation.objective_value
    )


def test_repair_is_deterministic(scenario) -> None:
    """Running repair twice on the same input must produce the same result."""
    evaluator = RouteEvaluator(scenario)

    original = make_plan(
        ("V1", ("J1", "J2", "J3", "J4")),
        ("V2", ("J5",)),
    )

    repairer = RouteRepairer(
        scenario,
        evaluator,
    )

    first = repairer.repair(original)
    second = repairer.repair(original)

    assert first.repaired_route_plan == second.repaired_route_plan
    assert first.repaired_evaluation == second.repaired_evaluation
    assert first.actions == second.actions
    assert first.unresolved_customer_ids == second.unresolved_customer_ids
    assert first.unresolved_reasons == second.unresolved_reasons


def test_repair_does_not_mutate_original_route_plan(scenario) -> None:
    """RoutePlan is immutable and must remain unchanged after repair."""
    evaluator = RouteEvaluator(scenario)

    original = make_plan(
        ("V1", ("J1", "J2", "J3", "J4")),
        ("V2", ("J5",)),
    )

    original_routes = original.vehicle_routes
    original_customer_ids = original.all_customer_ids()

    repairer = RouteRepairer(
        scenario,
        evaluator,
    )

    result = repairer.repair(original)

    assert original.vehicle_routes == original_routes
    assert original.all_customer_ids() == original_customer_ids

    assert result.repaired_route_plan is not original


def test_repair_reports_unresolved_customer_when_capacity_cannot_be_fixed(
    scenario,
) -> None:
    """Repair must report infeasibility if no legal relocation exists."""
    evaluator = RouteEvaluator(scenario)

    # Disable cross-vehicle relocation. The overloaded route therefore
    # cannot be repaired by moving customers elsewhere.
    original = make_plan(
        ("V1", ("J1", "J2", "J3", "J4", "J5")),
        ("V2", ()),
    )

    repairer = RouteRepairer(
        scenario,
        evaluator,
        config=RouteRepairConfig(
            allow_cross_vehicle_relocation=False,
            repair_connectivity=False,
            repair_time_windows=False,
        ),
    )

    result = repairer.repair(original)

    assert result.feasible is False
    assert result.complete is True

    assert (
        result.repaired_evaluation.capacity_feasible
        is False
    )

    assert result.unresolved_customer_ids


def test_repair_can_be_configured_to_only_clean_assignments(
    scenario,
) -> None:
    """Repair options can be disabled independently."""
    evaluator = RouteEvaluator(scenario)

    original = make_plan(
        ("V1", ("J1", "J1", "UNKNOWN")),
        ("V2", ("J4", "J5")),
    )

    repairer = RouteRepairer(
        scenario,
        evaluator,
        config=RouteRepairConfig(
            remove_unknown_customers=True,
            remove_duplicates=True,
            repair_capacity=False,
            repair_connectivity=False,
            repair_time_windows=False,
        ),
    )

    result = repairer.repair(original)

    repaired_ids = result.repaired_route_plan.all_customer_ids()

    assert "UNKNOWN" not in repaired_ids
    assert repaired_ids.count("J1") == 1

    assert any(
        action.action == "remove_unknown_customer"
        for action in result.actions
    )

    assert any(
        action.action == "remove_duplicate_customer"
        for action in result.actions
    )


def test_repair_result_keeps_original_and_repaired_evaluations(
    scenario,
) -> None:
    """Repair output must expose both sides of the transformation."""
    evaluator = RouteEvaluator(scenario)

    original = make_plan(
        ("V1", ("J1", "J2", "J3", "J4")),
        ("V2", ("J5",)),
    )

    repairer = RouteRepairer(
        scenario,
        evaluator,
    )

    result = repairer.repair(original)

    assert result.original_route_plan == original

    assert (
        result.original_evaluation
        == evaluator.evaluate(original)
    )

    assert (
        result.repaired_evaluation
        == evaluator.evaluate(
            result.repaired_route_plan
        )
    )

    assert result.passes_used >= 1


def test_repair_empty_plan_is_safe(scenario) -> None:
    """An empty logical plan should not crash the repair layer."""
    evaluator = RouteEvaluator(scenario)

    original = make_plan(
        ("V1", ()),
        ("V2", ()),
    )

    repairer = RouteRepairer(
        scenario,
        evaluator,
    )

    result = repairer.repair(original)

    assert result.repaired_route_plan == original
    assert result.changed is False

    assert result.repaired_route_plan.vehicle_count == 2
    assert result.repaired_route_plan.total_stop_count == 0

    assert result.complete is False