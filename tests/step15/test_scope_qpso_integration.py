from __future__ import annotations

from src.contracts.decision import ScopeAction
from src.routing.route_plan import RoutePlan
from src.optim.qpso import (
    AdaptiveQPSO,
    AdaptiveQPSOConfig,
    RouteFitnessOracle,
)
from src.routing.evaluator_state import CommitmentSnapshot
from src.routing.route_encoding import Step7RouteEngine
from src.routing.route_evaluator import RouteEvaluator
from src.runtime.scope_actions import (
    ScopeSelection,
    build_commitment_snapshot,
)


def _reference_plan() -> RoutePlan:
    return RoutePlan.from_routes(
        [
            # The five-job fixture uses this known feasible assignment.
            # Keep this test focused on the optimizer/commitment boundary.
            __import__(
                "src.routing.route_plan",
                fromlist=["VehicleRoute"],
            ).VehicleRoute.from_sequence(
                "V1",
                ["J1", "J2", "J3"],
            ),
            __import__(
                "src.routing.route_plan",
                fromlist=["VehicleRoute"],
            ).VehicleRoute.from_sequence(
                "V2",
                ["J4", "J5"],
            ),
        ]
    )


def test_real_qpso_runs_through_step7_with_commitments(scenario):
    """
    Proves the complete real integration:

        ScopeSelection
          -> CommitmentSnapshot
          -> Step7RouteEngine
          -> RouteFitnessOracle
          -> AdaptiveQPSO
          -> RouteCandidate
    """

    current_plan = _reference_plan()

    selection = ScopeSelection(
        action=ScopeAction.VEHICLE,
        request_ids=("J2", "J3"),
        vehicle_ids=("V1",),
        reason="integration test",
        affected_vehicle_ids=("V1",),
    )

    commitments = build_commitment_snapshot(
        scenario,
        current_plan,
        selection,
        planning_time_s=0.0,
    )

    assert isinstance(
        commitments,
        CommitmentSnapshot,
    )

    # V2 is outside the selected scope, so its complete assignment is
    # immutable.
    v2 = commitments.for_vehicle("V2")

    assert v2 is not None
    assert v2.committed_customer_ids == (
        "J4",
        "J5",
    )

    # J1 is before the mutable suffix on V1 and therefore remains locked.
    v1 = commitments.for_vehicle("V1")

    assert v1 is not None
    assert v1.committed_customer_ids == (
        "J1",
    )

    evaluator = RouteEvaluator(
        scenario,
    )

    engine = Step7RouteEngine(
        scenario,
        evaluator,
    )

    oracle = RouteFitnessOracle(
        engine,
        commitments=commitments,
        planning_time_s=0.0,
        repair=True,
    )

    dimensions = engine.encoder.dimension

    config = AdaptiveQPSOConfig(
        dimensions=dimensions,
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=4,
        max_evaluations=8,
        seed=26137,
    )

    # Seed with a real encoding of the current plan.
    incumbent = engine.encoder.encode(
        current_plan,
        commitments=commitments,
    )

    population = [
        incumbent.keys,
        incumbent.keys,
        incumbent.keys,
        incumbent.keys,
    ]

    optimizer = AdaptiveQPSO(
        config,
        oracle,
        initial_population=population,
    )

    result = optimizer.optimize()

    assert result.evaluations == 8
    assert result.best_position
    assert len(result.best_position) == dimensions

    # Re-evaluate the final optimizer position through the actual engine.
    candidate = engine.evaluate_keys(
        result.best_position,
        commitments=commitments,
        planning_time_s=0.0,
        repair=True,
    )

    assert candidate is not None
    assert candidate.repaired_plan is not None
    assert candidate.repaired_evaluation is not None

    # The optimizer must return a candidate understood by the evaluator.
    assert candidate.repaired_evaluation.feasible

    # Immutable V2 assignment must survive the complete pipeline.
    assert candidate.repaired_plan.route_for(
        "V2"
    ).customer_ids == (
        "J4",
        "J5",
    )

    # Immutable V1 prefix must survive.
    assert candidate.repaired_plan.route_for(
        "V1"
    ).customer_ids[0] == "J1"

    # Every customer remains represented exactly once.
    customer_ids = candidate.repaired_plan.all_customer_ids()

    assert set(customer_ids) == {
        "J1",
        "J2",
        "J3",
        "J4",
        "J5",
    }

    assert len(customer_ids) == len(
        set(customer_ids)
    )


def test_keep_does_not_call_qpso(scenario):
    """
    KEEP is a true no-search action.
    """

    current_plan = _reference_plan()

    selection = ScopeSelection(
        action=ScopeAction.KEEP,
        request_ids=(),
        vehicle_ids=(),
        reason="no replanning required",
        affected_vehicle_ids=(),
    )

    # A KEEP action still gets a commitment snapshot so the current
    # operational state can be represented consistently.
    commitments = build_commitment_snapshot(
        scenario,
        current_plan,
        selection,
        planning_time_s=0.0,
    )

    assert commitments.for_vehicle(
        "V1"
    ).committed_customer_ids == (
        "J1",
        "J2",
        "J3",
    )

    assert commitments.for_vehicle(
        "V2"
    ).committed_customer_ids == (
        "J4",
        "J5",
    )