from __future__ import annotations

import pytest

from src.contracts.decision import ScopeAction
from src.routing.route_plan import RoutePlan, VehicleRoute
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


def make_current_plan() -> RoutePlan:
    return RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(
                "V1",
                ["J1", "J2", "J3"],
            ),
            VehicleRoute.from_sequence(
                "V2",
                ["J4", "J5"],
            ),
        ]
    )


def run_real_qpso(
    scenario,
    current_plan,
    selection,
):
    commitments = build_commitment_snapshot(
        scenario,
        current_plan,
        selection,
        planning_time_s=0.0,
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

    incumbent = engine.encoder.encode(
        current_plan,
        commitments=commitments,
    )

    config = AdaptiveQPSOConfig(
        dimensions=engine.encoder.dimension,
        lower_bound=0.0,
        upper_bound=1.0,
        population_size=4,
        max_evaluations=8,
        seed=26137,
    )

    population = [
        list(incumbent.keys)
        for _ in range(4)
    ]

    optimizer = AdaptiveQPSO(
        config,
        oracle,
        initial_population=population,
    )

    result = optimizer.optimize()

    assert result.evaluations == 8

    candidate = engine.evaluate_keys(
        result.best_position,
        commitments=commitments,
        planning_time_s=0.0,
        repair=True,
    )

    assert candidate.repaired_plan is not None
    assert candidate.repaired_evaluation is not None
    assert candidate.repaired_evaluation.feasible

    return commitments, candidate.repaired_plan


@pytest.mark.parametrize(
    "action",
    [
        ScopeAction.LOCAL,
        ScopeAction.VEHICLE,
        ScopeAction.REGIONAL,
        ScopeAction.GLOBAL,
    ],
)
def test_each_replanning_action_reaches_real_qpso(
    scenario,
    action,
):
    """
    Every actual replanning action must reach the real Step-7/QPSO
    implementation and return a feasible route.
    """

    current_plan = make_current_plan()

    if action == ScopeAction.LOCAL:
        selection = ScopeSelection(
            action=action,
            request_ids=("J2", "J3"),
            vehicle_ids=("V1",),
            reason="local integration",
            affected_vehicle_ids=("V1",),
        )

    elif action == ScopeAction.VEHICLE:
        selection = ScopeSelection(
            action=action,
            request_ids=("J2", "J3"),
            vehicle_ids=("V1",),
            reason="vehicle integration",
            affected_vehicle_ids=("V1",),
        )

    elif action == ScopeAction.REGIONAL:
        selection = ScopeSelection(
            action=action,
            request_ids=("J2", "J3", "J4"),
            vehicle_ids=("V1", "V2"),
            reason="regional integration",
            affected_vehicle_ids=("V1",),
        )

    else:
        selection = ScopeSelection(
            action=action,
            request_ids=(
                "J1",
                "J2",
                "J3",
                "J4",
                "J5",
            ),
            vehicle_ids=("V1", "V2"),
            reason="global integration",
            affected_vehicle_ids=("V1", "V2"),
        )

    commitments, replanned = run_real_qpso(
        scenario,
        current_plan,
        selection,
    )

    assert isinstance(
        commitments,
        CommitmentSnapshot,
    )

    # Every request must still occur exactly once.
    customer_ids = replanned.all_customer_ids()

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


def test_keep_is_a_true_no_replan_action(
    scenario,
):
    current_plan = make_current_plan()

    selection = ScopeSelection(
        action=ScopeAction.KEEP,
        request_ids=(),
        vehicle_ids=(),
        reason="keep current plan",
        affected_vehicle_ids=(),
    )

    commitments = build_commitment_snapshot(
        scenario,
        current_plan,
        selection,
        planning_time_s=0.0,
    )

    # KEEP must lock the complete current assignment.
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


def test_local_never_exceeds_ten_jobs(
    scenario,
):
    jobs = []

    for index in range(25):
        from src.runtime.scope_actions import JobImpact

        jobs.append(
            JobImpact(
                request_id=f"J{index}",
                vehicle_id="V1",
                affected=True,
                deadline_slack_s=float(
                    index + 1
                ),
                route_overlap_fraction=1.0,
                congestion_exposure=1.0,
            )
        )

    from src.runtime.scope_actions import ScopeActionSelector

    selector = ScopeActionSelector()

    result = selector.select(
        ScopeAction.LOCAL,
        jobs=jobs,
        affected_vehicle_ids=("V1",),
    )

    assert len(result.request_ids) <= 10