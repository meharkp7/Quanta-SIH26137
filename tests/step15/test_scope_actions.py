from src.contracts.decision import ScopeAction
from src.runtime.scope_actions import (
    JobImpact,
    ScopeActionConfig,
    ScopeActionSelector,
)

from src.routing.route_plan import (
    RoutePlan,
    VehicleRoute,
)
from src.runtime.scope_actions import (
    build_scope_commitment_plan, build_commitment_snapshot, ScopeSelection
)
def make_jobs():
    return [
        JobImpact(
            request_id="r1",
            vehicle_id="v1",
            zone_id="z1",
            affected=True,
            deadline_slack_s=30,
            route_overlap_fraction=0.9,
            congestion_exposure=0.8,
        ),
        JobImpact(
            request_id="r2",
            vehicle_id="v1",
            zone_id="z1",
            affected=False,
            deadline_slack_s=300,
            route_overlap_fraction=0.2,
            congestion_exposure=0.2,
        ),
        JobImpact(
            request_id="r3",
            vehicle_id="v2",
            zone_id="z2",
            affected=True,
            deadline_slack_s=60,
            route_overlap_fraction=0.7,
            congestion_exposure=0.6,
        ),
        JobImpact(
            request_id="r4",
            vehicle_id="v2",
            zone_id="z2",
            affected=False,
            deadline_slack_s=500,
            route_overlap_fraction=0.1,
            congestion_exposure=0.1,
        ),
        JobImpact(
            request_id="r5",
            vehicle_id="v3",
            zone_id="z3",
            affected=False,
            deadline_slack_s=600,
            route_overlap_fraction=0.0,
            congestion_exposure=0.0,
        ),
    ]


def test_keep_preserves_current_plan_when_nothing_is_affected():
    selector = ScopeActionSelector()

    jobs = [
        JobImpact(
            request_id="r1",
            vehicle_id="v1",
            affected=False,
        )
    ]

    result = selector.select(
        ScopeAction.KEEP,
        jobs=jobs,
        affected_vehicle_ids=(),
    )

    assert result.action == ScopeAction.KEEP
    assert result.request_ids == ()
    assert result.vehicle_ids == ()
    assert result.overridden is False


def test_keep_is_overridden_when_affected_mutable_work_exists():
    selector = ScopeActionSelector()

    result = selector.select(
        ScopeAction.KEEP,
        jobs=make_jobs(),
        affected_vehicle_ids=("v1",),
    )

    assert result.overridden is True
    assert result.override_reason
    assert result.action in {
        ScopeAction.LOCAL,
        ScopeAction.GLOBAL,
    }

    assert "r1" in result.request_ids
    assert "v1" in result.vehicle_ids


def test_local_selects_at_most_ten_highest_impact_jobs():
    selector = ScopeActionSelector(
        ScopeActionConfig(local_max_jobs=2)
    )

    result = selector.select(
        ScopeAction.LOCAL,
        jobs=make_jobs(),
        affected_vehicle_ids=("v1", "v2"),
    )

    assert result.action == ScopeAction.LOCAL
    assert len(result.request_ids) <= 2
    assert len(result.request_ids) == len(set(result.request_ids))

    # r1 has the highest impact.
    assert result.request_ids[0] == "r1"


def test_vehicle_selects_only_one_vehicle():
    selector = ScopeActionSelector()

    result = selector.select(
        ScopeAction.VEHICLE,
        jobs=make_jobs(),
        affected_vehicle_ids=("v1", "v2"),
    )

    assert result.action == ScopeAction.VEHICLE
    assert len(result.vehicle_ids) == 1

    selected_vehicle = result.vehicle_ids[0]

    assert all(
        vehicle_id == selected_vehicle
        for vehicle_id in result.vehicle_ids
    )

    assert all(
        request_id in {"r1", "r2"}
        for request_id in result.request_ids
    )


def test_regional_selects_affected_and_zone_jobs():
    selector = ScopeActionSelector()

    result = selector.select(
        ScopeAction.REGIONAL,
        jobs=make_jobs(),
        affected_vehicle_ids=("v1",),
        affected_zone_ids=("z2",),
    )

    assert result.action == ScopeAction.REGIONAL

    assert set(result.request_ids) == {
        "r1",
        "r2",
        "r3",
        "r4",
    }

    assert "r5" not in result.request_ids


def test_global_selects_all_legally_mutable_jobs():
    selector = ScopeActionSelector()

    result = selector.select(
        ScopeAction.GLOBAL,
        jobs=make_jobs(),
    )

    assert result.action == ScopeAction.GLOBAL

    assert set(result.request_ids) == {
        "r1",
        "r2",
        "r3",
        "r4",
        "r5",
    }

    assert set(result.vehicle_ids) == {
        "v1",
        "v2",
        "v3",
    }


def test_illegal_requests_are_excluded_from_scope():
    selector = ScopeActionSelector()

    result = selector.select(
        ScopeAction.GLOBAL,
        jobs=make_jobs(),
        legally_mutable_request_ids={
            "r1",
            "r3",
        },
    )

    assert set(result.request_ids) == {
        "r1",
        "r3",
    }

    assert set(result.vehicle_ids) == {
        "v1",
        "v2",
    }


def test_illegal_vehicles_are_excluded_from_scope():
    selector = ScopeActionSelector()

    result = selector.select(
        ScopeAction.GLOBAL,
        jobs=make_jobs(),
        legally_mutable_vehicle_ids={
            "v1",
        },
    )

    assert set(result.request_ids) == {
        "r1",
        "r2",
    }

    assert result.vehicle_ids == ("v1",)


def test_duplicate_inputs_do_not_create_duplicate_outputs():
    selector = ScopeActionSelector()

    result = selector.select(
        ScopeAction.GLOBAL,
        jobs=make_jobs(),
        affected_vehicle_ids=("v1", "v1", "v2", "v2"),
    )

    assert len(result.request_ids) == len(set(result.request_ids))
    assert len(result.vehicle_ids) == len(set(result.vehicle_ids))


def test_empty_vehicle_scope_is_handled_consistently():
    selector = ScopeActionSelector()

    result = selector.select(
        ScopeAction.VEHICLE,
        jobs=[],
        affected_vehicle_ids=(),
    )

    assert result.action == ScopeAction.VEHICLE
    assert result.request_ids == ()
    assert result.vehicle_ids == ()


def test_local_scope_is_deterministic():
    selector = ScopeActionSelector(
        ScopeActionConfig(local_max_jobs=3)
    )

    jobs = make_jobs()

    first = selector.select(
        ScopeAction.LOCAL,
        jobs=jobs,
        affected_vehicle_ids=("v1", "v2"),
    )

    second = selector.select(
        ScopeAction.LOCAL,
        jobs=jobs,
        affected_vehicle_ids=("v1", "v2"),
    )

    assert first.request_ids == second.request_ids
    assert first.vehicle_ids == second.vehicle_ids


def test_scope_commitments_preserve_prefixes():
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(
                "v1",
                ["r1", "r2", "r3"],
            ),
            VehicleRoute.from_sequence(
                "v2",
                ["r4", "r5"],
            ),
        ]
    )

    selection = selector = ScopeActionSelector().select(
        ScopeAction.LOCAL,
        jobs=make_jobs(),
        affected_vehicle_ids=("v1",),
    )

    commitments = build_scope_commitment_plan(
        plan,
        selection,
    )

    # r1 is the first mutable request on v1,
    # therefore nothing after it is treated as committed prefix.
    assert commitments.committed_by_vehicle["v1"] == ()

    # v2 was not selected by LOCAL, so its complete current route stays fixed.
    assert commitments.committed_by_vehicle["v2"] == (
        "r4",
        "r5",
    )


def test_unselected_vehicle_is_completely_locked():
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(
                "v1",
                ["r1", "r2"],
            ),
            VehicleRoute.from_sequence(
                "v2",
                ["r3", "r4"],
            ),
        ]
    )

    selection = ScopeActionSelector().select(
        ScopeAction.VEHICLE,
        jobs=make_jobs(),
        affected_vehicle_ids=("v1",),
    )

    commitments = build_scope_commitment_plan(
        plan,
        selection,
    )

    assert commitments.committed_by_vehicle["v2"] == (
        "r3",
        "r4",
    )


def test_keep_commits_entire_current_plan():
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(
                "v1",
                ["r1", "r2"],
            ),
            VehicleRoute.from_sequence(
                "v2",
                ["r3"],
            ),
        ]
    )

    selection = ScopeActionSelector().select(
        ScopeAction.KEEP,
        jobs=[],
    )

    commitments = build_scope_commitment_plan(
        plan,
        selection,
    )

    assert commitments.mutable_request_ids == ()
    assert commitments.mutable_vehicle_ids == ()

    assert commitments.committed_by_vehicle == {
        "v1": ("r1", "r2"),
        "v2": ("r3",),
    }

def test_build_commitment_snapshot_locks_unselected_vehicle(
    scenario,
):
    from src.routing.route_plan import (
        RoutePlan,
        VehicleRoute,
    )

    plan = RoutePlan.from_routes(
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

    selector = ScopeActionSelector()

    selection = selector.select(
        ScopeAction.VEHICLE,
        jobs=make_jobs(),
        affected_vehicle_ids=("V1",),
    )

    snapshot = build_commitment_snapshot(
        scenario,
        plan,
        selection,
    )

    v2 = snapshot.for_vehicle("V2")

    assert v2 is not None
    assert v2.committed_customer_ids == (
        "J4",
        "J5",
    )


def test_build_commitment_snapshot_preserves_mutable_vehicle_prefix(
    scenario,
):
    from src.routing.route_plan import (
        RoutePlan,
        VehicleRoute,
    )

    plan = RoutePlan.from_routes(
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

    selector = ScopeActionSelector()

    selection = ScopeSelection(
        action=ScopeAction.VEHICLE,
        request_ids=("J2",),
        vehicle_ids=("V1",),
        reason="test",
        affected_vehicle_ids=("V1",),
    )

    snapshot = build_commitment_snapshot(
        scenario,
        plan,
        selection,
    )

    v1 = snapshot.for_vehicle("V1")

    assert v1 is not None

    # J1 is before the first mutable request J2.
    assert v1.committed_customer_ids == (
        "J1",
    )


def test_keep_locks_entire_current_plan(
    scenario,
):
    from src.routing.route_plan import (
        RoutePlan,
        VehicleRoute,
    )

    plan = RoutePlan.from_routes(
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

    selection = ScopeSelection(
        action=ScopeAction.KEEP,
        request_ids=(),
        vehicle_ids=(),
        reason="nothing to replan",
        affected_vehicle_ids=(),
    )

    snapshot = build_commitment_snapshot(
        scenario,
        plan,
        selection,
    )

    assert snapshot.for_vehicle(
        "V1"
    ).committed_customer_ids == (
        "J1",
        "J2",
        "J3",
    )

    assert snapshot.for_vehicle(
        "V2"
    ).committed_customer_ids == (
        "J4",
        "J5",
    )