from src.routing.route_plan import (
    RoutePlan,
    StopLocation,
    VehicleRoute,
)


def test_vehicle_route_from_sequence():
    route = VehicleRoute.from_sequence(
        "V1",
        ["J1", "J2", "J3"],
    )

    assert route.vehicle_id == "V1"
    assert route.customer_ids == ("J1", "J2", "J3")
    assert route.stop_count == 3
    assert not route.is_empty


def test_empty_vehicle_route():
    route = VehicleRoute.from_sequence("V1", [])

    assert route.is_empty
    assert route.stop_count == 0


def test_route_plan_rejects_duplicate_vehicle_ids():
    first = VehicleRoute.from_sequence("V1", ["J1"])
    second = VehicleRoute.from_sequence("V1", ["J2"])

    try:
        RoutePlan.from_routes([first, second])
    except ValueError as exc:
        assert "duplicate vehicle" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected duplicate vehicle IDs to be rejected"
        )


def test_route_plan_statistics():
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(
                "V1",
                ["J1", "J2", "J3"],
            ),
            VehicleRoute.from_sequence(
                "V2",
                ["J4"],
            ),
            VehicleRoute.from_sequence(
                "V3",
                [],
            ),
        ]
    )

    assert plan.vehicle_count == 3
    assert plan.non_empty_vehicle_count == 2
    assert plan.total_stop_count == 4


def test_route_plan_customer_operations():
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(
                "V1",
                ["J1", "J2"],
            ),
            VehicleRoute.from_sequence(
                "V2",
                ["J3", "J2"],
            ),
        ]
    )

    assert plan.all_customer_ids() == (
        "J1",
        "J2",
        "J3",
        "J2",
    )

    assert plan.unique_customer_ids() == (
        "J1",
        "J2",
        "J3",
    )

    assert plan.duplicate_customer_ids() == (
        "J2",
        "J2",
    )

    assert not plan.is_customer_unique()


def test_route_lookup():
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(
                "V1",
                ["J1"],
            ),
            VehicleRoute.from_sequence(
                "V2",
                ["J2"],
            ),
        ]
    )

    assert plan.route_for("V1").customer_ids == ("J1",)
    assert plan.route_for("V2").customer_ids == ("J2",)


def test_missing_route_raises_key_error():
    plan = RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(
                "V1",
                ["J1"],
            )
        ]
    )

    try:
        plan.route_for("V99")
    except KeyError:
        pass
    else:
        raise AssertionError(
            "Expected missing vehicle lookup to raise KeyError"
        )


def test_stop_location():
    stop = StopLocation(
        customer_id="J1",
        node_id="N1",
    )

    assert stop.customer_id == "J1"
    assert stop.node_id == "N1"