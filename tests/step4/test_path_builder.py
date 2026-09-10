import pytest

from src.routing.path_builder import (
    DirectedPathBuilder,
    InvalidTravelTimeError,
    PathNotFoundError,
)


def test_shortest_path_uses_directed_edges(scenario):
    builder = DirectedPathBuilder(
        scenario.edges
    )

    result = builder.shortest_path(
        "N1",
        "N2",
    )

    assert result.edge_ids == ("E12",)
    assert result.node_ids == ("N1", "N2")
    assert result.distance_m == pytest.approx(100.0)
    assert result.travel_time_s == pytest.approx(10.0)


def test_reverse_direction_is_not_inferred(scenario):
    builder = DirectedPathBuilder(
        scenario.edges
    )

    with pytest.raises(PathNotFoundError):
        builder.shortest_path(
            "N7",
            "N2",
        )


def test_closed_edge_is_excluded(scenario):
    builder = DirectedPathBuilder(
        scenario.edges,
        closed_edge_ids={"E12"},
    )

    result = builder.shortest_path(
        "N1",
        "N2",
    )

    assert result.edge_ids == (
        "E16",
        "E62",
    )

    assert result.travel_time_s == pytest.approx(
        30.0
    )


def test_static_free_flow_is_default(scenario):
    builder = DirectedPathBuilder(
        scenario.edges
    )

    result = builder.shortest_path(
        "N1",
        "N6",
    )

    assert result.edge_ids == ("E16",)
    assert result.travel_time_s == pytest.approx(
        15.0
    )


def test_same_node_path_is_empty(scenario):
    builder = DirectedPathBuilder(
        scenario.edges
    )

    result = builder.shortest_path(
        "N1",
        "N1",
    )

    assert result.node_ids == ("N1",)
    assert result.edge_ids == ()
    assert result.distance_m == pytest.approx(0.0)
    assert result.travel_time_s == pytest.approx(0.0)


def test_dynamic_provider_changes_path_choice(scenario):
    def dynamic_time(edge, departure_time):
        del departure_time

        if edge.edge_id == "E12":
            return 100.0

        return edge.length_m / edge.speed_limit_mps

    builder = DirectedPathBuilder(
        scenario.edges,
        travel_time_provider=dynamic_time,
    )

    result = builder.shortest_path(
        "N1",
        "N2",
    )

    assert result.edge_ids == (
        "E16",
        "E62",
    )

    assert result.travel_time_s == pytest.approx(
        30.0
    )


def test_dynamic_provider_is_time_dependent(scenario):
    observed_departures = []

    def dynamic_time(edge, departure_time):
        observed_departures.append(
            (edge.edge_id, departure_time)
        )

        if edge.edge_id == "E12":
            return (
                10.0
                if departure_time < 20.0
                else 40.0
            )

        return edge.length_m / edge.speed_limit_mps

    builder = DirectedPathBuilder(
        scenario.edges,
        travel_time_provider=dynamic_time,
    )

    result = builder.shortest_path(
        "N1",
        "N2",
        departure_time_s=20.0,
    )

    assert result.edge_ids == (
        "E16",
        "E62",
    )

    assert any(
        edge_id == "E12"
        and departure_time >= 20.0
        for edge_id, departure_time
        in observed_departures
    )


def test_dynamic_provider_is_used_for_each_leg(scenario):
    calls = []

    def dynamic_time(edge, departure_time):
        calls.append(
            (edge.edge_id, departure_time)
        )
        return edge.length_m / edge.speed_limit_mps

    builder = DirectedPathBuilder(
        scenario.edges,
        travel_time_provider=dynamic_time,
    )

    legs = builder.build_node_sequence(
        ["N1", "N2", "N3"],
        departure_time_s=0.0,
    )

    assert len(legs) == 2
    assert legs[0].travel_time_s == pytest.approx(10.0)
    assert legs[1].travel_time_s == pytest.approx(10.0)

    assert any(
        edge_id == "E23"
        and departure_time == pytest.approx(10.0)
        for edge_id, departure_time in calls
    )


def test_negative_dynamic_travel_time_is_rejected(scenario):
    def invalid_time(edge, departure_time):
        del edge, departure_time
        return -1.0

    builder = DirectedPathBuilder(
        scenario.edges,
        travel_time_provider=invalid_time,
    )

    with pytest.raises(InvalidTravelTimeError):
        builder.shortest_path(
            "N1",
            "N2",
        )


def test_nonfinite_dynamic_travel_time_is_rejected(scenario):
    def invalid_time(edge, departure_time):
        del edge, departure_time
        return float("inf")

    builder = DirectedPathBuilder(
        scenario.edges,
        travel_time_provider=invalid_time,
    )

    with pytest.raises(InvalidTravelTimeError):
        builder.shortest_path(
            "N1",
            "N2",
        )


def test_build_leg_preserves_endpoints(scenario):
    builder = DirectedPathBuilder(
        scenario.edges
    )

    leg = builder.build_leg(
        "N1",
        "N2",
    )

    assert leg.from_node == "N1"
    assert leg.to_node == "N2"
    assert leg.edge_ids == ("E12",)
    assert leg.edge_count == 1