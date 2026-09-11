from __future__ import annotations

import pytest

from src.contracts.scenario import RoadEdge
from src.routing.cost_view import (
    CostView,
    InvalidEdgeCostError,
)
from src.routing.path_builder import DirectedPathBuilder


def _edge(
    *,
    edge_id: str = "E1",
    from_node: str = "N1",
    to_node: str = "N2",
    length_m: float = 100.0,
    speed_limit_mps: float = 10.0,
) -> RoadEdge:
    return RoadEdge(
        edge_id=edge_id,
        parent_road_id="R1",
        from_node=from_node,
        to_node=to_node,
        length_m=length_m,
        road_class="collector",
        speed_limit_mps=speed_limit_mps,
        lane_count=1,
        capacity_veh_per_hour=900.0,
        source_edge_id=None,
        provenance="test",
        open_by_default=True,
    )


def test_free_flow_cost_is_coherent() -> None:
    edge = _edge()

    cost = CostView().edge_cost(
        edge,
        departure_time_s=0.0,
    )

    assert cost.distance_m == pytest.approx(100.0)
    assert cost.free_flow_time_s == pytest.approx(10.0)
    assert cost.travel_time_s == pytest.approx(10.0)
    assert cost.congestion_delay_s == pytest.approx(0.0)


def test_dynamic_cost_exposes_congestion_delay() -> None:
    edge = _edge()

    def provider(
        edge: RoadEdge,
        departure_time_s: float,
    ) -> float:
        del departure_time_s

        return (
            edge.length_m
            / edge.speed_limit_mps
            + 7.5
        )

    cost = CostView(
        travel_time_provider=provider,
    ).edge_cost(edge)

    assert cost.free_flow_time_s == pytest.approx(10.0)
    assert cost.travel_time_s == pytest.approx(17.5)
    assert cost.congestion_delay_s == pytest.approx(7.5)


def test_path_cost_propagates_dynamic_departure_times() -> None:
    first = _edge(
        edge_id="E1",
        length_m=100.0,
        speed_limit_mps=10.0,
    )

    second = first.model_copy(
        update={
            "edge_id": "E2",
            "from_node": "N2",
            "to_node": "N3",
        }
    )

    observed_departures: list[float] = []

    def provider(
        edge: RoadEdge,
        departure_time_s: float,
    ) -> float:
        observed_departures.append(
            departure_time_s
        )

        return (
            edge.length_m
            / edge.speed_limit_mps
            + departure_time_s
        )

    cost = CostView(
        travel_time_provider=provider,
    ).path_cost(
        (first, second),
        departure_time_s=5.0,
    )

    assert observed_departures == pytest.approx(
        [5.0, 20.0]
    )

    assert cost.distance_m == pytest.approx(
        200.0
    )

    assert cost.free_flow_time_s == pytest.approx(
        20.0
    )

    assert cost.travel_time_s == pytest.approx(
        45.0
    )

    assert cost.congestion_delay_s == pytest.approx(
        25.0
    )


def test_congestion_cannot_make_travel_time_below_free_flow() -> None:
    edge = _edge()

    def provider(
        edge: RoadEdge,
        departure_time_s: float,
    ) -> float:
        del edge, departure_time_s
        return 1.0

    with pytest.raises(
        InvalidEdgeCostError,
        match="below free-flow",
    ):
        CostView(
            travel_time_provider=provider,
        ).edge_cost(edge)


def test_invalid_speed_is_rejected() -> None:
    with pytest.raises(ValueError, match="speed_limit_mps must be positive"):
        _edge(
            speed_limit_mps=0.0,
        )


def test_versions_are_preserved() -> None:
    view = CostView(
        graph_version="graph-17",
        cost_version="traffic-42",
        forecast_version="forecast-9",
    )

    assert view.graph_version == "graph-17"
    assert view.cost_version == "traffic-42"
    assert view.forecast_version == "forecast-9"


def test_path_builder_uses_cost_view_for_path_selection() -> None:
    first = _edge(
        edge_id="E1",
        from_node="N1",
        to_node="N2",
        length_m=100.0,
        speed_limit_mps=10.0,
    )

    direct = _edge(
        edge_id="E2",
        from_node="N1",
        to_node="N3",
        length_m=150.0,
        speed_limit_mps=10.0,
    )

    alternative = _edge(
        edge_id="E3",
        from_node="N2",
        to_node="N3",
        length_m=100.0,
        speed_limit_mps=10.0,
    )

    def provider(
        edge: RoadEdge,
        departure_time_s: float,
    ) -> float:
        del departure_time_s

        if edge.edge_id == "E2":
            return 30.0

        return (
            edge.length_m
            / edge.speed_limit_mps
        )

    view = CostView(
        travel_time_provider=provider,
    )

    builder = DirectedPathBuilder(
        (first, direct, alternative),
        cost_view=view,
    )

    result = builder.shortest_path(
        "N1",
        "N3",
    )

    assert result.edge_ids == (
        "E1",
        "E3",
    )

    assert result.distance_m == pytest.approx(
        200.0
    )

    assert result.travel_time_s == pytest.approx(
        20.0
    )

    assert result.free_flow_time_s == pytest.approx(
        20.0
    )

    assert result.congestion_delay_s == pytest.approx(
        0.0
    )


def test_closed_edges_remain_illegal() -> None:
    edge = _edge(
        edge_id="E1",
    )

    builder = DirectedPathBuilder(
        (edge,),
        closed_edge_ids=("E1",),
    )

    with pytest.raises(
        Exception,
        match="No directed path",
    ):
        builder.shortest_path(
            "N1",
            "N2",
        )