"""Regression tests for src.data.osm_demand_generator.

The fixture networks are deliberately directed and imperfectly connected
(one-way "traps" that are reachable from the depot but cannot reach it
back, and vice versa) because that is exactly the shape of a real OSM
extract, and exactly the shape that broke a naive nearest-neighbour
reference-schedule builder in practice.
"""
from __future__ import annotations

import pytest

from src.contracts.core_types import NodeKind
from src.contracts.scenario import RoadEdge, RoadNode
from src.data.osm_demand_generator import (
    OSMDemandConfig,
    OSMDemandGenerationError,
    generate_osm_scenario,
)
from src.data.osm_ingestion import OSMNetwork
from src.routing.initial_solution import InitialSolutionBuilder, InitialSolutionConfig
from src.routing.route_evaluator import RouteEvaluator


def _node(node_id: str, x: float, y: float) -> RoadNode:
    return RoadNode(node_id=node_id, x_m=x, y_m=y, kind=NodeKind.JUNCTION, zone_id="osm")


def _edge(edge_id: str, u: str, v: str, length: float = 100.0) -> RoadEdge:
    return RoadEdge(
        edge_id=edge_id,
        parent_road_id=f"osm:road:{u}-{v}",
        from_node=u,
        to_node=v,
        length_m=length,
        road_class="local",
        speed_limit_mps=10.0,
        lane_count=1,
        capacity_veh_per_hour=900.0,
        provenance="openstreetmap",
    )


def make_grid_network(size: int = 4) -> OSMNetwork:
    """A fully bidirectional size x size grid: every node mutually reachable."""
    nodes = []
    edges = []
    for row in range(size):
        for col in range(size):
            nid = f"osm:n:{row}-{col}"
            nodes.append(_node(nid, x=col * 100.0, y=row * 100.0))

    def nid(row: int, col: int) -> str:
        return f"osm:n:{row}-{col}"

    for row in range(size):
        for col in range(size):
            if col + 1 < size:
                edges.append(_edge(f"e:{row}-{col}:R", nid(row, col), nid(row, col + 1)))
                edges.append(_edge(f"e:{row}-{col}:L", nid(row, col + 1), nid(row, col)))
            if row + 1 < size:
                edges.append(_edge(f"e:{row}-{col}:D", nid(row, col), nid(row + 1, col)))
                edges.append(_edge(f"e:{row}-{col}:U", nid(row + 1, col), nid(row, col)))

    return OSMNetwork(nodes=tuple(nodes), edges=tuple(edges), source_crs="EPSG:32643", projected=True)


def make_network_with_one_way_traps() -> OSMNetwork:
    """A well-connected core plus two one-way traps.

    `trap_out` is reachable FROM the core's hub but cannot reach it back
    (a dead-end service road). `trap_in` can reach the hub but the hub
    cannot reach it back (an inbound-only slip road). Neither belongs in
    the depot's strongly connected component, and a generator that didn't
    filter by SCC could pick one as a customer and then fail to find a
    path between it and another customer.
    """
    core = make_grid_network(size=4)
    nodes = list(core.nodes)
    edges = list(core.edges)

    hub = "osm:n:1-1"  # a well-connected interior node
    nodes.append(_node("osm:n:trap_out", x=999.0, y=999.0))
    edges.append(_edge("e:trap_out:in", hub, "osm:n:trap_out"))
    # deliberately no return edge

    nodes.append(_node("osm:n:trap_in", x=-999.0, y=-999.0))
    edges.append(_edge("e:trap_in:in", "osm:n:trap_in", hub))
    # deliberately no edge from hub to trap_in

    return OSMNetwork(nodes=tuple(nodes), edges=tuple(edges), source_crs="EPSG:32643", projected=True)


def make_unprojected_network() -> OSMNetwork:
    network = make_grid_network(size=3)
    return OSMNetwork(nodes=network.nodes, edges=network.edges, source_crs="EPSG:4326", projected=False)


def test_generates_valid_scenario_on_real_shaped_network():
    network = make_grid_network(size=4)
    result = generate_osm_scenario(
        network,
        OSMDemandConfig(customer_count=6, fleet_size=2, vehicle_capacity=20.0, seed=0),
        scenario_id="test-grid",
    )
    assert len(result.customer_node_ids) == 6
    assert result.depot_node_id not in result.customer_node_ids
    assert len(result.scenario.requests) == 6
    assert len(result.scenario.fleet) == 2
    depot_nodes = [n for n in result.scenario.nodes if n.node_id == result.depot_node_id]
    assert depot_nodes[0].kind == NodeKind.DEPOT


def test_generation_is_deterministic_given_same_seed():
    network = make_grid_network(size=4)
    config = OSMDemandConfig(customer_count=6, fleet_size=2, vehicle_capacity=20.0, seed=42)
    first = generate_osm_scenario(network, config, scenario_id="a")
    second = generate_osm_scenario(network, config, scenario_id="a")
    assert first.depot_node_id == second.depot_node_id
    assert first.customer_node_ids == second.customer_node_ids
    assert [r.demand for r in first.scenario.requests] == [
        r.demand for r in second.scenario.requests
    ]


def test_one_way_traps_are_never_selected_as_customers():
    network = make_network_with_one_way_traps()
    # Ask for every reachable-pair-safe node so the traps would be forced
    # in if they weren't excluded.
    result = generate_osm_scenario(
        network,
        OSMDemandConfig(customer_count=15, fleet_size=4, vehicle_capacity=20.0, seed=0),
        scenario_id="test-traps",
    )
    assert "osm:n:trap_out" not in result.customer_node_ids
    assert "osm:n:trap_in" not in result.customer_node_ids
    assert result.unreachable_candidates_skipped >= 2


def test_unprojected_network_is_rejected():
    network = make_unprojected_network()
    with pytest.raises(OSMDemandGenerationError, match="projected"):
        generate_osm_scenario(
            network,
            OSMDemandConfig(customer_count=2, fleet_size=1, vehicle_capacity=10.0),
            scenario_id="test-unprojected",
        )


def test_oversized_customer_count_is_rejected():
    network = make_grid_network(size=3)  # 9 nodes total, 8 non-depot at most
    with pytest.raises(OSMDemandGenerationError, match="customer_count"):
        generate_osm_scenario(
            network,
            OSMDemandConfig(customer_count=50, fleet_size=1, vehicle_capacity=10.0),
            scenario_id="test-oversized",
        )


def test_infeasible_capacity_is_rejected_with_actual_numbers():
    network = make_grid_network(size=4)
    with pytest.raises(OSMDemandGenerationError) as excinfo:
        generate_osm_scenario(
            network,
            OSMDemandConfig(
                customer_count=10,
                fleet_size=1,
                vehicle_capacity=1.0,
                demand_min=5.0,
                demand_max=5.0,
                seed=0,
            ),
            scenario_id="test-infeasible",
        )
    message = str(excinfo.value)
    assert "total demand" in message
    assert "total fleet capacity" in message


def test_generated_scenario_is_solvable_by_the_existing_routing_stack():
    network = make_grid_network(size=4)
    result = generate_osm_scenario(
        network,
        OSMDemandConfig(customer_count=8, fleet_size=3, vehicle_capacity=20.0, seed=0),
        scenario_id="test-solvable",
    )
    evaluator = RouteEvaluator(result.scenario)
    builder = InitialSolutionBuilder(
        result.scenario, evaluator, config=InitialSolutionConfig()
    )
    solution = builder.build()
    assert solution.feasible
    assert solution.unassigned_customer_ids == ()