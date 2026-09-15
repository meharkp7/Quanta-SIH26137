"""Regression tests for the dead-end / cul-de-sac U-turn handling.

Real OSM extracts contain genuine dead ends (a road that terminates at a
single node with only one connecting street). A real vehicle that drives
in must be able to turn around and drive back out. The pre-existing
``SumoExporter._write_connections`` forbade U-turns at *every* junction
unconditionally -- correct for avoiding illegal mid-block U-turns, wrong
for a legitimate cul-de-sac where the U-turn is the only physically
possible continuation. ``RouteBuilder._path_without_u_turns`` must permit
exactly the same exception, or its search disagrees with what the
compiled SUMO network (netconvert output) actually allows.

There was no existing test coverage for either module before this file.
"""
from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from src.contracts.core_types import NodeKind
from src.contracts.scenario import (
    CoordinateTransform,
    RandomSeeds,
    Request,
    RoadEdge,
    RoadNode,
    Scenario,
    Units,
    Vehicle,
)
from src.sim.route_builder import RouteBuilder
from src.sim.sumo_exporter import SumoExporter


def _node(node_id: str, x: float, y: float) -> RoadNode:
    return RoadNode(node_id=node_id, x_m=x, y_m=y, kind=NodeKind.JUNCTION, zone_id="test")


def _edge(edge_id: str, u: str, v: str, length: float = 100.0) -> RoadEdge:
    return RoadEdge(
        edge_id=edge_id,
        parent_road_id=f"road:{u}-{v}",
        from_node=u,
        to_node=v,
        length_m=length,
        road_class="local",
        speed_limit_mps=10.0,
        lane_count=1,
        capacity_veh_per_hour=900.0,
        provenance="test",
    )


def _dead_end_scenario() -> Scenario:
    """A: dead end (only connects to B). B: through junction connecting
    A, C, and D. C and D are further reachable nodes.

    A <-> B <-> C
          |
          v
          D  (B <-> D)
    """
    nodes = [
        _node("A", 0.0, 0.0),
        _node("B", 100.0, 0.0),
        _node("C", 200.0, 0.0),
        _node("D", 100.0, 100.0),
    ]
    edges = [
        _edge("e:A-B", "A", "B"),
        _edge("e:B-A", "B", "A"),
        _edge("e:B-C", "B", "C"),
        _edge("e:C-B", "C", "B"),
        _edge("e:B-D", "B", "D"),
        _edge("e:D-B", "D", "B"),
    ]
    seeds = RandomSeeds(
        scenario_seed=1, road_seed=2, traffic_seed=3, incident_seed=4,
        window_seed=5, optimizer_seed=6, learning_seed=7,
    )
    request = Request(
        request_id="c1", original_customer_id="c1", original_x=0.0, original_y=0.0,
        access_node_id="A", access_distance_m=0.0, demand=1.0, known_at_s=0.0,
        release_s=0.0, earliest_service_start_s=0.0, latest_service_start_s=1000.0,
        service_duration_s=30.0,
    )
    vehicle = Vehicle(vehicle_id="v1", capacity=10.0, start_node_id="C", depot_node_id="C")
    return Scenario(
        schema_version="1.2", scenario_id="dead-end-test", source_name="test",
        source_checksum=None, units=Units(),
        coordinate_transform=CoordinateTransform(
            scale=1.0, translation_x_m=0.0, translation_y_m=0.0, description="test"
        ),
        graph_version="test", nodes=tuple(nodes), edges=tuple(edges),
        requests=(request,), fleet=(vehicle,), seeds=seeds,
        generator_version="test", dataset_split="test", configuration_version="test",
    )


def test_dead_end_requires_uturn_to_leave_and_is_now_resolvable():
    scenario = _dead_end_scenario()
    builder = RouteBuilder(scenario, route_plan=None, sumo_mapping={})

    # Vehicle just arrived at A via B->A; must now continue to C. The only
    # way out of A is to reverse onto A->B -- this must now succeed instead
    # of raising "No SUMO-compatible no-U-turn path exists".
    path = builder._path_without_u_turns("A", "C", previous_edge_id="e:B-A")
    assert path == ["e:A-B", "e:B-C"]


def test_through_junction_still_forbids_unnecessary_uturn():
    scenario = _dead_end_scenario()
    builder = RouteBuilder(scenario, route_plan=None, sumo_mapping={})

    # Vehicle arrived at B via A->B and must continue to D. B is a real
    # through junction (A, C, D all reachable), so reversing back to A must
    # still never be chosen even though it would be a "shorter" 1-hop path
    # in the unconstrained graph.
    path = builder._path_without_u_turns("B", "D", previous_edge_id="e:A-B")
    assert path == ["e:B-D"]
    assert "e:B-A" not in path


def test_truly_unreachable_target_still_raises():
    scenario = _dead_end_scenario()
    builder = RouteBuilder(scenario, route_plan=None, sumo_mapping={})
    with pytest.raises(ValueError, match="No SUMO-compatible"):
        builder._path_without_u_turns("A", "nonexistent-node", previous_edge_id="e:B-A")


def test_sumo_export_permits_uturn_only_at_the_dead_end(tmp_path):
    scenario = _dead_end_scenario()
    exporter = SumoExporter(scenario, tmp_path)
    con_path = exporter._write_connections()

    root = ET.parse(con_path).getroot()
    pairs = {(c.get("from"), c.get("to")) for c in root.findall("connection")}

    from_a_map = exporter.edge_mapping.get("e:A-B", "e:A-B")
    to_b_map = exporter.edge_mapping.get("e:B-A", "e:B-A")
    # The dead-end U-turn (arrive at A via B->A, must leave via A->B) IS
    # permitted, because forbidding it would strand every vehicle that
    # enters A.
    assert (to_b_map, from_a_map) in pairs

    # But at the real through-junction B, arriving via A->B must NOT be
    # allowed to immediately reverse back onto B->A -- other options (B->C,
    # B->D) exist there.
    a_to_b = exporter.edge_mapping.get("e:A-B", "e:A-B")
    b_to_a = exporter.edge_mapping.get("e:B-A", "e:B-A")
    assert (a_to_b, b_to_a) not in pairs