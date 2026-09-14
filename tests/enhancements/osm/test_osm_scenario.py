import pytest

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
from src.data.osm_ingestion import OSMNetwork
from src.data.osm_scenario import OSMScenarioBuilder, OSMScenarioError, OSMSnapConfig


def make_network():
    nodes = (
        RoadNode(node_id="osm:n:1", x_m=0.0, y_m=0.0, kind="junction", zone_id="osm"),
        RoadNode(node_id="osm:n:2", x_m=100.0, y_m=0.0, kind="junction", zone_id="osm"),
        RoadNode(node_id="osm:n:3", x_m=200.0, y_m=0.0, kind="junction", zone_id="osm"),
    )
    edges = (
        RoadEdge(
            edge_id="osm:e:1",
            parent_road_id="osm:r:1",
            from_node="osm:n:1",
            to_node="osm:n:2",
            length_m=100.0,
            road_class="local",
            speed_limit_mps=10.0,
            lane_count=1,
            capacity_veh_per_hour=900.0,
            provenance="openstreetmap",
        ),
        RoadEdge(
            edge_id="osm:e:2",
            parent_road_id="osm:r:1",
            from_node="osm:n:2",
            to_node="osm:n:1",
            length_m=100.0,
            road_class="local",
            speed_limit_mps=10.0,
            lane_count=1,
            capacity_veh_per_hour=900.0,
            provenance="openstreetmap",
        ),
        RoadEdge(
            edge_id="osm:e:3",
            parent_road_id="osm:r:2",
            from_node="osm:n:2",
            to_node="osm:n:3",
            length_m=100.0,
            road_class="local",
            speed_limit_mps=10.0,
            lane_count=1,
            capacity_veh_per_hour=900.0,
            provenance="openstreetmap",
        ),
    )
    return OSMNetwork(
        nodes=nodes,
        edges=edges,
        source_crs="EPSG:32643",
        projected=True,
    )


def make_scenario():
    seeds = RandomSeeds(
        scenario_seed=1,
        road_seed=2,
        traffic_seed=3,
        incident_seed=4,
        window_seed=5,
        optimizer_seed=6,
        learning_seed=7,
    )
    request = Request(
        request_id="c1",
        original_customer_id="c1",
        original_x=999.0,
        original_y=999.0,
        access_node_id="synthetic-access",
        access_distance_m=0.0,
        demand=1.0,
        known_at_s=0.0,
        release_s=0.0,
        earliest_service_start_s=0.0,
        latest_service_start_s=1000.0,
        service_duration_s=30.0,
    )
    vehicle = Vehicle(
        vehicle_id="v1",
        capacity=10.0,
        start_node_id="synthetic-depot",
        depot_node_id="synthetic-depot",
    )
    return Scenario(
        schema_version="1.2",
        scenario_id="base",
        source_name="test",
        source_checksum=None,
        units=Units(),
        coordinate_transform=CoordinateTransform(
            scale=1.0,
            translation_x_m=0.0,
            translation_y_m=0.0,
            description="test",
        ),
        graph_version="base:graph:1",
        nodes=(
            RoadNode(
                node_id="synthetic-depot",
                x_m=0.0,
                y_m=0.0,
                kind="depot",
                zone_id="synthetic",
            ),
            RoadNode(
                node_id="synthetic-access",
                x_m=0.0,
                y_m=0.0,
                kind="customer_access",
                zone_id="synthetic",
            ),
        ),
        edges=(),
        requests=(request,),
        fleet=(vehicle,),
        seeds=seeds,
        generator_version="test",
        dataset_split="test",
        configuration_version="test",
    )


def test_build_reconstructs_and_validates_real_scenario_contract():
    result = OSMScenarioBuilder(make_network()).build(
        make_scenario(),
        request_coordinates={"c1": (98.0, 0.0)},
        vehicle_start_nodes={"v1": "osm:n:1"},
        vehicle_depot_nodes={"v1": "osm:n:1"},
    )

    s = result.scenario
    assert len(s.nodes) == 3
    assert len(s.edges) == 3
    assert s.requests[0].access_node_id == "osm:n:2"
    assert s.requests[0].access_distance_m == pytest.approx(2.0)
    assert s.fleet[0].start_node_id == "osm:n:1"
    assert s.fleet[0].depot_node_id == "osm:n:1"
    assert s.nodes[0].kind.value == "depot"
    assert s.nodes[1].kind.value == "customer_access"
    assert s.parent_instance_id == "base"
    assert s.source_checksum == result.network_fingerprint


def test_existing_osm_request_node_is_preserved_without_snapping():
    base = make_scenario()
    request = base.requests[0].model_copy(update={"access_node_id": "osm:n:3"})
    base = base.model_copy(update={"requests": (request,)})

    result = OSMScenarioBuilder(make_network()).build(
        base,
        vehicle_start_nodes={"v1": "osm:n:1"},
        vehicle_depot_nodes={"v1": "osm:n:1"},
    )
    assert result.scenario.requests[0].access_node_id == "osm:n:3"
    assert result.request_snap_distances_m == ()


def test_missing_coordinate_mapping_is_rejected_instead_of_guessing():
    with pytest.raises(OSMScenarioError, match="explicit projected coordinate"):
        OSMScenarioBuilder(make_network()).build(
            make_scenario(),
            vehicle_start_nodes={"v1": "osm:n:1"},
            vehicle_depot_nodes={"v1": "osm:n:1"},
        )


def test_vehicle_mapping_is_required_when_old_node_ids_are_not_osm():
    with pytest.raises(OSMScenarioError, match="start node"):
        OSMScenarioBuilder(make_network()).build(
            make_scenario(),
            request_coordinates={"c1": (0.0, 0.0)},
        )


def test_snap_distance_limit_is_enforced():
    with pytest.raises(OSMScenarioError, match="exceeding"):
        OSMScenarioBuilder(
            make_network(),
            snap_config=OSMSnapConfig(max_snap_distance_m=1.0),
        ).snap((50.0, 0.0))


def test_unprojected_network_is_rejected():
    network = make_network()
    network = OSMNetwork(
        nodes=network.nodes,
        edges=network.edges,
        source_crs="EPSG:4326",
        projected=False,
    )
    with pytest.raises(OSMScenarioError, match="projected"):
        OSMScenarioBuilder(network)


def test_fingerprint_is_deterministic():
    a = OSMScenarioBuilder(make_network()).network_fingerprint()
    b = OSMScenarioBuilder(make_network()).network_fingerprint()
    assert a == b
    assert len(a) == 64
