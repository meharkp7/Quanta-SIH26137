from pathlib import Path

from src.data.metrla_spatial import (
    SensorDistance,
    SensorLocation,
    build_spatial_graph,
    load_metrla_spatial,
    validate_metrla_spatial,
    write_metrla_spatial,
)


def test_build_spatial_graph_keeps_only_sensor_to_sensor_edges():
    sensors = ["A", "B", "C"]

    locations = {
        "A": SensorLocation("A", 1.0, 2.0),
        "B": SensorLocation("B", 3.0, 4.0),
        "C": SensorLocation("C", 5.0, 6.0),
    }

    rows = [
        SensorDistance("A", "A", 0.0),
        SensorDistance("A", "B", 10.0),
        SensorDistance("B", "A", 11.0),
        SensorDistance("B", "C", 20.0),
        SensorDistance("C", "ROAD_NODE", 30.0),
        SensorDistance("ROAD_NODE", "A", 40.0),
    ]

    edges, profile = build_spatial_graph(
        sensor_ids=sensors,
        locations=locations,
        distance_rows=rows,
    )

    assert len(edges) == 3
    assert all(
        edge.from_sensor_id in sensors
        and edge.to_sensor_id in sensors
        and edge.from_sensor_id != edge.to_sensor_id
        for edge in edges
    )

    assert profile.sensor_count == 3
    assert profile.direct_sensor_pair_count == 4
    assert profile.non_self_sensor_pair_count == 3
    assert profile.self_pair_count == 1
    assert profile.reciprocal_sensor_pair_count == 1
    assert profile.asymmetric_sensor_pair_count == 1
    assert profile.minimum_non_self_distance == 10.0
    assert profile.maximum_non_self_distance == 20.0


def test_validation_accepts_complete_sensor_layer():
    sensors = ["A", "B"]

    locations = {
        "A": SensorLocation("A", 1.0, 2.0),
        "B": SensorLocation("B", 3.0, 4.0),
    }

    edges = [
        SensorDistance("A", "B", 10.0),
        SensorDistance("B", "A", 10.0),
    ]

    _, profile = build_spatial_graph(
        sensor_ids=sensors,
        locations=locations,
        distance_rows=edges,
    )

    issues = validate_metrla_spatial(
        sensors,
        locations,
        edges,
        profile,
    )

    assert issues == []


def test_write_metrla_spatial(tmp_path: Path):
    sensors = ["A", "B"]

    locations = {
        "A": SensorLocation("A", 1.0, 2.0),
        "B": SensorLocation("B", 3.0, 4.0),
    }

    edges = [
        SensorDistance("A", "B", 10.0),
    ]

    _, profile = build_spatial_graph(
        sensor_ids=sensors,
        locations=locations,
        distance_rows=edges,
    )

    output = write_metrla_spatial(
        tmp_path,
        sensors,
        locations,
        edges,
        profile,
        source_sensor_ids="graph_sensor_ids.txt",
        source_locations="graph_sensor_locations.csv",
        source_distances="distances_la_2012.csv",
    )

    assert output == tmp_path
    assert (tmp_path / "sensor_locations.csv").exists()
    assert (tmp_path / "sensor_edges.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "profile.json").exists()


def test_load_metrla_spatial(tmp_path: Path):
    sensor_ids = tmp_path / "graph_sensor_ids.txt"
    locations = tmp_path / "graph_sensor_locations.csv"
    distances = tmp_path / "distances_la_2012.csv"

    sensor_ids.write_text("A,B,C\n", encoding="utf-8")

    locations.write_text(
        "index,sensor_id,latitude,longitude\n"
        "0,A,1.0,2.0\n"
        "1,B,3.0,4.0\n"
        "2,C,5.0,6.0\n",
        encoding="utf-8",
    )

    distances.write_text(
        "from,to,cost\n"
        "A,A,0\n"
        "A,B,10\n"
        "B,A,11\n"
        "B,C,20\n"
        "ROAD_NODE,A,99\n",
        encoding="utf-8",
    )

    loaded_ids, loaded_locations, edges, profile = load_metrla_spatial(
        sensor_ids,
        locations,
        distances,
    )

    assert loaded_ids == ["A", "B", "C"]
    assert len(loaded_locations) == 3
    assert len(edges) == 3

    assert profile.sensor_count == 3
    assert profile.location_count == 3
    assert profile.distance_graph_node_count == 4
    assert profile.distance_graph_row_count == 5