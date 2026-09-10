from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SensorLocation:
    sensor_id: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class SensorDistance:
    source_sensor_id: str
    target_sensor_id: str
    distance: float


@dataclass(frozen=True)
class SpatialGraphProfile:
    schema: str
    dataset_id: str
    sensor_count: int
    location_count: int
    distance_graph_node_count: int
    distance_graph_row_count: int
    sensor_edge_count: int
    self_pair_count: int
    reciprocal_pair_count: int
    asymmetric_pair_count: int
    min_distance: float | None
    max_distance: float | None


def _read_sensor_locations(path: str | Path) -> dict[str, SensorLocation]:
    path = Path(path)

    locations: dict[str, SensorLocation] = {}

    with path.open(newline="") as handle:
        reader = csv.reader(handle)

        for row in reader:
            if not row:
                continue

            # DCRNN PEMS-BAY format:
            # sensor_id, latitude, longitude
            if len(row) < 3:
                continue

            try:
                sensor_id = str(row[0]).strip()
                latitude = float(row[1])
                longitude = float(row[2])
            except ValueError:
                # Header row.
                continue

            locations[sensor_id] = SensorLocation(
                sensor_id=sensor_id,
                latitude=latitude,
                longitude=longitude,
            )

    return locations


def _read_distance_graph(
    path: str | Path,
) -> tuple[list[SensorDistance], set[str]]:
    path = Path(path)

    edges: list[SensorDistance] = []
    nodes: set[str] = set()

    with path.open(newline="") as handle:
        reader = csv.reader(handle)

        for row in reader:
            if len(row) < 3:
                continue

            try:
                source = str(row[0]).strip()
                target = str(row[1]).strip()
                distance = float(row[2])
            except ValueError:
                # Header row.
                continue

            nodes.add(source)
            nodes.add(target)

            edges.append(
                SensorDistance(
                    source_sensor_id=source,
                    target_sensor_id=target,
                    distance=distance,
                )
            )

    return edges, nodes


def _reciprocal_statistics(
    edges: list[SensorDistance],
) -> tuple[int, int]:
    pairs = {
        (edge.source_sensor_id, edge.target_sensor_id)
        for edge in edges
        if edge.source_sensor_id != edge.target_sensor_id
    }

    reciprocal_pairs = 0

    for source, target in pairs:
        if (target, source) in pairs:
            reciprocal_pairs += 1

    # Each reciprocal relationship appears twice in a directed graph.
    reciprocal_pairs //= 2

    asymmetric_pairs = 0

    undirected_pairs = {
        tuple(sorted((source, target)))
        for source, target in pairs
    }

    for source, target in undirected_pairs:
        forward = (source, target) in pairs
        backward = (target, source) in pairs

        if forward != backward:
            asymmetric_pairs += 1

    return reciprocal_pairs, asymmetric_pairs


def build_spatial_graph(
    locations: dict[str, SensorLocation],
    source_edges: list[SensorDistance],
    source_node_count: int,
) -> tuple[SpatialGraphProfile, list[SensorDistance]]:
    sensor_ids = set(locations)

    sensor_edges = [
        edge
        for edge in source_edges
        if edge.source_sensor_id in sensor_ids
        and edge.target_sensor_id in sensor_ids
        and edge.source_sensor_id != edge.target_sensor_id
    ]

    self_pair_count = sum(
        1
        for edge in source_edges
        if edge.source_sensor_id == edge.target_sensor_id
        and edge.source_sensor_id in sensor_ids
    )

    reciprocal_pair_count, asymmetric_pair_count = _reciprocal_statistics(
        sensor_edges
    )

    distances = [edge.distance for edge in sensor_edges]

    profile = SpatialGraphProfile(
        schema="spatial-graph-1.0",
        dataset_id="PEMS-BAY",
        sensor_count=len(sensor_ids),
        location_count=len(locations),
        distance_graph_node_count=source_node_count,
        distance_graph_row_count=len(source_edges),
        sensor_edge_count=len(sensor_edges),
        self_pair_count=self_pair_count,
        reciprocal_pair_count=reciprocal_pair_count,
        asymmetric_pair_count=asymmetric_pair_count,
        min_distance=min(distances) if distances else None,
        max_distance=max(distances) if distances else None,
    )

    return profile, sensor_edges


def load_pemsbay_spatial(
    location_path: str | Path,
    distance_path: str | Path,
) -> tuple[SpatialGraphProfile, dict[str, SensorLocation], list[SensorDistance]]:
    locations = _read_sensor_locations(location_path)
    source_edges, source_nodes = _read_distance_graph(distance_path)

    profile, sensor_edges = build_spatial_graph(
        locations,
        source_edges,
        len(source_nodes),
    )

    return profile, locations, sensor_edges


def validate_pemsbay_spatial(
    profile: SpatialGraphProfile,
    locations: dict[str, SensorLocation],
    edges: list[SensorDistance],
) -> list[str]:
    issues: list[str] = []

    if profile.dataset_id != "PEMS-BAY":
        issues.append(
            f"unexpected dataset_id: {profile.dataset_id}"
        )

    if profile.sensor_count != profile.location_count:
        issues.append(
            "sensor/location count mismatch"
        )

    edge_sensor_ids = {
        edge.source_sensor_id
        for edge in edges
    } | {
        edge.target_sensor_id
        for edge in edges
    }

    if not edge_sensor_ids.issubset(set(locations)):
        issues.append(
            "spatial edges contain sensor IDs absent from location metadata"
        )

    if any(edge.distance < 0 for edge in edges):
        issues.append("negative spatial distance detected")

    if profile.sensor_edge_count != len(edges):
        issues.append("profile edge count mismatch")

    return issues


def write_pemsbay_spatial(
    profile: SpatialGraphProfile,
    locations: dict[str, SensorLocation],
    edges: list[SensorDistance],
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "sensor_locations.csv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sensor_id",
                "latitude",
                "longitude",
            ]
        )

        for sensor_id in sorted(locations):
            location = locations[sensor_id]

            writer.writerow(
                [
                    location.sensor_id,
                    location.latitude,
                    location.longitude,
                ]
            )

    with (output_dir / "sensor_edges.csv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_sensor_id",
                "target_sensor_id",
                "distance",
            ]
        )

        for edge in edges:
            writer.writerow(
                [
                    edge.source_sensor_id,
                    edge.target_sensor_id,
                    edge.distance,
                ]
            )

    manifest = {
        "schema": "pemsbay-spatial-1.0",
        "dataset_id": "PEMS-BAY",
        "source": {
            "locations": "graph_sensor_locations_bay.csv",
            "distance_graph": "distances_bay_2017.csv",
        },
        "representation": {
            "locations": "sensor_locations.csv",
            "edges": "sensor_edges.csv",
        },
        "provenance": {
            "topology_source": "DCRNN precomputed road-network distances",
            "coordinates_source": "DCRNN PEMS-BAY sensor locations",
            "derived_edges": (
                "source distance rows whose endpoints are "
                "official PEMS-BAY traffic sensors"
            ),
            "self_pairs_excluded": True,
            "coordinate_topology_inference": False,
            "synthetic_edges": False,
        },
    }

    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    with (output_dir / "profile.json").open("w") as handle:
        json.dump(asdict(profile), handle, indent=2)

    return output_dir