from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SensorLocation:
    sensor_id: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class SensorDistance:
    from_sensor_id: str
    to_sensor_id: str
    distance: float


@dataclass(frozen=True)
class SpatialGraphProfile:
    dataset_id: str
    sensor_count: int
    location_count: int
    distance_graph_node_count: int
    distance_graph_row_count: int
    direct_sensor_pair_count: int
    non_self_sensor_pair_count: int
    self_pair_count: int
    directed_sensor_edge_count: int
    reciprocal_sensor_pair_count: int
    asymmetric_sensor_pair_count: int
    minimum_non_self_distance: float | None
    maximum_non_self_distance: float | None
    sensors_without_locations: list[str]
    sensors_without_distance_edges: list[str]


def _read_sensor_ids(sensor_ids_path: str | Path) -> list[str]:
    """
    Read the official DCRNN sensor ID file.

    The file is a single comma-separated line rather than one ID per line.
    """
    path = Path(sensor_ids_path)

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    sensor_ids: list[str] = []

    for token in text.replace("\n", ",").split(","):
        sensor_id = token.strip()
        if sensor_id:
            sensor_ids.append(sensor_id)

    # Preserve source ordering while rejecting accidental duplicates.
    return list(dict.fromkeys(sensor_ids))


def _read_sensor_locations(
    locations_path: str | Path,
) -> dict[str, SensorLocation]:
    path = Path(locations_path)

    locations: dict[str, SensorLocation] = {}

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        required = {"sensor_id", "latitude", "longitude"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing required location columns: {sorted(missing)}"
            )

        for row in reader:
            sensor_id = row["sensor_id"].strip()

            if not sensor_id:
                continue

            locations[sensor_id] = SensorLocation(
                sensor_id=sensor_id,
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
            )

    return locations


def _read_distance_graph(
    distances_path: str | Path,
) -> tuple[list[SensorDistance], set[str]]:
    """
    Read the complete source distance graph.

    The source file is intentionally retained conceptually as a road/network
    graph, not assumed to contain only traffic sensors.
    """
    path = Path(distances_path)

    rows: list[SensorDistance] = []
    node_ids: set[str] = set()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        required = {"from", "to", "cost"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing required distance columns: {sorted(missing)}"
            )

        for row in reader:
            source = row["from"].strip()
            target = row["to"].strip()
            distance = float(row["cost"])

            if not source or not target:
                continue

            node_ids.add(source)
            node_ids.add(target)

            rows.append(
                SensorDistance(
                    from_sensor_id=source,
                    to_sensor_id=target,
                    distance=distance,
                )
            )

    return rows, node_ids


def build_spatial_graph(
    sensor_ids: Iterable[str],
    locations: dict[str, SensorLocation],
    distance_rows: Iterable[SensorDistance],
) -> tuple[list[SensorDistance], SpatialGraphProfile]:
    """
    Derive the METR-LA sensor-level directed graph.

    Only distance rows where BOTH endpoints are official traffic sensors
    are included in the derived sensor graph.

    Self-pairs are retained in the source representation but excluded from
    the derived non-self edge set.
    """
    ordered_sensor_ids = list(dict.fromkeys(str(x) for x in sensor_ids))
    sensor_set = set(ordered_sensor_ids)

    direct_pairs: list[SensorDistance] = []

    source_rows = list(distance_rows)

    for row in source_rows:
        if (
            row.from_sensor_id in sensor_set
            and row.to_sensor_id in sensor_set
        ):
            direct_pairs.append(row)

    non_self_edges = [
        row
        for row in direct_pairs
        if row.from_sensor_id != row.to_sensor_id
    ]

    self_pairs = [
        row
        for row in direct_pairs
        if row.from_sensor_id == row.to_sensor_id
    ]

    # Treat (A,B) and (B,A) as a reciprocal pair for reporting only.
    # We do NOT collapse them because the source graph is directed.
    directed_lookup: dict[tuple[str, str], float] = {
        (row.from_sensor_id, row.to_sensor_id): row.distance
        for row in non_self_edges
    }

    reciprocal_pairs: set[frozenset[str]] = set()
    asymmetric_pairs: set[frozenset[str]] = set()

    for source, target in directed_lookup:
        reverse = (target, source)
        pair = frozenset((source, target))

        if reverse in directed_lookup:
            reciprocal_pairs.add(pair)

            if directed_lookup[(source, target)] != directed_lookup[reverse]:
                asymmetric_pairs.add(pair)

    sensors_with_edges = {
        row.from_sensor_id for row in non_self_edges
    } | {
        row.to_sensor_id for row in non_self_edges
    }

    sensors_without_edges = sorted(sensor_set - sensors_with_edges)
    sensors_without_locations = sorted(sensor_set - set(locations))

    non_self_distances = [row.distance for row in non_self_edges]

    profile = SpatialGraphProfile(
        dataset_id="METR-LA",
        sensor_count=len(ordered_sensor_ids),
        location_count=sum(
            1 for sensor_id in ordered_sensor_ids if sensor_id in locations
        ),
        distance_graph_node_count=0,
        distance_graph_row_count=len(source_rows),
        direct_sensor_pair_count=len(direct_pairs),
        non_self_sensor_pair_count=len(non_self_edges),
        self_pair_count=len(self_pairs),
        directed_sensor_edge_count=len(non_self_edges),
        reciprocal_sensor_pair_count=len(reciprocal_pairs),
        asymmetric_sensor_pair_count=len(asymmetric_pairs),
        minimum_non_self_distance=(
            min(non_self_distances) if non_self_distances else None
        ),
        maximum_non_self_distance=(
            max(non_self_distances) if non_self_distances else None
        ),
        sensors_without_locations=sensors_without_locations,
        sensors_without_distance_edges=sensors_without_edges,
    )

    return non_self_edges, profile


def load_metrla_spatial(
    sensor_ids_path: str | Path,
    locations_path: str | Path,
    distances_path: str | Path,
) -> tuple[list[str], dict[str, SensorLocation], list[SensorDistance], SpatialGraphProfile]:
    """
    Load the complete METR-LA spatial metadata and derive the sensor graph.
    """
    sensor_ids = _read_sensor_ids(sensor_ids_path)
    locations = _read_sensor_locations(locations_path)
    distance_rows, node_ids = _read_distance_graph(distances_path)

    sensor_edges, profile = build_spatial_graph(
        sensor_ids=sensor_ids,
        locations=locations,
        distance_rows=distance_rows,
    )

    profile = SpatialGraphProfile(
        **{
            **asdict(profile),
            "distance_graph_node_count": len(node_ids),
        }
    )

    return sensor_ids, locations, sensor_edges, profile


def validate_metrla_spatial(
    sensor_ids: list[str],
    locations: dict[str, SensorLocation],
    sensor_edges: list[SensorDistance],
    profile: SpatialGraphProfile,
) -> list[str]:
    """
    Validate structural properties of the spatial layer.

    These checks are deliberately conservative. They validate provenance and
    structural consistency rather than inventing assumptions about the road
    network.
    """
    issues: list[str] = []

    sensor_set = set(sensor_ids)

    if profile.sensor_count != len(sensor_set):
        issues.append(
            "sensor_count does not match the number of unique sensor IDs"
        )

    if profile.location_count != len(
        sensor_set.intersection(locations.keys())
    ):
        issues.append(
            "location_count does not match sensor/location intersection"
        )

    for row in sensor_edges:
        if row.from_sensor_id not in sensor_set:
            issues.append(
                f"sensor edge has unknown source sensor: {row.from_sensor_id}"
            )
            break

        if row.to_sensor_id not in sensor_set:
            issues.append(
                f"sensor edge has unknown target sensor: {row.to_sensor_id}"
            )
            break

        if row.from_sensor_id == row.to_sensor_id:
            issues.append(
                f"derived sensor edge contains self-pair: "
                f"{row.from_sensor_id}"
            )
            break

        if row.distance < 0:
            issues.append(
                f"sensor edge has negative distance: "
                f"{row.from_sensor_id}->{row.to_sensor_id}"
            )
            break

    if profile.direct_sensor_pair_count < profile.non_self_sensor_pair_count:
        issues.append(
            "non-self sensor-pair count cannot exceed direct sensor-pair count"
        )

    return issues


def write_metrla_spatial(
    output_dir: str | Path,
    sensor_ids: list[str],
    locations: dict[str, SensorLocation],
    sensor_edges: list[SensorDistance],
    profile: SpatialGraphProfile,
    *,
    source_sensor_ids: str,
    source_locations: str,
    source_distances: str,
) -> Path:
    """
    Write a provenance-preserving METR-LA spatial artifact.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    locations_path = output / "sensor_locations.csv"
    edges_path = output / "sensor_edges.csv"
    manifest_path = output / "manifest.json"
    profile_path = output / "profile.json"

    with locations_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sensor_id", "latitude", "longitude"])

        for sensor_id in sensor_ids:
            location = locations.get(sensor_id)

            if location is None:
                continue

            writer.writerow(
                [
                    location.sensor_id,
                    location.latitude,
                    location.longitude,
                ]
            )

    with edges_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "from_sensor_id",
                "to_sensor_id",
                "distance",
            ]
        )

        for row in sensor_edges:
            writer.writerow(
                [
                    row.from_sensor_id,
                    row.to_sensor_id,
                    row.distance,
                ]
            )

    manifest = {
        "schema": "metrla-spatial-1.0",
        "dataset_id": "METR-LA",
        "sensor_count": len(sensor_ids),
        "source": {
            "sensor_ids": source_sensor_ids,
            "locations": source_locations,
            "distances": source_distances,
        },
        "derived_artifacts": {
            "sensor_locations": "sensor_locations.csv",
            "sensor_edges": "sensor_edges.csv",
            "profile": "profile.json",
        },
        "semantics": {
            "sensor_edges": (
                "directed source distance relationships where both "
                "endpoints are official METR-LA traffic sensors"
            ),
            "self_pairs": (
                "retained in source metadata but excluded from "
                "derived sensor_edges.csv"
            ),
            "distance": (
                "source network distance value; not converted to "
                "travel time"
            ),
            "topology": (
                "no Euclidean, nearest-neighbor, grid, or Delaunay "
                "edges are fabricated"
            ),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    profile_path.write_text(
        json.dumps(asdict(profile), indent=2),
        encoding="utf-8",
    )

    return output