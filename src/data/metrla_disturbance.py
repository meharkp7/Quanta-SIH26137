from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DisturbanceConfig:
    min_relative_deviation: float = 0.20
    min_absolute_deviation: float = 8.0
    min_persistence_steps: int = 2
    recovery_window_steps: int = 6
    spatial_window_steps: int = 2
    min_spatial_sensors: int = 2

    # Prevent network-wide congestion regimes from becoming
    # localized disturbance candidates.
    max_network_affected_fraction: float = 0.25

    # Require the affected sensors to occupy a sufficiently small
    # fraction of the complete sensor network.
    max_event_sensors: int = 60


@dataclass(frozen=True)
class DisturbanceEvent:
    event_id: str
    start_index: int
    peak_index: int
    end_index: int
    duration_steps: int
    affected_sensors: tuple[str, ...]
    peak_sensor: str
    peak_relative_deviation: float
    peak_absolute_deviation: float
    spatial_sensor_count: int
    affected_fraction: float
    recovered: bool


def _timestamp_to_datetime(timestamp: str):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(
        int(timestamp) / 1_000_000_000,
        tz=timezone.utc,
    )


def _read_csv(path: str | Path):
    with Path(path).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        yield from csv.DictReader(handle)


def _load_baseline(path: str | Path):
    baseline = {}

    for row in _read_csv(path):
        baseline[
            (
                row["sensor_id"],
                int(row["weekday"]),
                int(row["hour"]),
            )
        ] = (
            float(row["mean"]),
            float(row["std"]),
        )

    return baseline


def _load_edges(path: str | Path):
    adjacency: dict[str, set[str]] = defaultdict(set)

    for row in _read_csv(path):
        source = row["from_sensor_id"]
        target = row["to_sensor_id"]

        if source != target:
            adjacency[source].add(target)

    return adjacency


def _load_series(path: str | Path):
    values: dict[str, dict[int, float]] = defaultdict(dict)
    timestamps: dict[int, str] = {}

    for row in _read_csv(path):
        if row["metric"] != "speed":
            continue

        index = int(row["sample_index"])

        timestamps[index] = row["timestamp"]
        values[row["sensor_id"]][index] = float(row["value"])

    return timestamps, values


def _score_deviation(
    value: float,
    baseline_mean: float,
    config: DisturbanceConfig,
):
    if value <= 0.0 or baseline_mean <= 0.0:
        return False, 0.0, 0.0

    absolute = baseline_mean - value
    relative = absolute / baseline_mean

    anomalous = (
        absolute >= config.min_absolute_deviation
        and relative >= config.min_relative_deviation
    )

    return anomalous, absolute, relative


def detect_disturbances(
    observations_path: str | Path,
    baseline_path: str | Path,
    edge_path: str | Path,
    output_dir: str | Path,
    config: DisturbanceConfig | None = None,
):
    """
    Extract empirical spatiotemporal traffic disturbances.

    These are NOT incident ground-truth labels.

    Network-wide congestion regimes are deliberately separated from
    localized disturbance candidates.
    """
    if config is None:
        config = DisturbanceConfig()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    baseline = _load_baseline(baseline_path)
    adjacency = _load_edges(edge_path)
    timestamps, values = _load_series(observations_path)

    sensors = sorted(values)
    sensor_count = len(sensors)

    # ---------------------------------------------------------------
    # 1. Temporal anomaly mask
    # ---------------------------------------------------------------

    anomalies: dict[int, set[str]] = defaultdict(set)
    scores = {}

    for sensor, series in values.items():
        for index, value in series.items():
            dt = _timestamp_to_datetime(timestamps[index])

            base = baseline.get(
                (
                    sensor,
                    dt.weekday(),
                    dt.hour,
                )
            )

            if base is None:
                continue

            mean_value, _ = base

            ok, absolute, relative = _score_deviation(
                value,
                mean_value,
                config,
            )

            if ok:
                anomalies[index].add(sensor)
                scores[(index, sensor)] = (
                    absolute,
                    relative,
                )

    # ---------------------------------------------------------------
    # 2. Remove obvious network-wide regimes
    # ---------------------------------------------------------------

    network_regimes: dict[int, set[str]] = defaultdict(set)
    persistent_candidates: dict[int, set[str]] = defaultdict(set)

    for index, affected in anomalies.items():
        fraction = len(affected) / sensor_count

        if fraction >= config.max_network_affected_fraction:
            network_regimes[index] = set(affected)

    # ---------------------------------------------------------------
    # 3. Persistence
    # ---------------------------------------------------------------

    for sensor, series in values.items():
        indices = sorted(series)

        run = []
        previous = None

        for index in indices:
            anomalous = (
                sensor in anomalies.get(index, set())
                and index not in network_regimes
            )

            if anomalous:
                if previous is None or index == previous + 1:
                    run.append(index)
                else:
                    run = [index]

                if len(run) >= config.min_persistence_steps:
                    for item in run:
                        persistent_candidates[item].add(sensor)
            else:
                run = []

            previous = index

    # ---------------------------------------------------------------
    # 4. Spatial connected components
    # ---------------------------------------------------------------

    candidate_nodes = {
        (index, sensor)
        for index in persistent_candidates
        for sensor in persistent_candidates[index]
    }

    visited = set()
    components = []

    def neighbours(node):
        index, sensor = node

        # Temporal continuity.
        for delta in range(
            -config.spatial_window_steps,
            config.spatial_window_steps + 1,
        ):
            if delta == 0:
                continue

            candidate = (index + delta, sensor)

            if candidate in candidate_nodes:
                yield candidate

        # Directed outgoing neighbours.
        for other_sensor in adjacency.get(sensor, set()):
            for delta in range(
                -config.spatial_window_steps,
                config.spatial_window_steps + 1,
            ):
                candidate = (
                    index + delta,
                    other_sensor,
                )

                if candidate in candidate_nodes:
                    yield candidate

        # Directed incoming neighbours.
        for source, targets in adjacency.items():
            if sensor not in targets:
                continue

            for delta in range(
                -config.spatial_window_steps,
                config.spatial_window_steps + 1,
            ):
                candidate = (
                    index + delta,
                    source,
                )

                if candidate in candidate_nodes:
                    yield candidate

    for node in sorted(candidate_nodes):
        if node in visited:
            continue

        queue = deque([node])
        visited.add(node)
        component = set()

        while queue:
            current = queue.popleft()
            component.add(current)

            for neighbour in neighbours(current):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        component_sensors = {
            sensor
            for _, sensor in component
        }

        if len(component_sensors) < config.min_spatial_sensors:
            continue

        if len(component_sensors) > config.max_event_sensors:
            continue

        components.append(component)

    # ---------------------------------------------------------------
    # 5. Build event records
    # ---------------------------------------------------------------

    events = []

    for number, component in enumerate(components, start=1):
        indices = sorted(
            index
            for index, _ in component
        )

        start_index = min(indices)
        end_index = max(indices)

        peak_node = max(
            component,
            key=lambda node: scores.get(
                node,
                (0.0, 0.0),
            )[1],
        )

        peak_index = peak_node[0]
        peak_sensor = peak_node[1]

        peak_absolute, peak_relative = scores.get(
            peak_node,
            (0.0, 0.0),
        )

        affected_sensors = tuple(
            sorted(
                {
                    sensor
                    for _, sensor in component
                }
            )
        )

        recovered = False

        recovery_end = (
            end_index
            + config.recovery_window_steps
        )

        for sensor in affected_sensors:
            series = values[sensor]

            for index in range(
                end_index + 1,
                recovery_end + 1,
            ):
                if index not in series:
                    continue

                timestamp = timestamps.get(index)

                if timestamp is None:
                    continue

                dt = _timestamp_to_datetime(timestamp)

                base = baseline.get(
                    (
                        sensor,
                        dt.weekday(),
                        dt.hour,
                    )
                )

                if base is None:
                    continue

                baseline_mean = base[0]

                if (
                    baseline_mean > 0
                    and series[index] >= 0.90 * baseline_mean
                ):
                    recovered = True
                    break

            if recovered:
                break

        events.append(
            DisturbanceEvent(
                event_id=f"METRLA-D-{number:06d}",
                start_index=start_index,
                peak_index=peak_index,
                end_index=end_index,
                duration_steps=end_index - start_index + 1,
                affected_sensors=affected_sensors,
                peak_sensor=peak_sensor,
                peak_relative_deviation=peak_relative,
                peak_absolute_deviation=peak_absolute,
                spatial_sensor_count=len(
                    affected_sensors
                ),
                affected_fraction=(
                    len(affected_sensors)
                    / sensor_count
                ),
                recovered=recovered,
            )
        )

    # ---------------------------------------------------------------
    # 6. Write events
    # ---------------------------------------------------------------

    event_path = output / "disturbance_events.csv"

    with event_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "event_id",
                "start_index",
                "peak_index",
                "end_index",
                "duration_steps",
                "affected_sensors",
                "peak_sensor",
                "peak_relative_deviation",
                "peak_absolute_deviation",
                "spatial_sensor_count",
                "affected_fraction",
                "recovered",
                "target_kind",
            ]
        )

        for event in events:
            writer.writerow(
                [
                    event.event_id,
                    event.start_index,
                    event.peak_index,
                    event.end_index,
                    event.duration_steps,
                    ";".join(
                        event.affected_sensors
                    ),
                    event.peak_sensor,
                    event.peak_relative_deviation,
                    event.peak_absolute_deviation,
                    event.spatial_sensor_count,
                    event.affected_fraction,
                    event.recovered,
                    "empirical_disturbance_candidate",
                ]
            )

    profile = {
        "schema": "metrla-spatiotemporal-disturbance-1.1",
        "dataset": "METR-LA",
        "semantics": {
            "ground_truth": False,
            "incident_labels": False,
            "causal_labels": False,
            "zero_values_used_as_events": False,
            "source_values_modified": False,
            "future_labels_used": False,
            "network_regimes_excluded": True,
        },
        "config": {
            "min_relative_deviation":
                config.min_relative_deviation,
            "min_absolute_deviation":
                config.min_absolute_deviation,
            "min_persistence_steps":
                config.min_persistence_steps,
            "recovery_window_steps":
                config.recovery_window_steps,
            "spatial_window_steps":
                config.spatial_window_steps,
            "min_spatial_sensors":
                config.min_spatial_sensors,
            "max_network_affected_fraction":
                config.max_network_affected_fraction,
            "max_event_sensors":
                config.max_event_sensors,
        },
        "statistics": {
            "sensor_count": sensor_count,
            "candidate_sensor_timepoints":
                len(candidate_nodes),
            "network_regime_timepoints":
                len(network_regimes),
            "spatial_components":
                len(components),
            "events":
                len(events),
            "recovered_events":
                sum(
                    event.recovered
                    for event in events
                ),
        },
    }

    with (
        output / "disturbance_profile.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            profile,
            handle,
            indent=2,
        )

    with (
        output / "manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema":
                    "metrla-spatiotemporal-disturbance-1.1",
                "dataset": "METR-LA",
                "inputs": {
                    "observations":
                        str(observations_path),
                    "baseline":
                        str(baseline_path),
                    "spatial_edges":
                        str(edge_path),
                },
                "outputs": [
                    "disturbance_events.csv",
                    "disturbance_profile.json",
                ],
                "warning": (
                    "These are empirical traffic disturbance "
                    "candidates, not verified incident labels."
                ),
            },
            handle,
            indent=2,
        )

    return events
