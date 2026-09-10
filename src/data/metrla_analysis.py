from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable


@dataclass(frozen=True)
class SensorStatistic:
    sensor_id: str
    observed_count: int
    missing_count: int
    missing_fraction: float
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    p05: float | None
    p95: float | None
    zero_count: int
    zero_fraction: float


@dataclass(frozen=True)
class TemporalProfile:
    sample_count: int
    sensor_count: int
    observed_count: int
    missing_count: int
    missing_fraction: float
    timestamp_count: int
    first_timestamp: str | None
    last_timestamp: str | None
    sampling_interval_s: float | None


@dataclass(frozen=True)
class ShockCandidate:
    sensor_id: str
    sample_index: int
    timestamp: str
    previous_speed: float
    current_speed: float
    delta_speed: float
    relative_change: float
    direction: str
    threshold: float


@dataclass(frozen=True)
class SpatialStatistics:
    sensor_count: int
    directed_edge_count: int
    reciprocal_pair_count: int
    asymmetric_pair_count: int
    mean_distance: float | None
    median_distance: float | None
    minimum_distance: float | None
    maximum_distance: float | None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _read_observations(path: str | Path):
    """
    Read the canonical METR-LA observation table.

    Expected columns:
        source_dataset,sample_index,timestamp,sensor_id,metric,value
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        required = {
            "source_dataset",
            "sample_index",
            "timestamp",
            "sensor_id",
            "metric",
            "value",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required observation columns: {sorted(missing)}"
            )

        for row in reader:
            value_text = row["value"].strip()

            yield {
                "source_dataset": row["source_dataset"],
                "sample_index": int(row["sample_index"]),
                "timestamp": row["timestamp"],
                "sensor_id": row["sensor_id"],
                "metric": row["metric"],
                "value": (
                    None
                    if value_text == ""
                    else float(value_text)
                ),
            }


def analyze_sensor_statistics(
    observations: Iterable[dict],
) -> list[SensorStatistic]:
    values: dict[str, list[float]] = defaultdict(list)
    missing: dict[str, int] = defaultdict(int)

    for row in observations:
        sensor_id = row["sensor_id"]
        value = row["value"]

        if value is None or math.isnan(value):
            missing[sensor_id] += 1
        else:
            values[sensor_id].append(value)

    sensor_ids = sorted(set(values) | set(missing))

    output = []

    for sensor_id in sensor_ids:
        sensor_values = values[sensor_id]
        missing_count = missing[sensor_id]
        total = len(sensor_values) + missing_count

        zero_count = sum(value == 0.0 for value in sensor_values)

        output.append(
            SensorStatistic(
                sensor_id=sensor_id,
                observed_count=len(sensor_values),
                missing_count=missing_count,
                missing_fraction=(
                    missing_count / total if total else 0.0
                ),
                minimum=min(sensor_values) if sensor_values else None,
                maximum=max(sensor_values) if sensor_values else None,
                mean=mean(sensor_values) if sensor_values else None,
                median=median(sensor_values) if sensor_values else None,
                p05=_percentile(sensor_values, 0.05),
                p95=_percentile(sensor_values, 0.95),
                zero_count=zero_count,
                zero_fraction=(
                    zero_count / len(sensor_values)
                    if sensor_values
                    else 0.0
                ),
            )
        )

    return output


def analyze_temporal_profile(
    observations: Iterable[dict],
) -> TemporalProfile:
    timestamps: set[str] = set()
    sensor_ids: set[str] = set()

    observed_count = 0
    missing_count = 0

    for row in observations:
        timestamps.add(row["timestamp"])
        sensor_ids.add(row["sensor_id"])

        value = row["value"]

        if value is None or math.isnan(value):
            missing_count += 1
        else:
            observed_count += 1

    ordered = sorted(timestamps)

    interval_s = None

    if len(ordered) >= 2:
        try:
            numeric = [int(value) for value in ordered]
            diffs = [
                b - a
                for a, b in zip(numeric, numeric[1:])
                if b > a
            ]

            if diffs:
                interval_s = median(diffs) / 1_000_000_000
        except ValueError:
            interval_s = None

    total = observed_count + missing_count

    return TemporalProfile(
        sample_count=len(timestamps),
        sensor_count=len(sensor_ids),
        observed_count=observed_count,
        missing_count=missing_count,
        missing_fraction=(
            missing_count / total if total else 0.0
        ),
        timestamp_count=len(timestamps),
        first_timestamp=ordered[0] if ordered else None,
        last_timestamp=ordered[-1] if ordered else None,
        sampling_interval_s=interval_s,
    )


def detect_shock_candidates(
    observations: Iterable[dict],
    *,
    absolute_drop: float = 10.0,
    relative_drop: float = 0.20,
) -> list[ShockCandidate]:
    """
    Detect abrupt empirical speed changes.

    IMPORTANT:
    These are candidate traffic shocks, NOT incident ground-truth labels.
    """
    series: dict[str, list[dict]] = defaultdict(list)

    for row in observations:
        if row["value"] is None:
            continue

        if math.isnan(row["value"]):
            continue

        series[row["sensor_id"]].append(row)

    candidates: list[ShockCandidate] = []

    for sensor_id, rows in series.items():
        rows.sort(key=lambda row: row["sample_index"])

        for previous, current in zip(rows, rows[1:]):
            previous_speed = previous["value"]
            current_speed = current["value"]

            delta = current_speed - previous_speed

            if previous_speed == 0:
                relative_change = 0.0
            else:
                relative_change = delta / previous_speed

            is_large_absolute_drop = (
                delta <= -abs(absolute_drop)
            )

            is_large_relative_drop = (
                relative_change <= -abs(relative_drop)
            )

            if is_large_absolute_drop or is_large_relative_drop:
                candidates.append(
                    ShockCandidate(
                        sensor_id=sensor_id,
                        sample_index=current["sample_index"],
                        timestamp=current["timestamp"],
                        previous_speed=previous_speed,
                        current_speed=current_speed,
                        delta_speed=delta,
                        relative_change=relative_change,
                        direction="drop",
                        threshold=max(
                            abs(absolute_drop),
                            abs(relative_drop),
                        ),
                    )
                )

    candidates.sort(
        key=lambda item: (
            item.sample_index,
            item.sensor_id,
        )
    )

    return candidates


def analyze_spatial_statistics(
    sensor_edges_path: str | Path,
) -> SpatialStatistics:
    distances: list[float] = []
    directed_edges = 0
    pairs: dict[frozenset[str], dict[tuple[str, str], float]] = defaultdict(dict)
    sensors: set[str] = set()

    with Path(sensor_edges_path).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "from_sensor_id",
            "to_sensor_id",
            "distance",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required spatial columns: {sorted(missing)}"
            )

        for row in reader:
            source = row["from_sensor_id"]
            target = row["to_sensor_id"]
            distance = float(row["distance"])

            sensors.add(source)
            sensors.add(target)
            distances.append(distance)
            directed_edges += 1

            pair = frozenset((source, target))
            pairs[pair][(source, target)] = distance

    reciprocal = 0
    asymmetric = 0

    for pair, directions in pairs.items():
        if len(pair) != 2:
            continue

        source, target = tuple(pair)

        if (source, target) in directions and (
            target,
            source,
        ) in directions:
            reciprocal += 1

            if directions[(source, target)] != directions[
                (target, source)
            ]:
                asymmetric += 1

    return SpatialStatistics(
        sensor_count=len(sensors),
        directed_edge_count=directed_edges,
        reciprocal_pair_count=reciprocal,
        asymmetric_pair_count=asymmetric,
        mean_distance=mean(distances) if distances else None,
        median_distance=median(distances) if distances else None,
        minimum_distance=min(distances) if distances else None,
        maximum_distance=max(distances) if distances else None,
    )


def write_analysis(
    output_dir: str | Path,
    temporal: TemporalProfile,
    sensor_stats: list[SensorStatistic],
    shocks: list[ShockCandidate],
    spatial: SpatialStatistics,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    (output / "temporal_profile.json").write_text(
        json.dumps(asdict(temporal), indent=2),
        encoding="utf-8",
    )

    with (output / "sensor_statistics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        if sensor_stats:
            fieldnames = list(asdict(sensor_stats[0]).keys())
        else:
            fieldnames = [
                "sensor_id",
                "observed_count",
                "missing_count",
                "missing_fraction",
                "minimum",
                "maximum",
                "mean",
                "median",
                "p05",
                "p95",
                "zero_count",
                "zero_fraction",
            ]

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for statistic in sensor_stats:
            writer.writerow(asdict(statistic))

    with (output / "shock_candidates.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = list(asdict(ShockCandidate(
            sensor_id="",
            sample_index=0,
            timestamp="",
            previous_speed=0.0,
            current_speed=0.0,
            delta_speed=0.0,
            relative_change=0.0,
            direction="",
            threshold=0.0,
        )).keys())

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for shock in shocks:
            writer.writerow(asdict(shock))

    (output / "spatial_statistics.json").write_text(
        json.dumps(asdict(spatial), indent=2),
        encoding="utf-8",
    )

    manifest = {
        "schema": "metrla-analysis-1.0",
        "dataset_id": "METR-LA",
        "artifacts": {
            "temporal_profile": "temporal_profile.json",
            "sensor_statistics": "sensor_statistics.csv",
            "shock_candidates": "shock_candidates.csv",
            "spatial_statistics": "spatial_statistics.json",
        },
        "semantics": {
            "shock_candidates": (
                "abrupt empirical speed changes; not incident "
                "ground-truth annotations"
            ),
            "zero_speed": (
                "retained as observed values; not automatically "
                "converted to missing"
            ),
            "imputation": "none",
            "smoothing": "none",
        },
    }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return output
