from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BaselineProfile:
    sample_count: int
    sensor_count: int
    observed_count: int
    missing_count: int
    hour_bins: int
    weekday_bins: int
    global_mean: float
    global_std: float


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return values[lower]

    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _timestamp_to_datetime(timestamp: str):
    """
    METR-LA timestamps are preserved in the canonical layer as nanoseconds
    since Unix epoch.

    We deliberately interpret them here only for temporal grouping.
    The canonical observations remain unchanged.
    """
    from datetime import datetime, timezone

    value = int(timestamp)

    # METR-LA uses nanoseconds since epoch.
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc)


def _read_observations(
    observations_path: str | Path,
) -> Iterable[dict[str, str]]:
    with Path(observations_path).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        yield from csv.DictReader(handle)


def build_temporal_baseline(
    observations_path: str | Path,
    output_dir: str | Path,
) -> BaselineProfile:
    """
    Build an empirical sensor × weekday × hour baseline.

    Important semantics:
    - source observations are never modified;
    - zero values remain zero in the canonical data;
    - zero values are excluded from the statistical baseline because the
      previous audit showed long zero runs that are unlikely to represent
      ordinary traffic speed;
    - no interpolation, smoothing, or future information is used;
    - grouping is performed using UTC because that is the timestamp
      representation currently preserved in the canonical dataset.

    The baseline is intentionally descriptive. It is not a causal
    incident label and does not infer incidents from the observations.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # sensor -> (weekday, hour) -> observed positive speeds
    context_values: dict[
        str,
        dict[tuple[int, int], list[float]],
    ] = defaultdict(lambda: defaultdict(list))

    # sensor -> hour -> values
    hour_values: dict[
        str,
        dict[int, list[float]],
    ] = defaultdict(lambda: defaultdict(list))

    # sensor -> weekday -> values
    weekday_values: dict[
        str,
        dict[int, list[float]],
    ] = defaultdict(lambda: defaultdict(list))

    sensors: set[str] = set()
    timestamps: set[str] = set()

    observed_count = 0
    missing_count = 0
    positive_values: list[float] = []

    for row in _read_observations(observations_path):
        sensor_id = row["sensor_id"]
        timestamp = row["timestamp"]

        sensors.add(sensor_id)
        timestamps.add(timestamp)

        value_text = row["value"]

        if value_text == "":
            missing_count += 1
            continue

        value = float(value_text)

        # Preserve zero observations in canonical data, but do not allow
        # the suspicious zero encoding to dominate the normal-speed baseline.
        if value <= 0.0:
            continue

        observed_count += 1
        positive_values.append(value)

        dt = _timestamp_to_datetime(timestamp)

        weekday = dt.weekday()
        hour = dt.hour

        context_values[sensor_id][(weekday, hour)].append(value)
        hour_values[sensor_id][hour].append(value)
        weekday_values[sensor_id][weekday].append(value)

    # ------------------------------------------------------------------
    # sensor × weekday × hour baseline
    # ------------------------------------------------------------------

    context_path = output / "sensor_context_baseline.csv"

    with context_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "sensor_id",
                "weekday",
                "hour",
                "count",
                "mean",
                "median",
                "p05",
                "p95",
                "std",
            ]
        )

        for sensor_id in sorted(context_values):
            for weekday, hour in sorted(context_values[sensor_id]):
                values = context_values[sensor_id][(weekday, hour)]

                mean_value = sum(values) / len(values)

                if len(values) > 1:
                    variance = sum(
                        (value - mean_value) ** 2
                        for value in values
                    ) / (len(values) - 1)
                    std_value = math.sqrt(variance)
                else:
                    std_value = 0.0

                writer.writerow(
                    [
                        sensor_id,
                        weekday,
                        hour,
                        len(values),
                        mean_value,
                        _percentile(values, 0.50),
                        _percentile(values, 0.05),
                        _percentile(values, 0.95),
                        std_value,
                    ]
                )

    # ------------------------------------------------------------------
    # sensor × hour baseline
    # ------------------------------------------------------------------

    hour_path = output / "sensor_hour_baseline.csv"

    with hour_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "sensor_id",
                "hour",
                "count",
                "mean",
                "median",
                "p05",
                "p95",
                "std",
            ]
        )

        for sensor_id in sorted(hour_values):
            for hour in sorted(hour_values[sensor_id]):
                values = hour_values[sensor_id][hour]

                mean_value = sum(values) / len(values)

                if len(values) > 1:
                    variance = sum(
                        (value - mean_value) ** 2
                        for value in values
                    ) / (len(values) - 1)
                    std_value = math.sqrt(variance)
                else:
                    std_value = 0.0

                writer.writerow(
                    [
                        sensor_id,
                        hour,
                        len(values),
                        mean_value,
                        _percentile(values, 0.50),
                        _percentile(values, 0.05),
                        _percentile(values, 0.95),
                        std_value,
                    ]
                )

    # ------------------------------------------------------------------
    # sensor × weekday baseline
    # ------------------------------------------------------------------

    weekday_path = output / "sensor_weekday_baseline.csv"

    with weekday_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "sensor_id",
                "weekday",
                "count",
                "mean",
                "median",
                "p05",
                "p95",
                "std",
            ]
        )

        for sensor_id in sorted(weekday_values):
            for weekday in sorted(weekday_values[sensor_id]):
                values = weekday_values[sensor_id][weekday]

                mean_value = sum(values) / len(values)

                if len(values) > 1:
                    variance = sum(
                        (value - mean_value) ** 2
                        for value in values
                    ) / (len(values) - 1)
                    std_value = math.sqrt(variance)
                else:
                    std_value = 0.0

                writer.writerow(
                    [
                        sensor_id,
                        weekday,
                        len(values),
                        mean_value,
                        _percentile(values, 0.50),
                        _percentile(values, 0.05),
                        _percentile(values, 0.95),
                        std_value,
                    ]
                )

    # ------------------------------------------------------------------
    # Global profile
    # ------------------------------------------------------------------

    global_mean = (
        sum(positive_values) / len(positive_values)
        if positive_values
        else math.nan
    )

    if len(positive_values) > 1:
        variance = sum(
            (value - global_mean) ** 2
            for value in positive_values
        ) / (len(positive_values) - 1)
        global_std = math.sqrt(variance)
    else:
        global_std = 0.0

    profile = BaselineProfile(
        sample_count=len(timestamps),
        sensor_count=len(sensors),
        observed_count=observed_count,
        missing_count=missing_count,
        hour_bins=len(sensors) * 24,
        weekday_bins=len(sensors) * 7,
        global_mean=global_mean,
        global_std=global_std,
    )

    with (output / "baseline_profile.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema": "metrla-temporal-baseline-1.0",
                "semantics": {
                    "source_values_unchanged": True,
                    "zero_values_preserved": True,
                    "zero_values_used_in_baseline": False,
                    "imputation": False,
                    "smoothing": False,
                    "timestamp_grouping": "UTC",
                },
                "statistics": {
                    "sample_count": profile.sample_count,
                    "sensor_count": profile.sensor_count,
                    "observed_count": profile.observed_count,
                    "missing_count": profile.missing_count,
                    "hour_bins": profile.hour_bins,
                    "weekday_bins": profile.weekday_bins,
                    "global_mean": profile.global_mean,
                    "global_std": profile.global_std,
                },
            },
            handle,
            indent=2,
        )

    with (output / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema": "metrla-temporal-baseline-1.0",
                "dataset": "METR-LA",
                "inputs": {
                    "observations": str(observations_path),
                },
                "outputs": [
                    "baseline_profile.json",
                    "sensor_context_baseline.csv",
                    "sensor_hour_baseline.csv",
                    "sensor_weekday_baseline.csv",
                ],
                "notes": [
                    "Baseline is empirical and descriptive.",
                    "Canonical observations are not modified.",
                    "Zero-valued observations are preserved but excluded "
                    "from the normal-speed baseline.",
                    "No future observations are used to alter individual "
                    "historical records.",
                ],
            },
            handle,
            indent=2,
        )

    return profile
