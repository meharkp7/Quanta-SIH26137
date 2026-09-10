"""Real-data-conditioned traffic profiles and episode generation.

This module deliberately separates three things that should never be conflated:

1. empirical observations (kept unchanged),
2. empirical traffic profiles/motifs mined from those observations, and
3. simulator-generated interventions such as closures/incidents.

METR-LA/PEMS-BAY-style HDF5 files can be read through pandas when the optional
runtime dependencies are installed. Generic timestamp x sensor CSV files are
also supported. No real dataset is copied into this repository.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrafficProfileConfig:
    source_name: str
    source_path: str
    interval_s: int = 300
    baseline_bins: int = 288
    anomaly_z: float = 3.5
    min_valid_fraction: float = 0.85


@dataclass(frozen=True)
class TrafficProfile:
    source_name: str
    source_path: str
    sensor_ids: tuple[str, ...]
    timestamps: tuple[str, ...]
    interval_s: int
    values: tuple[tuple[float | None, ...], ...]
    sensor_stats: tuple[dict, ...]
    anomaly_events: tuple[dict, ...]
    metadata: dict


def load_timestamp_sensor_csv(
    path: str | Path,
    timestamp_column: str = "timestamp",
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Load a CSV with one timestamp column and one column per sensor."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or timestamp_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain {timestamp_column!r}")

        sensors = [c for c in reader.fieldnames if c != timestamp_column]

        timestamps: list[str] = []
        rows: list[list[float | None]] = []

        for row in reader:
            timestamps.append(str(row[timestamp_column]))

            vals: list[float | None] = []

            for sensor in sensors:
                raw = row.get(sensor, "")

                if raw in (None, "", "NA", "NaN", "nan"):
                    vals.append(None)
                else:
                    try:
                        value = float(raw)
                        vals.append(value if math.isfinite(value) else None)
                    except ValueError:
                        vals.append(None)

            rows.append(vals)

    return timestamps, sensors, rows


def load_hdf5(
    path: str | Path,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Load a pandas/DCRNN-style HDF5 traffic table without pandas/tables.

    DCRNN-style files store a table under ``df`` with ``axis0`` (sensor ids),
    ``axis1`` (timestamps) and ``block0_values``.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Install h5py to read HDF5 traffic data"
        ) from exc

    with h5py.File(path, "r") as h5:
        group = h5["df"] if "df" in h5 else h5

        required = {"axis0", "axis1", "block0_values"}

        if not required.issubset(group.keys()):
            raise ValueError(
                "Unsupported HDF5 layout; expected "
                "df/axis0, df/axis1 and df/block0_values"
            )

        sensor_raw = group["axis0"][:]
        time_raw = group["axis1"][:]
        matrix = group["block0_values"][:]

    sensors = [_decode_scalar(x) for x in sensor_raw]
    timestamps = [_decode_scalar(x) for x in time_raw]

    values = []

    for row in matrix:
        vals = []

        for value in row:
            v = float(value)
            vals.append(v if math.isfinite(v) else None)

        values.append(vals)

    return timestamps, sensors, values


def _decode_scalar(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def load_real_traffic(
    path: str | Path,
    timestamp_column: str = "timestamp",
) -> tuple[list[str], list[str], list[list[float | None]]]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in {".h5", ".hdf5", ".hdf"}:
        return load_hdf5(p)

    if suffix == ".csv":
        return load_timestamp_sensor_csv(p, timestamp_column)

    raise ValueError(
        f"Unsupported traffic file type: {p.suffix}; "
        "use CSV or HDF5"
    )


def build_profile(config: TrafficProfileConfig) -> TrafficProfile:
    timestamps, sensors, values = load_real_traffic(config.source_path)

    if not timestamps or not sensors:
        raise ValueError("Traffic source contains no observations")

    stats: list[dict] = []

    for j, sensor in enumerate(sensors):
        series = [
            row[j]
            for row in values
            if row[j] is not None
        ]

        if len(series) / len(values) < config.min_valid_fraction:
            quality = "sparse"
        else:
            quality = "usable"

        mean = statistics.fmean(series) if series else None
        median = statistics.median(series) if series else None
        stdev = (
            statistics.stdev(series)
            if len(series) > 1
            else 0.0
        )

        q05 = _quantile(series, 0.05) if series else None
        q95 = _quantile(series, 0.95) if series else None

        stats.append(
            {
                "sensor_id": sensor,
                "valid_fraction": len(series) / len(values),
                "quality": quality,
                "mean": mean,
                "median": median,
                "std": stdev,
                "q05": q05,
                "q95": q95,
                "min": min(series) if series else None,
                "max": max(series) if series else None,
            }
        )

    anomalies = _mine_anomalies(
        timestamps,
        sensors,
        values,
        config.anomaly_z,
    )

    return TrafficProfile(
        source_name=config.source_name,
        source_path=str(config.source_path),
        sensor_ids=tuple(sensors),
        timestamps=tuple(timestamps),
        interval_s=config.interval_s,
        values=tuple(tuple(row) for row in values),
        sensor_stats=tuple(stats),
        anomaly_events=tuple(anomalies),
        metadata={
            "source_kind": "real_observed_traffic",
            "observation_count": len(values),
            "sensor_count": len(sensors),
            "interval_s": config.interval_s,
            "anomaly_method": "rolling-difference-zscore",
            "anomaly_note": (
                "Anomalies are empirical traffic-change candidates, "
                "not incident ground truth."
            ),
        },
    )


def write_profile(
    profile: TrafficProfile,
    output_dir: str | Path,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "profile.json").write_text(
        json.dumps(
            {
                "schema_version": "empirical-traffic-1.0",
                "source_name": profile.source_name,
                "source_path": profile.source_path,
                "sensor_ids": list(profile.sensor_ids),
                "timestamps": list(profile.timestamps),
                "interval_s": profile.interval_s,
                "sensor_stats": list(profile.sensor_stats),
                "anomaly_events": list(profile.anomaly_events),
                "metadata": profile.metadata,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    with (out / "observations.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "timestamp",
                "sensor_id",
                "value",
            ]
        )

        for i, ts in enumerate(profile.timestamps):
            for j, sensor in enumerate(profile.sensor_ids):
                writer.writerow(
                    [
                        ts,
                        sensor,
                        profile.values[i][j],
                    ]
                )

    (out / "README.json").write_text(
        json.dumps(
            {
                "empirical_truth": "observations.csv",
                "profile": "profile.json",
                "warning": (
                    "Do not label mined anomalies as incidents "
                    "without an external incident source."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return out


def _mine_anomalies(
    timestamps: list[str],
    sensors: list[str],
    values: list[list[float | None]],
    z_threshold: float,
) -> list[dict]:
    events: list[dict] = []

    for j, sensor in enumerate(sensors):
        series = [row[j] for row in values]

        diffs = [None] + [
            (
                None
                if series[i] is None or series[i - 1] is None
                else series[i] - series[i - 1]
            )
            for i in range(1, len(series))
        ]

        valid = [
            abs(x)
            for x in diffs
            if x is not None
        ]

        if len(valid) < 10:
            continue

        med = statistics.median(valid)

        mad = statistics.median(
            [
                abs(x - med)
                for x in valid
            ]
        ) or 1e-9

        scale = 1.4826 * mad

        for i, diff in enumerate(diffs):
            if diff is None:
                continue

            robust_z = abs(abs(diff) - med) / scale

            if robust_z >= z_threshold:
                events.append(
                    {
                        "sensor_id": sensor,
                        "timestamp": timestamps[i],
                        "change": diff,
                        "robust_z": robust_z,
                        "candidate_type": "abrupt_speed_change",
                    }
                )

    return events


def _quantile(
    values: list[float],
    q: float,
) -> float:
    if not values:
        raise ValueError("quantile requires values")

    xs = sorted(values)

    pos = (len(xs) - 1) * q

    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return xs[lo]

    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)