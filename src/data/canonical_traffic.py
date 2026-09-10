"""Canonical real-traffic ingestion for the research data layer.

The routing contracts remain untouched. This module converts heterogeneous
traffic datasets into one explicit, loss-minimising representation while
preserving source-specific provenance and missingness.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TrafficObservation:
    source_dataset: str
    sensor_id: str
    sample_index: int
    timestamp: str | None
    metric: str
    value: float | None


@dataclass(frozen=True)
class TrafficDataset:
    dataset_id: str
    source_path: str
    sampling_interval_s: int | None
    sensor_ids: tuple[str, ...]
    metrics: tuple[str, ...]
    observations: tuple[TrafficObservation, ...]
    metadata: dict

    @property
    def sample_count(self) -> int:
        if not self.observations:
            return 0
        return max(o.sample_index for o in self.observations) + 1


def load_dataset(
    dataset_id: str,
    path: str | Path,
    *,
    interval_s: int | None = None,
    timestamps: Sequence[str] | None = None,
    npz_channels: dict[str, int] | None = None,
) -> TrafficDataset:
    """Load one supported real traffic source into the canonical schema.

    Supported source formats:
    - METR-LA / PEMS-BAY DCRNN-style HDF5: one speed matrix.
    - PEMS04 / PEMS08 NPZ: ``data`` shaped [time, sensor, feature].
    - Long CSV: timestamp,sensor_id,metric,value.
    - Wide CSV: timestamp,sensor1,sensor2,... (interpreted as speed).

    For NPZ sources, timestamps are intentionally optional. If absent, the
    dataset retains sample_index and records that timestamps were not supplied;
    no artificial calendar is invented.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".h5", ".hdf5", ".hdf"}:
        return _load_dcrnn_hdf5(dataset_id, p, interval_s, timestamps)
    if suffix == ".npz":
        return _load_npz(dataset_id, p, interval_s, timestamps, npz_channels)
    if suffix == ".csv":
        return _load_csv(dataset_id, p, interval_s)
    raise ValueError(f"Unsupported traffic source: {p.suffix}")


def _load_dcrnn_hdf5(dataset_id: str, path: Path, interval_s: int | None, timestamps: Sequence[str] | None) -> TrafficDataset:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install h5py to read HDF5 traffic datasets") from exc
    with h5py.File(path, "r") as h5:
        group = h5["df"] if "df" in h5 else h5
        required = {"axis0", "axis1", "block0_values"}
        if not required.issubset(group.keys()):
            raise ValueError("Expected DCRNN-style HDF5 datasets: axis0, axis1, block0_values")
        sensors = tuple(_decode(x) for x in group["axis0"][:])
        raw_ts = tuple(_decode(x) for x in group["axis1"][:])
        matrix = group["block0_values"][:]
    if matrix.ndim != 2 or matrix.shape != (len(raw_ts), len(sensors)):
        raise ValueError(f"Unexpected HDF5 matrix shape {matrix.shape}; expected (time, sensors)")
    ts = tuple(timestamps) if timestamps is not None else raw_ts
    if len(ts) != len(raw_ts):
        raise ValueError("Timestamp count does not match HDF5 sample count")
    obs = []
    for i, stamp in enumerate(ts):
        for j, sensor in enumerate(sensors):
            obs.append(TrafficObservation(dataset_id, sensor, i, stamp, "speed", _finite_or_none(matrix[i, j])))
    return _dataset(dataset_id, path, interval_s or 300, sensors, ("speed",), obs, {
        "format": "dcrnn_hdf5", "timestamp_source": "source_axis1",
    })


def _load_npz(dataset_id: str, path: Path, interval_s: int | None, timestamps: Sequence[str] | None, channels: dict[str, int] | None) -> TrafficDataset:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install numpy to read PEMS NPZ datasets") from exc
    with np.load(path, allow_pickle=False) as data:
        if "data" not in data:
            raise ValueError("PEMS NPZ must contain a 'data' array")
        arr = data["data"]
    if arr.ndim != 3:
        raise ValueError(f"Expected NPZ data shaped [time, sensor, feature], got {arr.shape}")
    n_time, n_sensor, n_features = arr.shape
    if timestamps is not None and len(timestamps) != n_time:
        raise ValueError("Timestamp count does not match NPZ sample count")
    channels = channels or {"flow": 0, "occupancy": 1, "speed": 2}
    bad = {k: v for k, v in channels.items() if v < 0 or v >= n_features}
    if bad:
        raise ValueError(f"Channel indexes outside NPZ feature dimension {n_features}: {bad}")
    sensors = tuple(str(i) for i in range(n_sensor))
    obs = []
    for i in range(n_time):
        stamp = timestamps[i] if timestamps is not None else None
        for metric, channel in channels.items():
            for j, sensor in enumerate(sensors):
                obs.append(TrafficObservation(dataset_id, sensor, i, stamp, metric, _finite_or_none(arr[i, j, channel])))
    return _dataset(dataset_id, path, interval_s or 300, sensors, tuple(channels.keys()), obs, {
        "format": "pems_npz", "timestamp_source": "external" if timestamps is not None else "not_supplied",
        "channel_mapping": channels,
    })


def _load_csv(dataset_id: str, path: Path, interval_s: int | None) -> TrafficDataset:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    if {"timestamp", "sensor_id", "metric", "value"}.issubset(fields):
        obs = []
        sample_by_timestamp: dict[str, int] = {}
        sensors: set[str] = set()
        metrics: set[str] = set()
        for row in rows:
            stamp = row["timestamp"]
            if stamp not in sample_by_timestamp:
                sample_by_timestamp[stamp] = len(sample_by_timestamp)
            sensor, metric = row["sensor_id"], row["metric"]
            sensors.add(sensor); metrics.add(metric)
            obs.append(TrafficObservation(dataset_id, sensor, sample_by_timestamp[stamp], stamp, metric, _parse_float(row["value"])))
        return _dataset(dataset_id, path, interval_s, tuple(sorted(sensors)), tuple(sorted(metrics)), obs, {
            "format": "long_csv", "timestamp_source": "csv_timestamp",
        })
    if "timestamp" in fields and len(fields) > 1:
        sensors = tuple(c for c in fields if c != "timestamp")
        obs = []
        for i, row in enumerate(rows):
            for sensor in sensors:
                obs.append(TrafficObservation(dataset_id, sensor, i, row["timestamp"], "speed", _parse_float(row[sensor])))
        return _dataset(dataset_id, path, interval_s, sensors, ("speed",), obs, {
            "format": "wide_csv", "timestamp_source": "csv_timestamp",
        })
    raise ValueError("CSV must be long (timestamp,sensor_id,metric,value) or wide (timestamp,sensor...)")


def validate_dataset(dataset: TrafficDataset) -> list[str]:
    """Return deterministic quality findings; empty means no hard findings."""
    findings: list[str] = []
    if not dataset.dataset_id.strip(): findings.append("dataset_id is empty")
    if not dataset.sensor_ids: findings.append("no sensors")
    if not dataset.metrics: findings.append("no metrics")
    seen: set[tuple] = set()
    for o in dataset.observations:
        key = (o.sensor_id, o.sample_index, o.metric)
        if key in seen:
            findings.append(f"duplicate observation: {key}")
            break
        seen.add(key)
        if o.value is not None and not math.isfinite(o.value):
            findings.append(f"non-finite value: {key}")
            break
        if o.metric in {"speed", "flow", "occupancy"} and o.value is not None and o.value < 0:
            findings.append(f"negative {o.metric}: {key}")
            break
    stamps = sorted({o.timestamp for o in dataset.observations if o.timestamp is not None})
    if len(stamps) != len({o.sample_index for o in dataset.observations if o.timestamp is not None}):
        findings.append("timestamp/sample-index cardinality mismatch")
    return findings


def profile_dataset(dataset: TrafficDataset) -> dict:
    """Create a compact, lossless-quality profile for audit and comparison."""
    by_metric: dict[str, list[float]] = {m: [] for m in dataset.metrics}
    missing: dict[str, int] = {m: 0 for m in dataset.metrics}
    for o in dataset.observations:
        if o.value is None:
            missing[o.metric] = missing.get(o.metric, 0) + 1
        else:
            by_metric.setdefault(o.metric, []).append(o.value)
    metrics = {}
    for metric, vals in by_metric.items():
        xs = sorted(vals)
        metrics[metric] = {
            "observed_count": len(vals),
            "missing_count": missing.get(metric, 0),
            "missing_fraction": missing.get(metric, 0) / max(1, len(vals) + missing.get(metric, 0)),
            "min": xs[0] if xs else None,
            "max": xs[-1] if xs else None,
            "mean": sum(xs) / len(xs) if xs else None,
            "p05": _quantile(xs, .05) if xs else None,
            "p50": _quantile(xs, .50) if xs else None,
            "p95": _quantile(xs, .95) if xs else None,
        }
    return {
        "schema_version": "canonical-traffic-1.0",
        "dataset_id": dataset.dataset_id,
        "source_path": dataset.source_path,
        "sample_count": dataset.sample_count,
        "sensor_count": len(dataset.sensor_ids),
        "metrics": list(dataset.metrics),
        "sampling_interval_s": dataset.sampling_interval_s,
        "timestamp_available": any(o.timestamp is not None for o in dataset.observations),
        "validation_findings": validate_dataset(dataset),
        "metric_profiles": metrics,
        "metadata": dataset.metadata,
    }


def write_canonical(dataset: TrafficDataset, output_dir: str | Path) -> Path:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "observations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_dataset", "sample_index", "timestamp", "sensor_id", "metric", "value"])
        writer.writeheader()
        for o in dataset.observations:
            writer.writerow(asdict(o))
    (out / "profile.json").write_text(json.dumps(profile_dataset(dataset), indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "schema_version": "canonical-traffic-1.0",
        "dataset_id": dataset.dataset_id,
        "sensor_ids": list(dataset.sensor_ids),
        "metrics": list(dataset.metrics),
        "metadata": dataset.metadata,
        "provenance": "observations.csv is a canonical representation; source values are not imputed or smoothed.",
    }, indent=2), encoding="utf-8")
    return out


def _dataset(dataset_id, path, interval_s, sensors, metrics, obs, metadata):
    dataset = TrafficDataset(dataset_id, str(path), interval_s, tuple(sensors), tuple(metrics), tuple(obs), dict(metadata))
    findings = validate_dataset(dataset)
    hard = [x for x in findings if x.startswith(("duplicate", "negative", "non-finite"))]
    if hard:
        raise ValueError("Invalid traffic dataset: " + "; ".join(hard))
    return dataset


def _parse_float(value: object) -> float | None:
    if value is None or str(value).strip().lower() in {"", "na", "nan", "null", "none"}:
        return None
    try: return _finite_or_none(float(value))
    except (TypeError, ValueError): return None


def _finite_or_none(value: object) -> float | None:
    x = float(value)
    return x if math.isfinite(x) else None


def _decode(value: object) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _quantile(xs: list[float], q: float) -> float:
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi: return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)