"""Loader for PeMS traffic NPZ files.

The adapter accepts common tensor layouts such as:
  (time, sensor)
  (time, sensor, feature)

Feature names can be supplied explicitly. No missing values are imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PEMSNPZConfig:
    source_dataset: str
    interval_s: int = 300
    feature_names: Sequence[str] = ("flow", "occupancy", "speed")
    start_timestamp: str | None = None


def _select_array(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    preferred = ("data", "x", "arr_0")
    for key in preferred:
        if key in npz.files:
            return np.asarray(npz[key])
    arrays = [k for k in npz.files if np.asarray(npz[k]).ndim >= 2]
    if not arrays:
        raise ValueError("NPZ contains no 2-D or 3-D traffic array")
    return np.asarray(npz[arrays[0]])


def load_pems_npz(
    path: str | Path,
    config: PEMSNPZConfig,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as npz:
        values = _select_array(npz)

    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3:
        raise ValueError(
            f"Expected traffic tensor with 2 or 3 dimensions, got {values.shape}"
        )

    time_count, sensor_count, feature_count = values.shape
    names = list(config.feature_names)
    if feature_count > len(names):
        raise ValueError(
            f"NPZ has {feature_count} features but only {len(names)} names were supplied"
        )
    names = names[:feature_count]

    if config.start_timestamp is None:
        timestamps = pd.RangeIndex(time_count, name="sample_index")
    else:
        timestamps = pd.date_range(
            start=config.start_timestamp,
            periods=time_count,
            freq=pd.to_timedelta(config.interval_s, unit="s"),
            name="timestamp",
        )

    rows = []
    for feature_index, metric in enumerate(names):
        block = values[:, :, feature_index]
        frame = pd.DataFrame(block, index=timestamps)
        frame.columns = [f"sensor_{i}" for i in range(sensor_count)]
        long = (
            frame.rename_axis("timestamp")
            .rename_axis("sensor_id", axis=1)
            .stack(future_stack=True)
            .rename("value")
            .reset_index()
        )
        long["metric"] = metric
        rows.append(long)

    result = pd.concat(rows, ignore_index=True)
    result["source_dataset"] = config.source_dataset

    if "sample_index" in result.columns:
        result["timestamp"] = pd.to_numeric(result["timestamp"], errors="raise")

    return result[
        ["source_dataset", "timestamp", "sensor_id", "metric", "value"]
    ].sort_values(["timestamp", "sensor_id", "metric"], kind="stable").reset_index(drop=True)


def load_pems04(path: str | Path, start_timestamp: str | None = None) -> pd.DataFrame:
    return load_pems_npz(
        path,
        PEMSNPZConfig(
            source_dataset="PEMS04",
            feature_names=("flow", "occupancy", "speed"),
            start_timestamp=start_timestamp,
        ),
    )


def load_pems08(path: str | Path, start_timestamp: str | None = None) -> pd.DataFrame:
    return load_pems_npz(
        path,
        PEMSNPZConfig(
            source_dataset="PEMS08",
            feature_names=("flow", "occupancy", "speed"),
            start_timestamp=start_timestamp,
        ),
    )