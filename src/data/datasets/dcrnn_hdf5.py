"""Loader for DCRNN-style traffic HDF5 files.

Supports the common METR-LA / PEMS-BAY layout used by DCRNN:
  /df/axis0
  /df/axis1
  /df/block0_values

The loader preserves the source values and does not impute or smooth them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DCRNNHDF5Config:
    source_dataset: str
    value_name: str = "speed"
    interval_s: int = 300
    timezone: Optional[str] = None


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def load_dcrnn_hdf5(
    path: str | Path,
    config: DCRNNHDF5Config,
) -> pd.DataFrame:
    """Load a DCRNN HDF5 file into canonical long-form observations.

    Returns columns:
        source_dataset, timestamp, sensor_id, metric, value
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for DCRNN-style HDF5 files. "
            "Install it with `pip install h5py`."
        ) from exc

    with h5py.File(path, "r") as h5:
        if "df" not in h5:
            raise ValueError(f"{path} does not contain the expected 'df' group")
        group = h5["df"]
        required = {"axis0", "axis1", "block0_values"}
        missing = required.difference(group.keys())
        if missing:
            raise ValueError(
                f"{path} is missing DCRNN datasets: {sorted(missing)}"
            )

        sensors = [_decode(x) for x in group["axis0"][()]]
        timestamps = [_decode(x) for x in group["axis1"][()]]
        values = np.asarray(group["block0_values"][()])

    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D value matrix, got shape {values.shape}")
    if values.shape != (len(timestamps), len(sensors)):
        raise ValueError(
            "HDF5 axis dimensions do not match block0_values: "
            f"values={values.shape}, timestamps={len(timestamps)}, sensors={len(sensors)}"
        )

    ts = pd.to_datetime(timestamps, errors="raise")
    if config.timezone:
        if ts.tz is None:
            ts = ts.tz_localize(config.timezone)
        else:
            ts = ts.tz_convert(config.timezone)

    frame = pd.DataFrame(values, index=ts, columns=[str(x) for x in sensors])
    frame.index.name = "timestamp"

    long = (
        frame.rename_axis("sensor_id", axis=1)
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
    )
    long["source_dataset"] = config.source_dataset
    long["metric"] = config.value_name

    return long[
        ["source_dataset", "timestamp", "sensor_id", "metric", "value"]
    ].sort_values(["timestamp", "sensor_id"], kind="stable").reset_index(drop=True)


def load_metr_la(path: str | Path) -> pd.DataFrame:
    return load_dcrnn_hdf5(
        path,
        DCRNNHDF5Config(source_dataset="METR-LA"),
    )


def load_pems_bay(path: str | Path) -> pd.DataFrame:
    return load_dcrnn_hdf5(
        path,
        DCRNNHDF5Config(source_dataset="PEMS-BAY"),
    )