import numpy as np
import pandas as pd

from src.data.datasets.dcrnn_hdf5 import DCRNNHDF5Config, load_dcrnn_hdf5
from src.data.datasets.pems_npz import PEMSNPZConfig, load_pems_npz


def test_dcrnn_hdf5_loader(tmp_path):
    import h5py

    path = tmp_path / "traffic.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("df")
        group.create_dataset("axis0", data=np.array([b"s1", b"s2"]))
        group.create_dataset(
            "axis1",
            data=np.array([b"2020-01-01 00:00:00", b"2020-01-01 00:05:00"]),
        )
        group.create_dataset(
            "block0_values",
            data=np.array([[10.0, 20.0], [11.0, np.nan]]),
        )

    result = load_dcrnn_hdf5(
        path,
        DCRNNHDF5Config(source_dataset="TEST"),
    )

    assert len(result) == 4
    assert set(result["sensor_id"]) == {"s1", "s2"}
    assert result["metric"].unique().tolist() == ["speed"]
    assert result["value"].isna().sum() == 1


def test_pems_npz_loader_preserves_features(tmp_path):
    path = tmp_path / "traffic.npz"
    values = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    np.savez(path, data=values)

    result = load_pems_npz(
        path,
        PEMSNPZConfig(
            source_dataset="TEST",
            feature_names=("flow", "occupancy", "speed"),
            start_timestamp="2020-01-01 00:00:00",
        ),
    )

    assert len(result) == 18
    assert set(result["metric"]) == {"flow", "occupancy", "speed"}
    assert result["sensor_id"].nunique() == 3
    assert result["timestamp"].nunique() == 2