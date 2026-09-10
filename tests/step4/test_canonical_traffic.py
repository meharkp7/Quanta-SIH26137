import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.canonical_traffic import load_dataset, profile_dataset, write_canonical


def test_wide_csv_preserves_missingness_and_profile(tmp_path):
    p = tmp_path / "metr.csv"
    p.write_text("timestamp,a,b\n2026-01-01T00:00:00,60,50\n2026-01-01T00:05:00,,49\n", encoding="utf-8")
    ds = load_dataset("METR-LA", p, interval_s=300)
    assert ds.sample_count == 2
    assert len(ds.observations) == 4
    assert any(o.value is None for o in ds.observations)
    assert profile_dataset(ds)["timestamp_available"]


def test_pems_npz_maps_three_channels_without_inventing_timestamps(tmp_path):
    p = tmp_path / "PEMS04.npz"
    arr = np.zeros((2, 3, 3), dtype=float)
    arr[0, :, 0] = 10
    arr[0, :, 1] = 2
    arr[0, :, 2] = 55
    arr[1, :, 0] = 11
    arr[1, :, 1] = 3
    arr[1, :, 2] = 52
    np.savez(p, data=arr)
    ds = load_dataset("PEMS04", p, interval_s=300)
    assert ds.metrics == ("flow", "occupancy", "speed")
    assert not any(o.timestamp for o in ds.observations)
    assert ds.sample_count == 2


def test_long_csv_and_canonical_write(tmp_path):
    p = tmp_path / "traffic.csv"
    p.write_text("timestamp,sensor_id,metric,value\n2026-01-01T00:00:00,s1,speed,40\n2026-01-01T00:00:00,s1,flow,10\n", encoding="utf-8")
    ds = load_dataset("test", p, interval_s=300)
    out = write_canonical(ds, tmp_path / "out")
    assert (out / "observations.csv").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_version"] == "canonical-traffic-1.0"
    assert set(manifest["metrics"]) == {"speed", "flow"}


def test_negative_physical_value_is_rejected(tmp_path):
    p = tmp_path / "traffic.csv"
    p.write_text("timestamp,sensor_id,metric,value\n2026-01-01T00:00:00,s1,speed,-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="negative speed"):
        load_dataset("test", p)