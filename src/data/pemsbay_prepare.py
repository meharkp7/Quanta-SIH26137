from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


RAW = Path("data/raw/PEMS-BAY")
OUT = Path("data/canonical/PEMS-BAY")
SPATIAL = OUT / "spatial"

OUT.mkdir(parents=True, exist_ok=True)
SPATIAL.mkdir(parents=True, exist_ok=True)


def decode(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def md5(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================
# TRAFFIC DATA
# ================================================================

traffic_path = RAW / "pems-bay.h5"

with h5py.File(traffic_path, "r") as f:
    sensor_ids = [
        decode(x)
        for x in f["speed/axis0"][:]
    ]

    timestamps = f["speed/axis1"][:].astype(np.int64)

    values = f["speed/block0_values"][:].astype(float)


# ================================================================
# TIMESTAMP AUDIT
# ================================================================

diffs = np.diff(timestamps)

unique_diffs, diff_counts = np.unique(
    diffs,
    return_counts=True,
)

timestamp_gaps = []

for delta, count in zip(
    unique_diffs,
    diff_counts,
):
    if delta != 300_000_000_000:
        timestamp_gaps.append(
            {
                "delta_ns": int(delta),
                "delta_seconds": float(
                    delta / 1_000_000_000
                ),
                "occurrences": int(count),
            }
        )


# ================================================================
# CANONICAL OBSERVATIONS
# ================================================================

observations_path = OUT / "observations.csv"

with observations_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "source_dataset",
            "sample_index",
            "timestamp",
            "sensor_id",
            "metric",
            "value",
        ]
    )

    for i, timestamp in enumerate(timestamps):
        for j, sensor in enumerate(sensor_ids):
            writer.writerow(
                [
                    "PEMS-BAY",
                    i,
                    int(timestamp),
                    sensor,
                    "speed",
                    float(values[i, j]),
                ]
            )


# ================================================================
# SPATIAL METADATA
# ================================================================

with h5py.File(
    RAW / "pems-bay-meta.h5",
    "r",
) as f:

    # The metadata HDF5 has the columns split across blocks.
    axis0 = [
        decode(x)
        for x in f["meta/axis0"][:]
    ]

    meta_ids = [
        decode(x)
        for x in f["meta/axis1"][:]
    ]

    block0_items = [
        decode(x)
        for x in f["meta/block0_items"][:]
    ]

    block1_items = [
        decode(x)
        for x in f["meta/block1_items"][:]
    ]

    block0 = f["meta/block0_values"][:]
    block1 = f["meta/block1_values"][:]

    # block2 is stored as an object representation in this file.
    # We do not need it for the spatial artifact.
    #
    # Spatially relevant fields are available in block0:
    # City, Abs_PM, Latitude, Longitude, Length
    #
    # Road attributes available in block1:
    # Fwy, District, County, Lanes, User_ID_4

    block0_index = {
        name: i
        for i, name in enumerate(block0_items)
    }

    block1_index = {
        name: i
        for i, name in enumerate(block1_items)
    }

    metadata = {}

    for i, sensor in enumerate(meta_ids):
        metadata[sensor] = {
            "City": float(
                block0[i, block0_index["City"]]
            ),
            "Abs_PM": float(
                block0[i, block0_index["Abs_PM"]]
            ),
            "Latitude": float(
                block0[i, block0_index["Latitude"]]
            ),
            "Longitude": float(
                block0[i, block0_index["Longitude"]]
            ),
            "Length": float(
                block0[i, block0_index["Length"]]
            ),
            "Fwy": int(
                block1[i, block1_index["Fwy"]]
            ),
            "District": int(
                block1[i, block1_index["District"]]
            ),
            "County": int(
                block1[i, block1_index["County"]]
            ),
            "Lanes": int(
                block1[i, block1_index["Lanes"]]
            ),
            "User_ID_4": int(
                block1[i, block1_index["User_ID_4"]]
            ),
        }


# ================================================================
# SENSOR JOIN
# ================================================================

traffic_set = set(sensor_ids)
metadata_set = set(meta_ids)

traffic_only = sorted(
    traffic_set - metadata_set
)

metadata_only = sorted(
    metadata_set - traffic_set
)


# ================================================================
# SENSOR LOCATIONS
# ================================================================

locations_path = SPATIAL / "sensor_locations.csv"

with locations_path.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fields = [
        "sensor_id",
        "latitude",
        "longitude",
        "city_raw",
        "abs_pm",
        "length",
        "fwy",
        "district",
        "county",
        "lanes",
        "user_id_4",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    for sensor in sensor_ids:
        row = metadata[sensor]

        writer.writerow(
            {
                "sensor_id": sensor,
                "latitude": row["Latitude"],
                "longitude": row["Longitude"],
                "city_raw": row["City"],
                "abs_pm": row["Abs_PM"],
                "length": row["Length"],
                "fwy": row["Fwy"],
                "district": row["District"],
                "county": row["County"],
                "lanes": row["Lanes"],
                "user_id_4": row["User_ID_4"],
            }
        )


# ================================================================
# PROFILE
# ================================================================

missing = int(np.isnan(values).sum())
zeros = int((values == 0).sum())

profile = {
    "schema": "pems-bay-empirical-1.0",
    "dataset": "PEMS-BAY",
    "traffic": {
        "sensor_count": len(sensor_ids),
        "sample_count": len(timestamps),
        "observation_count": int(values.size),
        "missing_count": missing,
        "zero_count": zeros,
        "min_speed": float(np.nanmin(values)),
        "max_speed": float(np.nanmax(values)),
        "mean_speed": float(np.nanmean(values)),
        "median_speed": float(np.nanmedian(values)),
    },
    "timestamps": {
        "first": int(timestamps[0]),
        "last": int(timestamps[-1]),
        "nominal_interval_seconds": 300,
        "unique_intervals": [
            {
                "seconds": float(
                    delta / 1_000_000_000
                ),
                "occurrences": int(count),
            }
            for delta, count in zip(
                unique_diffs,
                diff_counts,
            )
        ],
        "non_nominal_intervals": timestamp_gaps,
    },
    "spatial": {
        "traffic_sensors": len(traffic_set),
        "metadata_sensors": len(metadata_set),
        "traffic_only": traffic_only,
        "metadata_only": metadata_only,
        "location_count": len(sensor_ids),
    },
    "provenance": {
        "traffic_source": str(traffic_path),
        "traffic_md5": md5(traffic_path),
        "metadata_source": str(
            RAW / "pems-bay-meta.h5"
        ),
        "metadata_md5": md5(
            RAW / "pems-bay-meta.h5"
        ),
        "imputation": False,
        "smoothing": False,
        "source_values_modified": False,
        "timestamp_gaps_preserved": True,
    },
}


with (OUT / "profile.json").open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(profile, f, indent=2)


with (OUT / "manifest.json").open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "schema": "canonical-traffic-1.0",
            "dataset_id": "PEMS-BAY",
            "metric": ["speed"],
            "sensor_count": len(sensor_ids),
            "sample_count": len(timestamps),
            "files": [
                "observations.csv",
                "profile.json",
                "spatial/sensor_locations.csv",
            ],
            "source_format": "dcrnn_hdf5",
            "timestamp_source": "source_axis1",
            "raw_observations_preserved": True,
        },
        f,
        indent=2,
    )


# ================================================================
# PRINT
# ================================================================

print("=" * 80)
print("PEMS-BAY PREPARATION")
print("=" * 80)

print("\nTRAFFIC")
print("sensors:", len(sensor_ids))
print("samples:", len(timestamps))
print("observations:", values.size)
print("missing:", missing)
print("zeros:", zeros)
print("min speed:", np.nanmin(values))
print("max speed:", np.nanmax(values))
print("mean speed:", np.nanmean(values))
print("median speed:", np.nanmedian(values))

print("\nTIMESTAMPS")
print("first:", timestamps[0])
print("last:", timestamps[-1])
print("intervals:")

for delta, count in zip(
    unique_diffs,
    diff_counts,
):
    print(
        " ",
        delta / 1_000_000_000,
        "seconds:",
        int(count),
    )

print("\nNON-NOMINAL GAPS")

if timestamp_gaps:
    for gap in timestamp_gaps:
        print(
            " ",
            gap["delta_seconds"],
            "seconds:",
            gap["occurrences"],
        )
else:
    print(" none")

print("\nSPATIAL JOIN")
print("traffic sensors:", len(traffic_set))
print("metadata sensors:", len(metadata_set))
print("traffic-only:", traffic_only)
print("metadata-only:", metadata_only)

print("\nOUTPUT")
print(observations_path)
print(locations_path)
print(OUT / "profile.json")
print(OUT / "manifest.json")

print("=" * 80)
