from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


# ================================================================
# CONFIGURATION
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit empirical traffic disturbance severity."
    )

    parser.add_argument(
        "--dataset",
        default="METR-LA",
        choices=("METR-LA", "PEMS-BAY"),
        help="Dataset to audit.",
    )

    return parser.parse_args()


args = parse_args()

DATASET_ROOTS = {
    "METR-LA": Path("data/canonical/METR-LA"),
    "PEMS-BAY": Path("data/canonical/PEMS-BAY"),
}

DATASET = args.dataset
ROOT = DATASET_ROOTS[DATASET]

OBS = ROOT / "observations.csv"
EVENTS = ROOT / "disturbances" / "disturbance_events.csv"
BASELINE = ROOT / "baseline" / "sensor_context_baseline.csv"
OUT = ROOT / "event_audit"

OUT.mkdir(parents=True, exist_ok=True)


# ================================================================
# HELPERS
# ================================================================

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    k = (len(values) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)

    if lo == hi:
        return values[int(k)]

    return values[lo] + (values[hi] - values[lo]) * (k - lo)


# ================================================================
# LOAD OBSERVATIONS
# ================================================================

series = defaultdict(dict)

for row in read_csv(OBS):
    if row["metric"] != "speed":
        continue

    sensor = row["sensor_id"]
    index = int(row["sample_index"])
    value = float(row["value"])

    series[sensor][index] = value


# ================================================================
# LOAD SENSOR EMPIRICAL DISTRIBUTIONS
# ================================================================

sensor_values = defaultdict(list)

for sensor, values in series.items():
    for value in values.values():
        if value > 0:
            sensor_values[sensor].append(value)


sensor_sorted = {
    sensor: sorted(values)
    for sensor, values in sensor_values.items()
}


def empirical_percentile(sensor, value):
    values = sensor_sorted.get(sensor, [])

    if not values:
        return 0.0

    count = 0

    for x in values:
        if x <= value:
            count += 1
        else:
            break

    return count / len(values)


# ================================================================
# LOAD BASELINE
# ================================================================

baseline = {}

for row in read_csv(BASELINE):
    baseline[
        (
            row["sensor_id"],
            int(row["weekday"]),
            int(row["hour"]),
        )
    ] = float(row["mean"])


# ================================================================
# LOAD EVENTS
# ================================================================

events = []

for row in read_csv(EVENTS):
    sensors = [
        s
        for s in row["affected_sensors"].split(";")
        if s
    ]

    events.append(
        {
            "event_id": row["event_id"],
            "start": int(row["start_index"]),
            "peak": int(row["peak_index"]),
            "end": int(row["end_index"]),
            "duration": int(row["duration_steps"]),
            "sensors": sensors,
            "recovered": row["recovered"].lower() == "true",
        }
    )


# ================================================================
# TIMESTAMP / CONTEXT
# ================================================================

timestamps = {}

for row in read_csv(OBS):
    if row["metric"] == "speed":
        timestamps[int(row["sample_index"])] = row["timestamp"]


def context_for(index):
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(
        int(timestamps[index]) / 1_000_000_000,
        tz=timezone.utc,
    )

    return dt.weekday(), dt.hour


# ================================================================
# EVENT SEVERITY
# ================================================================

rows = []

for event in events:

    for sensor in event["sensors"]:

        values = series.get(sensor, {})

        if event["start"] not in values:
            continue

        if event["peak"] not in values:
            continue

        start_speed = values[event["start"]]
        peak_speed = values[event["peak"]]

        event_values = [
            values[i]
            for i in range(
                event["start"],
                event["end"] + 1,
            )
            if i in values
        ]

        positive_event_values = [
            x for x in event_values
            if x > 0
        ]

        if not positive_event_values:
            min_speed = min(event_values)
        else:
            min_speed = min(
                positive_event_values
            )

        weekday, hour = context_for(event["peak"])

        contextual_baseline = baseline.get(
            (
                sensor,
                weekday,
                hour,
            ),
            0.0,
        )

        absolute_drop = (
            contextual_baseline - peak_speed
        )

        relative_drop = (
            absolute_drop / contextual_baseline
            if contextual_baseline > 0
            else 0.0
        )

        empirical_pct = empirical_percentile(
            sensor,
            peak_speed,
        )

        low_speed_fraction = (
            sum(x < 10 for x in event_values)
            / len(event_values)
            if event_values
            else 0.0
        )

        zero_fraction = (
            sum(x == 0 for x in event_values)
            / len(event_values)
            if event_values
            else 0.0
        )

        rows.append(
            {
                "event_id": event["event_id"],
                "sensor_id": sensor,
                "start_speed": start_speed,
                "peak_speed": peak_speed,
                "minimum_event_speed": min_speed,
                "contextual_baseline": contextual_baseline,
                "absolute_drop": absolute_drop,
                "relative_drop": relative_drop,
                "empirical_speed_percentile": empirical_pct,
                "event_duration_steps": event["duration"],
                "low_speed_fraction": low_speed_fraction,
                "zero_speed_fraction": zero_fraction,
                "recovered": event["recovered"],
            }
        )


# ================================================================
# WRITE SENSOR-EVENT TABLE
# ================================================================

detail_path = OUT / "event_severity_by_sensor.csv"

if rows:
    with detail_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = list(rows[0].keys())

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# ================================================================
# EVENT-LEVEL AGGREGATION
# ================================================================

by_event = defaultdict(list)

for row in rows:
    by_event[row["event_id"]].append(row)


event_rows = []

for event_id, values in by_event.items():

    event_rows.append(
        {
            "event_id": event_id,
            "sensor_count": len(values),
            "median_peak_speed": percentile(
                [x["peak_speed"] for x in values],
                0.50,
            ),
            "median_baseline": percentile(
                [x["contextual_baseline"] for x in values],
                0.50,
            ),
            "median_absolute_drop": percentile(
                [x["absolute_drop"] for x in values],
                0.50,
            ),
            "median_relative_drop": percentile(
                [x["relative_drop"] for x in values],
                0.50,
            ),
            "median_empirical_percentile": percentile(
                [
                    x["empirical_speed_percentile"]
                    for x in values
                ],
                0.50,
            ),
            "median_low_speed_fraction": percentile(
                [
                    x["low_speed_fraction"]
                    for x in values
                ],
                0.50,
            ),
            "median_zero_fraction": percentile(
                [
                    x["zero_speed_fraction"]
                    for x in values
                ],
                0.50,
            ),
        }
    )


event_path = OUT / "event_severity_summary.csv"

if event_rows:
    with event_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = list(event_rows[0].keys())

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(event_rows)


# ================================================================
# GLOBAL AUDIT
# ================================================================

peak = [
    x["peak_speed"]
    for x in rows
]

baseline_values = [
    x["contextual_baseline"]
    for x in rows
]

absolute = [
    x["absolute_drop"]
    for x in rows
]

relative = [
    x["relative_drop"]
    for x in rows
]

percentiles = [
    x["empirical_speed_percentile"]
    for x in rows
]

low_fraction = [
    x["low_speed_fraction"]
    for x in rows
]

zero_fraction = [
    x["zero_speed_fraction"]
    for x in rows
]


summary = {
    "dataset": DATASET,
    "event_count": len(events),
    "event_sensor_records": len(rows),
    "severity": {
        "peak_speed": {
            "median": percentile(peak, 0.50),
            "p05": percentile(peak, 0.05),
            "p25": percentile(peak, 0.25),
            "p75": percentile(peak, 0.75),
            "p95": percentile(peak, 0.95),
        },
        "contextual_baseline": {
            "median": percentile(
                baseline_values,
                0.50,
            ),
        },
        "absolute_drop": {
            "median": percentile(absolute, 0.50),
            "p90": percentile(absolute, 0.90),
            "p95": percentile(absolute, 0.95),
        },
        "relative_drop": {
            "median": percentile(relative, 0.50),
            "p90": percentile(relative, 0.90),
            "p95": percentile(relative, 0.95),
        },
        "empirical_speed_percentile": {
            "median": percentile(
                percentiles,
                0.50,
            ),
            "p10": percentile(
                percentiles,
                0.10,
            ),
            "p25": percentile(
                percentiles,
                0.25,
            ),
            "p50": percentile(
                percentiles,
                0.50,
            ),
        },
        "low_speed_fraction": {
            "median": percentile(
                low_fraction,
                0.50,
            ),
        },
        "zero_speed_fraction": {
            "median": percentile(
                zero_fraction,
                0.50,
            ),
        },
    },
    "semantics": {
        "incident_ground_truth": False,
        "causal_labels": False,
        "empirical_observation_audit": True,
        "source_observations_modified": False,
    },
}


with (OUT / "severity_summary.json").open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2)


# ================================================================
# PRINT
# ================================================================

print("=" * 80)
print(f"{DATASET} EVENT SEVERITY SANITY AUDIT")
print("=" * 80)

print("\nEVENT-SENSOR RECORDS:", len(rows))

print("\nOBSERVED PEAK SPEED")
print(
    "median:",
    round(percentile(peak, 0.50), 2),
)
print(
    "p05:",
    round(percentile(peak, 0.05), 2),
)
print(
    "p25:",
    round(percentile(peak, 0.25), 2),
)
print(
    "p75:",
    round(percentile(peak, 0.75), 2),
)
print(
    "p95:",
    round(percentile(peak, 0.95), 2),
)

print("\nCONTEXTUAL BASELINE")
print(
    "median:",
    round(
        percentile(
            baseline_values,
            0.50,
        ),
        2,
    ),
)

print("\nABSOLUTE DROP")
print(
    "median:",
    round(
        percentile(absolute, 0.50),
        2,
    ),
)
print(
    "p90:",
    round(
        percentile(absolute, 0.90),
        2,
    ),
)
print(
    "p95:",
    round(
        percentile(absolute, 0.95),
        2,
    ),
)

print("\nRELATIVE DROP")
print(
    "median:",
    round(
        percentile(relative, 0.50),
        4,
    ),
)
print(
    "p90:",
    round(
        percentile(relative, 0.90),
        4,
    ),
)
print(
    "p95:",
    round(
        percentile(relative, 0.95),
        4,
    ),
)

print("\nEMPIRICAL SPEED PERCENTILE")
print(
    "median:",
    round(
        percentile(percentiles, 0.50),
        4,
    ),
)
print(
    "p10:",
    round(
        percentile(percentiles, 0.10),
        4,
    ),
)
print(
    "p25:",
    round(
        percentile(percentiles, 0.25),
        4,
    ),
)
print(
    "p50:",
    round(
        percentile(percentiles, 0.50),
        4,
    ),
)

print("\nLOW-SPEED AUDIT")
print(
    "median event low-speed fraction:",
    round(
        percentile(low_fraction, 0.50),
        4,
    ),
)
print(
    "median event zero-speed fraction:",
    round(
        percentile(zero_fraction, 0.50),
        4,
    ),
)

print("\nSEMANTICS")
print("incident ground truth: False")
print("causal labels: False")
print("empirical observation audit: True")
print("source observations modified: False")

print("=" * 80)