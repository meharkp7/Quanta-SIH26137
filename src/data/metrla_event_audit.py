from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit empirical traffic disturbance events."
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
LOCATIONS = ROOT / "spatial" / "sensor_locations.csv"
BASELINE = ROOT / "baseline" / "sensor_context_baseline.csv"
OUT = ROOT / "event_audit"

OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0

    k = (len(values) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))

    if lo == hi:
        return values[lo]

    return values[lo] + (values[hi] - values[lo]) * (k - lo)


# ================================================================
# 1. LOAD SPATIAL DATA
# ================================================================

locations = {}

for row in read_csv(LOCATIONS):
    locations[row["sensor_id"]] = (
        float(row["latitude"]),
        float(row["longitude"]),
    )

# ================================================================
# 2. LOAD EVENTS
# ================================================================

events = []

for row in read_csv(EVENTS):
    sensors = {
        s for s in row["affected_sensors"].split(";") if s
    }

    events.append(
        {
            "event_id": row["event_id"],
            "start": int(row["start_index"]),
            "peak": int(row["peak_index"]),
            "end": int(row["end_index"]),
            "duration": int(row["duration_steps"]),
            "sensors": sensors,
            "sensor_count": int(row["spatial_sensor_count"]),
            "fraction": float(row["affected_fraction"]),
            "relative_drop": float(
                row["peak_relative_deviation"]
            ),
            "absolute_drop": float(
                row["peak_absolute_deviation"]
            ),
            "recovered": row["recovered"].lower() == "true",
        }
    )


# ================================================================
# 3. SPATIAL COHERENCE
# ================================================================

spatial_rows = []

for event in events:
    coords = [
        locations[s]
        for s in event["sensors"]
        if s in locations
    ]

    if not coords:
        continue

    centroid_lat = sum(x[0] for x in coords) / len(coords)
    centroid_lon = sum(x[1] for x in coords) / len(coords)

    radii = [
        haversine_km(
            centroid_lat,
            centroid_lon,
            lat,
            lon,
        )
        for lat, lon in coords
    ]

    radius = max(radii)

    # Pairwise diameter. Only <=207 sensors, so this is cheap.
    diameter = 0.0

    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = haversine_km(
                coords[i][0],
                coords[i][1],
                coords[j][0],
                coords[j][1],
            )
            diameter = max(diameter, d)

    spatial_rows.append(
        {
            "event_id": event["event_id"],
            "sensor_count": event["sensor_count"],
            "affected_fraction": event["fraction"],
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "radius_km": radius,
            "diameter_km": diameter,
            "relative_drop": event["relative_drop"],
            "absolute_drop": event["absolute_drop"],
            "duration_steps": event["duration"],
            "recovered": event["recovered"],
        }
    )


with (OUT / "spatial_event_statistics.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(spatial_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(spatial_rows)


# ================================================================
# 4. TEMPORAL + SENSOR OVERLAP
# ================================================================

overlap_rows = []

for i in range(len(events)):
    a = events[i]

    for j in range(i + 1, len(events)):
        b = events[j]

        temporal = (
            a["start"] <= b["end"]
            and b["start"] <= a["end"]
        )

        if not temporal:
            continue

        shared = a["sensors"] & b["sensors"]

        overlap_rows.append(
            {
                "event_a": a["event_id"],
                "event_b": b["event_id"],
                "temporal_overlap": True,
                "shared_sensor_count": len(shared),
                "shared_sensors": ";".join(sorted(shared)),
                "gap_steps": max(
                    a["start"] - b["end"],
                    b["start"] - a["end"],
                    0,
                ),
            }
        )


with (OUT / "event_overlaps.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    fields = [
        "event_a",
        "event_b",
        "temporal_overlap",
        "shared_sensor_count",
        "shared_sensors",
        "gap_steps",
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(overlap_rows)


# ================================================================
# 5. SEVERITY / LOW-SPEED AUDIT
# ================================================================

observations = read_csv(OBS)

# Only observations belonging to sensors participating in events.
event_sensors = set()

for e in events:
    event_sensors.update(e["sensors"])

event_values = []

for row in observations:
    if row["sensor_id"] not in event_sensors:
        continue

    value = float(row["value"])

    if value >= 0:
        event_values.append(value)

# Event-level severity.
relative = [e["relative_drop"] for e in events]
absolute = [e["absolute_drop"] for e in events]
duration = [e["duration"] for e in events]
sensor_counts = [e["sensor_count"] for e in events]
radii = [x["radius_km"] for x in spatial_rows]
diameters = [x["diameter_km"] for x in spatial_rows]

zero_event_sensor_values = sum(
    1 for x in event_values if x == 0
)

low_speed_values = sum(
    1 for x in event_values if x < 10
)


# ================================================================
# 6. AUTOMATIC EVENT REGIME CLASSIFICATION
# ================================================================

classified = []

for row in spatial_rows:
    sensors = row["sensor_count"]
    fraction = row["affected_fraction"]
    diameter = row["diameter_km"]

    # This is deliberately descriptive rather than a claim
    # about real-world incidents.
    if fraction <= 0.10 and diameter <= 20:
        regime = "localized"
    elif fraction <= 0.25 and diameter <= 40:
        regime = "regional"
    else:
        regime = "network_regime_like"

    row = dict(row)
    row["spatial_regime"] = regime
    classified.append(row)


with (OUT / "classified_events.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(classified[0].keys()),
    )
    writer.writeheader()
    writer.writerows(classified)


# ================================================================
# 7. SUMMARY
# ================================================================

regime_counts = Counter(
    row["spatial_regime"]
    for row in classified
)

summary = {
    "dataset": DATASET,
    "event_count": len(events),
    "recovered_events": sum(
        e["recovered"] for e in events
    ),
    "recovery_fraction": (
        sum(e["recovered"] for e in events)
        / len(events)
        if events
        else 0.0
    ),
    "spatial_regimes": dict(regime_counts),
    "severity": {
        "relative_drop_mean": sum(relative) / len(relative),
        "relative_drop_median": percentile(relative, 0.50),
        "relative_drop_p90": percentile(relative, 0.90),
        "relative_drop_p95": percentile(relative, 0.95),
        "relative_drop_max": max(relative),
        "absolute_drop_mean": sum(absolute) / len(absolute),
        "absolute_drop_median": percentile(absolute, 0.50),
        "duration_median_steps": percentile(duration, 0.50),
        "duration_p95_steps": percentile(duration, 0.95),
    },
    "spatial": {
        "radius_median_km": percentile(radii, 0.50),
        "radius_p95_km": percentile(radii, 0.95),
        "diameter_median_km": percentile(diameters, 0.50),
        "diameter_p95_km": percentile(diameters, 0.95),
        "diameter_max_km": max(diameters),
    },
    "overlap": {
        "temporal_overlap_pairs": len(overlap_rows),
        "sensor_overlap_pairs": sum(
            x["shared_sensor_count"] > 0
            for x in overlap_rows
        ),
    },
    "low_speed_audit": {
        "event_sensor_observations": len(event_values),
        "zero_speed_fraction": (
            zero_event_sensor_values / len(event_values)
            if event_values else 0.0
        ),
        "below_10_speed_fraction": (
            low_speed_values / len(event_values)
            if event_values else 0.0
        ),
    },
    "semantics": {
        "incident_ground_truth": False,
        "causal_labels": False,
        "empirical_candidates_only": True,
        "source_observations_modified": False,
    },
}

with (OUT / "summary.json").open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2)


# ================================================================
# 8. PRINT AUDIT
# ================================================================

print("=" * 80)
print(f"{DATASET} COMBINED EVENT AUDIT")
print("=" * 80)

print("\nEVENTS")
print("total:", len(events))
print(
    "recovered:",
    sum(e["recovered"] for e in events),
)

print("\nSPATIAL REGIMES")

for regime in (
    "localized",
    "regional",
    "network_regime_like",
):
    count = regime_counts.get(regime, 0)
    fraction = count / len(events) if events else 0

    print(
        f"{regime:22s}: "
        f"{count:4d} "
        f"({fraction:.1%})"
    )

print("\nSPATIAL EXTENT")

print(
    "radius median:",
    round(percentile(radii, 0.50), 2),
    "km",
)

print(
    "radius p95:",
    round(percentile(radii, 0.95), 2),
    "km",
)

print(
    "diameter median:",
    round(percentile(diameters, 0.50), 2),
    "km",
)

print(
    "diameter p95:",
    round(percentile(diameters, 0.95), 2),
    "km",
)

print(
    "diameter max:",
    round(max(diameters), 2),
    "km",
)

print("\nOVERLAP")

print(
    "temporal overlap pairs:",
    len(overlap_rows),
)

print(
    "sensor-overlap pairs:",
    sum(
        x["shared_sensor_count"] > 0
        for x in overlap_rows
    ),
)

print("\nSEVERITY")

print(
    "relative drop median:",
    round(percentile(relative, 0.50), 3),
)

print(
    "relative drop p90:",
    round(percentile(relative, 0.90), 3),
)

print(
    "relative drop p95:",
    round(percentile(relative, 0.95), 3),
)

print(
    "absolute drop median:",
    round(percentile(absolute, 0.50), 2),
)

print(
    "duration median:",
    round(percentile(duration, 0.50), 1),
    "steps",
)

print(
    "duration p95:",
    round(percentile(duration, 0.95), 1),
    "steps",
)

print("\nLOW-SPEED CHECK")

print(
    "zero-speed fraction:",
    round(
        zero_event_sensor_values / len(event_values),
        4,
    ),
)

print(
    "below-10-speed fraction:",
    round(
        low_speed_values / len(event_values),
        4,
    ),
)

print("\nOUTPUT")
print(OUT / "spatial_event_statistics.csv")
print(OUT / "event_overlaps.csv")
print(OUT / "classified_events.csv")
print(OUT / "summary.json")

print("=" * 80)
