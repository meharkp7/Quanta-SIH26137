import csv
import json

from src.data.metrla_disturbance import (
    DisturbanceConfig,
    detect_disturbances,
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_spatiotemporal_disturbance_detection(tmp_path):
    observations = tmp_path / "observations.csv"
    baseline = tmp_path / "baseline.csv"
    edges = tmp_path / "edges.csv"
    output = tmp_path / "output"

    timestamps = [
        "1330560000000000000",
        "1330560300000000000",
        "1330560600000000000",
        "1330560900000000000",
        "1330561200000000000",
    ]

    rows = []

    for sensor in ["A", "B"]:
        values = [60.0, 40.0, 38.0, 60.0, 60.0]

        for index, (timestamp, value) in enumerate(
            zip(timestamps, values)
        ):
            rows.append(
                {
                    "source_dataset": "TEST",
                    "sample_index": str(index),
                    "timestamp": timestamp,
                    "sensor_id": sensor,
                    "metric": "speed",
                    "value": str(value),
                }
            )

    _write_csv(
        observations,
        [
            "source_dataset",
            "sample_index",
            "timestamp",
            "sensor_id",
            "metric",
            "value",
        ],
        rows,
    )

    # All timestamps are Thursday in the fixture.
    baseline_rows = []

    for sensor in ["A", "B"]:
        baseline_rows.append(
            {
                "sensor_id": sensor,
                "weekday": "3",
                "hour": "0",
                "count": "10",
                "mean": "60.0",
                "median": "60.0",
                "p05": "60.0",
                "p95": "60.0",
                "std": "2.0",
            }
        )

    _write_csv(
        baseline,
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
        ],
        baseline_rows,
    )

    _write_csv(
        edges,
        [
            "from_sensor_id",
            "to_sensor_id",
            "distance",
        ],
        [
            {
                "from_sensor_id": "A",
                "to_sensor_id": "B",
                "distance": "100",
            }
        ],
    )

    events = detect_disturbances(
        observations,
        baseline,
        edges,
        output,
        DisturbanceConfig(
            min_relative_deviation=0.20,
            min_absolute_deviation=8.0,
            min_persistence_steps=2,
            min_spatial_sensors=2,
        ),
    )

    assert events
    assert len(events[0].affected_sensors) == 2
    assert events[0].spatial_sensor_count == 2
    assert events[0].recovered is True

    with (output / "disturbance_profile.json").open(
        encoding="utf-8"
    ) as handle:
        profile = json.load(handle)

    assert profile["semantics"]["ground_truth"] is False
    assert profile["semantics"]["zero_values_used_as_events"] is False
