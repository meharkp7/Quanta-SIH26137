import csv
import json

from src.data.metrla_baseline import build_temporal_baseline


def _write_fixture(path):
    rows = []

    timestamps = [
        "1330560000000000000",  # 2012-03-01 00:00 UTC
        "1330560300000000000",
        "1330560600000000000",
        "1330560900000000000",
    ]

    values = [60.0, 62.0, 0.0, 64.0]

    for sensor_id in ["A", "B"]:
        for timestamp, value in zip(timestamps, values):
            rows.append(
                {
                    "source_dataset": "TEST",
                    "sample_index": str(len(rows)),
                    "timestamp": timestamp,
                    "sensor_id": sensor_id,
                    "metric": "speed",
                    "value": str(value),
                }
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_dataset",
                "sample_index",
                "timestamp",
                "sensor_id",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_temporal_baseline_excludes_zero_from_statistics(tmp_path):
    observations = tmp_path / "observations.csv"
    output = tmp_path / "baseline"

    _write_fixture(observations)

    profile = build_temporal_baseline(
        observations,
        output,
    )

    assert profile.sensor_count == 2
    assert profile.sample_count == 4
    assert profile.observed_count == 6

    with (output / "sensor_context_baseline.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows

    # Every context has only positive observations.
    assert all(float(row["mean"]) > 0 for row in rows)


def test_temporal_baseline_writes_manifest(tmp_path):
    observations = tmp_path / "observations.csv"
    output = tmp_path / "baseline"

    _write_fixture(observations)

    build_temporal_baseline(
        observations,
        output,
    )

    with (output / "baseline_profile.json").open(
        encoding="utf-8",
    ) as handle:
        profile = json.load(handle)

    assert profile["schema"] == "metrla-temporal-baseline-1.0"
    assert profile["semantics"]["source_values_unchanged"] is True
    assert profile["semantics"]["zero_values_preserved"] is True
    assert profile["semantics"]["zero_values_used_in_baseline"] is False

    assert (output / "manifest.json").exists()
    assert (output / "sensor_hour_baseline.csv").exists()
    assert (output / "sensor_weekday_baseline.csv").exists()
