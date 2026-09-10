from pathlib import Path

from src.data.metrla_analysis import (
    analyze_sensor_statistics,
    analyze_spatial_statistics,
    analyze_temporal_profile,
    detect_shock_candidates,
    write_analysis,
)


def observations():
    return [
        {
            "source_dataset": "TEST",
            "sample_index": 0,
            "timestamp": "1000000000",
            "sensor_id": "A",
            "metric": "speed",
            "value": 60.0,
        },
        {
            "source_dataset": "TEST",
            "sample_index": 1,
            "timestamp": "301000000000",
            "sensor_id": "A",
            "metric": "speed",
            "value": 40.0,
        },
        {
            "source_dataset": "TEST",
            "sample_index": 2,
            "timestamp": "601000000000",
            "sensor_id": "A",
            "metric": "speed",
            "value": 42.0,
        },
        {
            "source_dataset": "TEST",
            "sample_index": 0,
            "timestamp": "1000000000",
            "sensor_id": "B",
            "metric": "speed",
            "value": 0.0,
        },
    ]


def test_sensor_statistics():
    result = analyze_sensor_statistics(observations())

    assert len(result) == 2

    sensor_a = next(item for item in result if item.sensor_id == "A")

    assert sensor_a.observed_count == 3
    assert sensor_a.missing_count == 0
    assert sensor_a.minimum == 40.0
    assert sensor_a.maximum == 60.0


def test_temporal_profile():
    result = analyze_temporal_profile(observations())

    assert result.sensor_count == 2
    assert result.timestamp_count == 3
    assert result.observed_count == 4
    assert result.missing_count == 0


def test_shock_candidates():
    result = detect_shock_candidates(
        observations(),
        absolute_drop=10.0,
        relative_drop=0.20,
    )

    assert len(result) == 1
    assert result[0].sensor_id == "A"
    assert result[0].current_speed == 40.0
    assert result[0].direction == "drop"


def test_spatial_statistics(tmp_path: Path):
    path = tmp_path / "sensor_edges.csv"

    path.write_text(
        "from_sensor_id,to_sensor_id,distance\n"
        "A,B,10\n"
        "B,A,11\n"
        "B,C,20\n",
        encoding="utf-8",
    )

    result = analyze_spatial_statistics(path)

    assert result.sensor_count == 3
    assert result.directed_edge_count == 3
    assert result.reciprocal_pair_count == 1
    assert result.asymmetric_pair_count == 1
    assert result.minimum_distance == 10.0
    assert result.maximum_distance == 20.0


def test_write_analysis(tmp_path: Path):
    temporal = analyze_temporal_profile(observations())
    stats = analyze_sensor_statistics(observations())
    shocks = detect_shock_candidates(observations())

    spatial_path = tmp_path / "edges.csv"
    spatial_path.write_text(
        "from_sensor_id,to_sensor_id,distance\n"
        "A,B,10\n",
        encoding="utf-8",
    )

    spatial = analyze_spatial_statistics(spatial_path)

    output = write_analysis(
        tmp_path / "analysis",
        temporal,
        stats,
        shocks,
        spatial,
    )

    assert output.exists()
    assert (output / "manifest.json").exists()
    assert (output / "temporal_profile.json").exists()
    assert (output / "sensor_statistics.csv").exists()
    assert (output / "shock_candidates.csv").exists()
    assert (output / "spatial_statistics.json").exists()
