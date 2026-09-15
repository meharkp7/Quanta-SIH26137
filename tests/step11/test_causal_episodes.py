import csv
import json

from src.data.causal_episodes import generate_causal_episode, generate_causal_pilot, stream_seeds
from src.data.causal_labels import mature_speed_proxy_label, mature_traversal_label
from src.data.dynamic_episodes import DynamicEpisodeConfig
from src.platform.catalog import load_scenario


def _tiny_config(**kwargs) -> DynamicEpisodeConfig:
    values = dict(
        duration_s=2100,
        interval_s=60,
        warmup_s=180,
        regime="morning_peak",
        event_type="closure",
        event_count=1,
        seed=77,
        backend="causal_field_v1",
    )
    values.update(kwargs)
    return DynamicEpisodeConfig(**values)


def test_streams_are_independent():
    a = stream_seeds(10)
    b = stream_seeds(10)
    c = stream_seeds(11)
    assert a == b
    assert len({a.traffic, a.incident, a.observation, a.trajectory}) == 4
    assert a.traffic != c.traffic


def test_speed_proxy_matures_only_when_observation_exists():
    observations = {("E01", 900): {"missing": 0, "observed_speed_mps": 8.5}}
    ready = mature_speed_proxy_label(
        episode_id="ep",
        edge_id="E01",
        issue_time_s=600,
        target_time_s=900,
        observation_by_bucket=observations,
    )
    missing = mature_speed_proxy_label(
        episode_id="ep",
        edge_id="E01",
        issue_time_s=600,
        target_time_s=960,
        observation_by_bucket=observations,
    )
    assert ready.missing is False
    assert ready.available_at_s == 900
    assert ready.value == 8.5
    assert missing.missing is True
    assert missing.value is None


def test_sparse_traversal_is_missing_not_zero():
    label = mature_traversal_label(
        episode_id="ep",
        edge_id="E99",
        issue_time_s=0,
        target_time_s=300,
        traversals=[],
    )
    assert label.missing is True
    assert label.value is None
    assert label.target_kind == "realized_traversal"


def test_traversal_aligns_to_entry_bucket_and_matures_at_exit():
    label = mature_traversal_label(
        episode_id="ep",
        edge_id="E01",
        issue_time_s=0,
        target_time_s=300,
        traversals=[
            {
                "edge_id": "E01",
                "entry_time_s": 310,
                "exit_time_s": 355,
            }
        ],
    )
    assert label.missing is False
    assert label.available_at_s == 355
    assert label.value == 45.0


def test_episode_is_causal_and_keeps_forecasts_separate(tmp_path):
    scenario = load_scenario("S3_BASE")
    out = tmp_path / "ep-001"
    result = generate_causal_episode(
        scenario,
        out,
        config=_tiny_config(),
        episode_id="ep-001",
        split="train",
    )
    observations = list(csv.DictReader((out / "observations.csv").open()))
    assert observations
    for row in observations:
        assert "effect_start_s" not in row
        assert "active_event_id" not in row
        assert "true_speed_mps" not in row
        assert "target_available_at_s" not in row
    events = json.loads((out / "events.json").read_text())
    assert events
    assert all(event["reveal_time_s"] <= event["effect_start_s"] for event in events)
    labels = [
        json.loads(line)
        for line in (out / "labels.jsonl").read_text().splitlines()
        if line
    ]
    assert labels
    for item in labels:
        if not item["missing"] and item["value"] is not None:
            assert item["available_at_s"] is not None
            assert item["target_time_s"] > item["issue_time_s"]
    issued = [
        json.loads(line)
        for line in (out / "issued_forecasts.jsonl").read_text().splitlines()
        if line
    ]
    assert issued
    assert "labels" not in issued[0]
    assert issued[0]["issued_at_s"] < min(issued[0]["target_times_s"])
    assert result.label_coverage["speed_proxy"]["all_have_availability_when_valid"]


def test_six_episode_pilot_splits_before_windows(tmp_path):
    scenario = load_scenario("S3_BASE")
    coverage = generate_causal_pilot(
        scenario,
        tmp_path / "pilot",
        duration_s=2100,
        warmup_s=180,
        base_seed=9,
    )
    split = json.loads((tmp_path / "pilot" / "corpus_manifest.json").read_text())
    assert set(split["train"]) == {"ep-001", "ep-002", "ep-003", "ep-004"}
    assert split["validation"] == ["ep-005"]
    assert split["test"] == ["ep-006"]
    assert coverage["episodes"] == 6
    assert coverage["split_before_windows"] is True
    assert coverage["disk_bytes"] > 0
    assert coverage["wall_clock_s"] >= 0


def test_persistence_uses_latest_valid_observation_when_issue_bucket_is_missing():
    """
    Persistence forecast must use the latest valid observation at or before
    the issue time.

    It must:
    - use the exact issue bucket when valid;
    - fall back to the latest earlier valid bucket when issue bucket is missing;
    - never use a future observation;
    - preserve missingness when no historical valid observation exists.
    """
    from src.data.causal_episodes import _issue_forecasts
    from src.data.dynamic_episodes import DynamicEpisodeConfig

    scenario = load_scenario("S3_BASE")
    edge_id = scenario.edges[0].edge_id

    rows = []

    for t in range(0, 901, 60):
        for edge in scenario.edges:
            if edge.edge_id == edge_id:
                if t == 720:
                    speed = None
                    missing = 1
                elif t == 780:
                    # This must NEVER be used for the 720s forecast.
                    speed = 99.0
                    missing = 0
                else:
                    speed = 7.0
                    missing = 0
            else:
                speed = 8.0
                missing = 0

            rows.append(
                {
                    "observation_time_s": t,
                    "edge_id": edge.edge_id,
                    "observed_speed_mps": speed,
                    "missing": missing,
                }
            )

    config = DynamicEpisodeConfig(
        duration_s=1800,
        interval_s=60,
        warmup_s=0,
        seed=1,
    )

    records = _issue_forecasts(
        scenario=scenario,
        episode_id="ep",
        config=config,
        observations=rows,
    )

    rec = next(
        r for r in records
        if r["issued_at_s"] == 720.0
    )

    idx = rec["edge_ids"].index(edge_id)

    # 720s is missing, so persistence must fall back to 660s = 7.0.
    # It must NOT use the future 780s value = 99.0.
    assert rec["prediction"][idx][0] == 7.0