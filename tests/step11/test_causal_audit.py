import json

from src.data.causal_audit import audit_episode
from src.data.causal_episodes import generate_causal_episode
from src.data.dynamic_episodes import DynamicEpisodeConfig
from src.platform.catalog import load_scenario


def test_causal_audit_accepts_valid_episode(tmp_path):
    out = tmp_path / "ep"
    generate_causal_episode(
        load_scenario("S3_BASE"), out,
        config=DynamicEpisodeConfig(duration_s=600, interval_s=60, warmup_s=60, seed=9),
        episode_id="ep-audit", split="train",
    )
    report = audit_episode(out)
    assert report["ok"] is True
    assert len(report["artifact_sha256"]) == 7


def test_causal_audit_rejects_leaked_observation_field(tmp_path):
    out = tmp_path / "ep"
    generate_causal_episode(
        load_scenario("S3_BASE"), out,
        config=DynamicEpisodeConfig(duration_s=600, interval_s=60, warmup_s=60, seed=9),
        episode_id="ep-audit", split="train",
    )
    path = out / "observations.csv"
    text = path.read_text()
    path.write_text(text.replace("observation_time_s", "effect_start_s,observation_time_s", 1))
    report = audit_episode(out)
    assert report["ok"] is False
    assert any("leaked" in error for error in report["errors"])
