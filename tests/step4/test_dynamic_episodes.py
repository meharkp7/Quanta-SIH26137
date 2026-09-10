import csv
import json

from src.data.dataset_generator import DynamicDatasetConfig, generate_dynamic_scenario
from src.data.dynamic_episodes import DynamicEpisodeConfig, generate_episode


def _scenario(seed=26137):
    return generate_dynamic_scenario(
        DynamicDatasetConfig(customer_count=24, road_junction_count=100, seed=seed, topology_family="hybrid")
    )


def test_episode_is_reproducible_and_causal(tmp_path):
    scenario = _scenario()
    cfg = DynamicEpisodeConfig(
        duration_s=600, interval_s=60, warmup_s=120,
        regime="morning_peak", event_type="incident", seed=77,
    )
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_episode(scenario, a, cfg, episode_id="ep-a")
    generate_episode(scenario, b, cfg, episode_id="ep-a")
    assert (a / "edge_truth.csv").read_bytes() == (b / "edge_truth.csv").read_bytes()
    assert (a / "observations.csv").read_bytes() == (b / "observations.csv").read_bytes()

    events = json.loads((a / "events.json").read_text())
    observations = list(csv.DictReader((a / "observations.csv").open()))
    for event in events:
        affected = event["affected_parent_road_ids"]
        for row in observations:
            if row["edge_id"] and row["observation_time_s"]:
                # The observation file intentionally contains no event ID or
                # future effect time. This is a structural leakage check.
                assert "effect_start_s" not in row
                assert "active_event_id" not in row
        # Event reveal metadata remains environment-side only.
        assert event["effect_start_s"] >= event["reveal_time_s"]


def test_closure_changes_truth_but_not_unrevealed_observation(tmp_path):
    scenario = _scenario(27137)
    cfg = DynamicEpisodeConfig(
        duration_s=900, interval_s=60, warmup_s=120,
        regime="normal", event_type="closure", seed=88, reveal_lead_s=120,
    )
    out = tmp_path / "episode"
    generate_episode(scenario, out, cfg, episode_id="closure")
    truth = list(csv.DictReader((out / "edge_truth.csv").open()))
    observations = list(csv.DictReader((out / "observations.csv").open()))
    events = json.loads((out / "events.json").read_text())
    event = events[0]
    affected = set(event["affected_parent_road_ids"])
    closed_truth = [r for r in truth if r["parent_road_id"] in affected and r["is_closed"] == "True"]
    assert closed_truth
    # The observation schema contains only current measured fields and a
    # boolean known_closed flag; it never contains future effect times.
    pre_reveal = [r for r in observations if int(r["observation_time_s"]) < event["reveal_time_s"]]
    assert pre_reveal
    assert all("effect_start_s" not in r for r in pre_reveal)


def test_all_topology_families_generate(tmp_path):
    for family in ("grid", "irregular", "radial", "hybrid"):
        scenario = generate_dynamic_scenario(
            DynamicDatasetConfig(
                customer_count=24, road_junction_count=100,
                topology_family=family, seed=9000 + len(family),
            )
        )
        assert scenario.nodes
        assert scenario.edges
        assert scenario.requests
