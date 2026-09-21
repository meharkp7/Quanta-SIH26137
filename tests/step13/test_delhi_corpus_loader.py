"""Checks for the real Delhi-1615 manifest shape using a tiny local corpus."""
from __future__ import annotations

import json

import pytest

from src.data.causal_episodes import generate_causal_pilot
from src.learning.loader import load_corpus_manifest, load_pilot_windows
from src.learning.train_forecaster import train_forecaster
from src.platform.catalog import load_scenario


def test_loader_reads_current_delhi_manifest_shape(tmp_path):
    scenario = load_scenario("S3_BASE")
    root = tmp_path / "delhi_like"
    generate_causal_pilot(
        scenario,
        root,
        duration_s=2100,
        warmup_s=180,
        base_seed=21,
    )
    legacy = json.loads((root / "corpus_manifest.json").read_text())
    episode_id = legacy["train"][0]
    scenario_id = scenario.scenario_id
    scenario_path = "maps/map_000/scenario.json"
    scenario_file = root / scenario_path
    scenario_file.parent.mkdir(parents=True)
    scenario_file.write_text(scenario.model_dump_json(), encoding="utf-8")

    # Keep only a train episode: this deliberately models a small local smoke
    # set. It must not be mistaken for a complete train/validation/test corpus.
    delhi_manifest = {
        "schema_version": "quanta-real-sumo-osm-corpus-v1",
        "split_episode_counts": {"train": 1, "validation": 0, "test": 0},
        "map_records": {
            "0": {
                "split": "train",
                "zone": "sample",
                "scenario_id": scenario_id,
                "scenario_path": scenario_path,
                "map_fingerprint": "sample-map-fingerprint",
            }
        },
        "episodes": [{"episode_id": episode_id, "split": "train", "map_index": 0}],
    }
    (root / "corpus_manifest_1615.json").write_text(
        json.dumps(delhi_manifest), encoding="utf-8"
    )

    manifest, path = load_corpus_manifest(root)
    datasets = load_pilot_windows(root)

    assert path.name == "corpus_manifest_1615.json"
    assert manifest["episodes"][0]["episode_id"] == episode_id
    assert set(datasets) == {"train"}
    assert len(datasets["train"]) > 0

    with pytest.raises(ValueError, match="separate validation split"):
        train_forecaster(root, tmp_path / "artifact", epochs=1)
