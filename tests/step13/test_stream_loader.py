"""Parity: streamed episode batches match preloaded WindowDataset batches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.data.causal_episodes import generate_causal_pilot
from src.learning.loader import EpisodeStream, WindowDataset, episode_scenario_map
from src.learning.scaling import FeatureScaler
from src.learning.schema import FEATURE_NAMES
from src.learning.train_forecaster import _evaluate, _stream_validation_mae
from src.learning.gnn_transformer import CausalGNNTransformer
from src.platform.catalog import load_scenario


def _tiny_pilot(tmp_path: Path):
    scenario = load_scenario("S3_BASE")
    root = tmp_path / "pilot"
    generate_causal_pilot(scenario, root, duration_s=2100, warmup_s=180, base_seed=21)
    manifest = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
    split_ids = {k: manifest[k] for k in ("train", "validation", "test")}
    ep_map = episode_scenario_map(root, manifest, split_ids)
    return scenario, root, split_ids, ep_map


def test_stream_counts_and_batches_match_preloaded(tmp_path: Path) -> None:
    scenario, root, split_ids, ep_map = _tiny_pilot(tmp_path)
    scenarios = {scenario.scenario_id: scenario}
    datasets = {}
    for name in ("train", "validation", "test"):
        from src.learning.windows import build_episode_windows

        windows = []
        for eid in split_ids[name]:
            folder = root / "episodes" / eid
            ep_manifest = json.loads((folder / "episode_manifest.json").read_text(encoding="utf-8"))
            windows.extend(build_episode_windows(folder, scenario, split=ep_manifest["split"]))
        datasets[name] = WindowDataset(windows)
    scaler = FeatureScaler.fit_windows(datasets["train"].windows, FEATURE_NAMES)

    stream = EpisodeStream(
        root, "train", split_ids["train"], scenarios, ep_map, scaler=scaler
    )
    assert len(stream) == len(datasets["train"])

    seen = set()
    for batch in stream.iter_batches(2, shuffle=False):
        assert batch.features.shape[0] <= 2
        for eid, issue in zip(batch.episode_ids, batch.issue_times_s):
            seen.add((eid, int(issue)))
    expected = {(w.episode_id, w.issue_time_s) for w in datasets["train"].windows}
    assert seen == expected

    # Same scaler bytes in both paths: compare one identical batch.
    first_episode = split_ids["train"][0]
    ep_windows = [w for w in datasets["train"].windows if w.episode_id == first_episode][:2]
    reference = WindowDataset(ep_windows, scaler=scaler).batch()
    streamed = next(
        b for b in EpisodeStream(
            root, "train", [first_episode], scenarios, ep_map, scaler=scaler
        ).iter_batches(2, shuffle=False)
    )
    assert np.allclose(streamed.features, reference.features)


def test_stream_validation_mae_matches_evaluate(tmp_path: Path) -> None:
    scenario, root, split_ids, ep_map = _tiny_pilot(tmp_path)
    scenarios = {scenario.scenario_id: scenario}
    from src.learning.windows import build_episode_windows

    val_windows = []
    for eid in split_ids["validation"]:
        folder = root / "episodes" / eid
        ep_manifest = json.loads((folder / "episode_manifest.json").read_text(encoding="utf-8"))
        val_windows.extend(build_episode_windows(folder, scenario, split=ep_manifest["split"]))
    train_windows = []
    for eid in split_ids["train"][:1]:
        folder = root / "episodes" / eid
        ep_manifest = json.loads((folder / "episode_manifest.json").read_text(encoding="utf-8"))
        train_windows.extend(build_episode_windows(folder, scenario, split=ep_manifest["split"]))
    scaler = FeatureScaler.fit_windows(train_windows, FEATURE_NAMES)
    val_ds = WindowDataset(val_windows, scaler=scaler)
    stream = EpisodeStream(
        root, "validation", split_ids["validation"], scenarios, ep_map, scaler=scaler
    )

    torch.manual_seed(0)
    model = CausalGNNTransformer(width=8, heads=2, layers=1)
    device = torch.device("cpu")
    expected, _ = _evaluate(model, val_ds, scenarios, device, batch_size=2)
    got = _stream_validation_mae(model, stream, scenarios, device, 2, total_edges=100)
    for horizon in ("5", "10", "15"):
        # NOTE: exact equality cannot hold across different batch groupings:
        # the temporal Transformer has no padding mask, so real-edge outputs
        # shift slightly with the number of padded roads in the batch (also
        # true of _iter_batches across batch sizes/seeds). Same math, close.
        assert got[horizon]["mae"] is not None
        assert abs(got[horizon]["mae"] - expected[horizon]["mae"]) < 0.15

    # Identical grouping -> bit-exact parity of the accumulator math.
    single = EpisodeStream(
        root, "validation", split_ids["validation"][:1], scenarios, ep_map,
        scaler=scaler,
    )
    single_windows = [w for w in val_windows if w.episode_id == split_ids["validation"][0]]
    single_ds = WindowDataset(single_windows, scaler=scaler)
    expected_single, _ = _evaluate(model, single_ds, scenarios, device, batch_size=2)
    got_single = _stream_validation_mae(model, single, scenarios, device, 2, total_edges=100)
    for horizon in ("5", "10", "15"):
        assert got_single[horizon]["mae"] == expected_single[horizon]["mae"]
