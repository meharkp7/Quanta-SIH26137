"""Regression: v1 window cache reused the first window's times for all windows.

On corpus_v2 every episode has one issue slot per 60 s with sliding history
and per-issue targets, so windows 2..N failed assert_window_causal with
"target times must be strictly after issue_time" and blocked ALL
GNN-Transformer training/sweep loads. v2 stores per-window time arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.learning.window_cache import load_episode_windows, save_episode_windows
from src.learning.windows import ForecastWindow, assert_window_causal


def _window(issue: int) -> ForecastWindow:
    history = tuple(issue - 660 + index * 60 for index in range(12))
    targets = tuple(issue + 300 + index * 300 for index in range(3))
    edge_count, horizons, features = 3, 3, 6
    available = np.empty((edge_count, horizons, 2), dtype=np.float32)
    for index, target in enumerate(targets):
        available[:, index, 0] = float(target + 10)
        available[:, index, 1] = float(target + 20)
    return ForecastWindow(
        episode_id="ep-test",
        split="train",
        issue_time_s=issue,
        history_times_s=history,
        target_times_s=targets,
        edge_ids=("a", "b", "c"),
        features=np.zeros((len(history), edge_count, features), dtype=np.float32),
        feature_mask=np.ones((len(history), edge_count, features), dtype=bool),
        speed_targets=np.ones((edge_count, horizons), dtype=np.float32),
        speed_target_mask=np.ones((edge_count, horizons), dtype=bool),
        traversal_targets=np.ones((edge_count, horizons), dtype=np.float32),
        traversal_target_mask=np.ones((edge_count, horizons), dtype=bool),
        label_available_at_s=available,
        scenario_id="scenario",
        graph_version="graph",
    )


def test_cache_roundtrip_preserves_per_window_times(tmp_path: Path) -> None:
    windows = [_window(issue) for issue in (1020, 1080, 1320)]
    save_episode_windows(tmp_path, "ep-test", "fingerprint", windows)
    reloaded = load_episode_windows(tmp_path, "ep-test", "fingerprint")
    assert reloaded is not None and len(reloaded) == len(windows)
    for original, cached in zip(windows, reloaded):
        assert cached.issue_time_s == original.issue_time_s
        assert cached.history_times_s == original.history_times_s
        assert cached.target_times_s == original.target_times_s
        assert_window_causal(cached)


def test_v1_cache_without_time_arrays_is_a_miss(tmp_path: Path) -> None:
    windows = [_window(issue) for issue in (1020, 1080)]
    save_episode_windows(tmp_path, "ep-test", "fingerprint", windows)
    npz_path = tmp_path / "ep-test.windows.npz"
    data = dict(np.load(npz_path, allow_pickle=False))
    del data["history_times_s"]
    del data["target_times_s"]
    np.savez_compressed(npz_path, **data)
    assert load_episode_windows(tmp_path, "ep-test", "fingerprint") is None
