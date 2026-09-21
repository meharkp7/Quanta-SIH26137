"""On-disk cache for built forecast windows (NPZ per episode).

Rebuilding windows from per-episode CSVs costs ~1 h on a 1750-episode corpus,
paid on EVERY training invocation (each sweep config reloads from scratch).
This cache stores built windows keyed by episode, validated by a fingerprint
of the source files, so repeat runs load in minutes.

Invalidation: fingerprint covers file sizes + mtimes of observations.csv,
labels.jsonl and episode_manifest.json plus a schema constant. Any content
change (size/mtime shift) or code change (bump WINDOW_CACHE_VERSION) is a
miss and the episode rebuilds. Corrupt cache files are treated as misses,
never as errors.

Version history:
  v1 stored a single history/target time vector from the first window and
  reused it for every window of the episode. Forecast windows have
  per-issue times (history slides with issue_time_s), so every cached
  window after the first failed ``assert_window_causal`` with
  "target times must be strictly after issue_time" and blocked all
  corpus_v2 training/sweep loads. v2 stores per-window time arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.learning.windows import ForecastWindow

WINDOW_CACHE_VERSION = 2


def episode_fingerprint(episode_dir: Path) -> str:
    parts = [f"v{WINDOW_CACHE_VERSION}"]
    for name in ("observations.csv", "labels.jsonl", "episode_manifest.json"):
        path = episode_dir / name
        stat = path.stat()
        parts.append(f"{name}:{stat.st_size}:{int(stat.st_mtime_ns)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _cache_paths(cache_dir: Path, episode_id: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"{episode_id}.windows.npz",
        cache_dir / f"{episode_id}.meta.json",
    )


def save_episode_windows(
    cache_dir: Path,
    episode_id: str,
    fingerprint: str,
    windows: list[ForecastWindow],
) -> None:
    if not windows:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = _cache_paths(cache_dir, episode_id)
    first = windows[0]
    stacked = {
        "features": np.stack([w.features for w in windows]),
        "feature_mask": np.stack([w.feature_mask for w in windows]),
        "speed_targets": np.stack([w.speed_targets for w in windows]),
        "speed_target_mask": np.stack([w.speed_target_mask for w in windows]),
        "traversal_targets": np.stack([w.traversal_targets for w in windows]),
        "traversal_target_mask": np.stack(
            [w.traversal_target_mask for w in windows]
        ),
        "label_available_at_s": np.stack(
            [w.label_available_at_s for w in windows]
        ),
        "issue_time_s": np.asarray(
            [w.issue_time_s for w in windows], dtype=np.int64
        ),
        # Per-window chronology: history slides with issue_time_s and each
        # issue has its own target buckets. All windows of an episode share
        # L/H lengths; store the full [N, L] / [N, H] arrays so a reload is
        # byte-identical to a fresh build_episode_windows() call.
        "history_times_s": np.asarray(
            [w.history_times_s for w in windows], dtype=np.int64
        ),
        "target_times_s": np.asarray(
            [w.target_times_s for w in windows], dtype=np.int64
        ),
    }
    tmp = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **stacked)
    tmp.replace(npz_path)
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "cache_version": WINDOW_CACHE_VERSION,
                "episode_id": first.episode_id,
                "split": first.split,
                "scenario_id": first.scenario_id,
                "graph_version": first.graph_version,
                "edge_ids": list(first.edge_ids),
            }
        ),
        encoding="utf-8",
    )


def load_episode_windows(
    cache_dir: Path,
    episode_id: str,
    fingerprint: str,
) -> list[ForecastWindow] | None:
    npz_path, meta_path = _cache_paths(cache_dir, episode_id)
    try:
        if not (npz_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            return None
        # v1 cache files stored one shared time vector (first window) and are
        # causally wrong for every later window: force a rebuild.
        if meta.get("cache_version", 1) < WINDOW_CACHE_VERSION:
            return None
        data = np.load(npz_path, allow_pickle=False)
        if "history_times_s" not in data or "target_times_s" not in data:
            return None
        edge_ids = tuple(meta["edge_ids"])
        count = int(data["issue_time_s"].shape[0])
        histories = np.asarray(data["history_times_s"])
        targets = np.asarray(data["target_times_s"])
        if histories.shape[0] != count or targets.shape[0] != count:
            return None
        return [
            ForecastWindow(
                episode_id=meta["episode_id"],
                split=meta["split"],
                issue_time_s=int(data["issue_time_s"][i]),
                history_times_s=tuple(int(v) for v in histories[i].tolist()),
                target_times_s=tuple(int(v) for v in targets[i].tolist()),
                edge_ids=edge_ids,
                features=np.asarray(data["features"][i], dtype=np.float32),
                feature_mask=np.asarray(data["feature_mask"][i], dtype=bool),
                speed_targets=np.asarray(
                    data["speed_targets"][i], dtype=np.float32
                ),
                speed_target_mask=np.asarray(
                    data["speed_target_mask"][i], dtype=bool
                ),
                traversal_targets=np.asarray(
                    data["traversal_targets"][i], dtype=np.float32
                ),
                traversal_target_mask=np.asarray(
                    data["traversal_target_mask"][i], dtype=bool
                ),
                label_available_at_s=np.asarray(
                    data["label_available_at_s"][i], dtype=np.float32
                ),
                scenario_id=meta["scenario_id"],
                graph_version=meta["graph_version"],
            )
            for i in range(count)
        ]
    except Exception:
        return None
