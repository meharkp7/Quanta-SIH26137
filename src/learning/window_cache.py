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
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.learning.windows import ForecastWindow

WINDOW_CACHE_VERSION = 1


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
    }
    tmp = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **stacked)
    tmp.replace(npz_path)
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "episode_id": first.episode_id,
                "split": first.split,
                "scenario_id": first.scenario_id,
                "graph_version": first.graph_version,
                "edge_ids": list(first.edge_ids),
                "history_times_s": list(first.history_times_s),
                "target_times_s": list(first.target_times_s),
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
        data = np.load(npz_path, allow_pickle=False)
        edge_ids = tuple(meta["edge_ids"])
        history = tuple(meta["history_times_s"])
        targets = tuple(meta["target_times_s"])
        count = int(data["issue_time_s"].shape[0])
        return [
            ForecastWindow(
                episode_id=meta["episode_id"],
                split=meta["split"],
                issue_time_s=int(data["issue_time_s"][i]),
                history_times_s=history,
                target_times_s=targets,
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
