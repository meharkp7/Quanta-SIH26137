"""Causal graph-aware batches with explicit road padding and metadata."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from src.contracts.scenario import Scenario
from src.learning.scaling import FeatureScaler
from src.learning.schema import FEATURE_NAMES, HISTORY_MINUTES
from src.learning.windows import ForecastWindow, assert_window_causal, build_episode_windows
from src.learning.evaluation import validate_manifest_splits

@dataclass(frozen=True)
class ForecastBatch:
    features: np.ndarray
    raw_speed_ratio: np.ndarray
    feature_mask: np.ndarray
    speed_targets: np.ndarray
    speed_target_mask: np.ndarray
    traversal_targets: np.ndarray
    traversal_target_mask: np.ndarray
    issue_times_s: tuple[int, ...]
    episode_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]  # Legacy common ordering; empty for mixed graphs.
    edge_padding_mask: np.ndarray  # [B,E], True means padding, NOT missing sensor.
    edge_ids_by_sample: np.ndarray
    history_times_s: np.ndarray
    target_times_s: np.ndarray
    label_available_at_s: np.ndarray
    scenario_ids: tuple[str, ...]
    graph_versions: tuple[str, ...]
    splits: tuple[str, ...]

    @property
    def shape(self):
        return tuple(int(item) for item in self.features.shape)

class WindowDataset:
    def __init__(self, windows, *, scaler=None, edge_ids=None):
        if not windows:
            raise ValueError("WindowDataset requires at least one window")
        self.windows = windows
        self.scaler = scaler
        ordering = {}
        for window in windows:
            assert_window_causal(window)
            key = (window.scenario_id, window.graph_version)
            if key in ordering and ordering[key] != window.edge_ids:
                raise ValueError("edge order is not stable across windows of the same graph")
            ordering[key] = window.edge_ids
            if edge_ids is not None and tuple(edge_ids) != window.edge_ids:
                raise ValueError("edge order is not stable across windows")
        first = windows[0].edge_ids
        self.edge_ids = first if all(w.edge_ids == first for w in windows) else ()

    def __len__(self):
        return len(self.windows)

    def batch(self, size=None):
        selected = self.windows if size is None else self.windows[:size]
        if not selected:
            raise ValueError("Batch must contain at least one window")
        b, l, e, f = len(selected), len(selected[0].history_times_s), max(len(w.edge_ids) for w in selected), len(FEATURE_NAMES)
        h = len(selected[0].target_times_s)
        raw = np.full((b,l,e,f), np.nan, dtype=np.float32)
        feature_mask = np.zeros(raw.shape, dtype=bool)
        speed = np.full((b,e,h), np.nan, dtype=np.float32)
        traversal = speed.copy()
        speed_mask = np.zeros(speed.shape, dtype=bool)
        traversal_mask = speed_mask.copy()
        available = np.full((b,e,h,2), np.nan, dtype=np.float64)
        padding = np.ones((b,e), dtype=bool)
        width = max(len(eid) for w in selected for eid in w.edge_ids)
        ids = np.full((b,e), "", dtype=f"<U{max(width, 1)}")
        for i,w in enumerate(selected):
            n = len(w.edge_ids)
            if len(w.history_times_s) != l or len(w.target_times_s) != h:
                raise ValueError("Inconsistent history or horizon lengths")
            raw[i,:,:n] = np.where(w.feature_mask, w.features, np.nan)
            feature_mask[i,:,:n] = w.feature_mask
            speed[i,:n], traversal[i,:n] = w.speed_targets, w.traversal_targets
            speed_mask[i,:n], traversal_mask[i,:n] = w.speed_target_mask, w.traversal_target_mask
            available[i,:n] = w.label_available_at_s
            padding[i,:n] = False
            ids[i,:n] = w.edge_ids
        features = self.scaler.transform(raw) if self.scaler is not None else raw.copy()
        features = np.where(feature_mask & np.isfinite(features), features, 0.0)
        return ForecastBatch(features, raw[...,0].copy(), feature_mask, speed, speed_mask,
            traversal, traversal_mask, tuple(w.issue_time_s for w in selected),
            tuple(w.episode_id for w in selected), self.edge_ids, padding, ids,
            np.asarray([w.history_times_s for w in selected]),
            np.asarray([w.target_times_s for w in selected]), available,
            tuple(w.scenario_id for w in selected), tuple(w.graph_version for w in selected),
            tuple(w.split for w in selected))

def _build_episode_job(args) -> tuple[list, bool]:
    """Picklable episode build for the process pool (ordered map).

    Returns (windows, cache_hit). Cache misses rebuild from CSV and
    repopulate the cache; corrupt entries are misses, never errors.
    """
    from src.learning.window_cache import (
        episode_fingerprint,
        load_episode_windows,
        save_episode_windows,
    )
    from src.learning.windows import build_episode_windows
    from src.contracts.scenario import Scenario

    root_s, episode_id, scenario_json, split, cache_s = args
    scenario = Scenario.model_validate_json(scenario_json)
    folder = Path(root_s) / "episodes" / episode_id
    cache = Path(cache_s) if cache_s is not None else None
    fingerprint = ""
    episode_windows = None
    if cache is not None:
        fingerprint = episode_fingerprint(folder)
        episode_windows = load_episode_windows(cache, episode_id, fingerprint)
    hit = episode_windows is not None
    if not hit:
        episode_windows = build_episode_windows(folder, scenario, split=split)
        if cache is not None:
            save_episode_windows(cache, episode_id, fingerprint, episode_windows)
    return episode_windows, hit


def load_pilot_windows(
    pilot_dir: Path,
    scenario: Scenario | None = None,
    *,
    cache_dir: Path | str | None = None,
) -> dict:
    """Load causal windows, optionally via the per-episode NPZ window cache.

    ``cache_dir`` (e.g. ``<corpus>/.window_cache``) turns repeat loads from
    ~1 h into minutes on thousand-episode corpora. Cached windows are
    validated by source-file fingerprint; misses and corrupt entries rebuild
    transparently. The scaler is always fit fresh on training data.
    """
    from src.learning.window_cache import (
        episode_fingerprint,
        load_episode_windows,
        save_episode_windows,
    )

    root = Path(pilot_dir)
    cache = Path(cache_dir) if cache_dir is not None else None
    split = json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8"))
    names = ("train", "validation", "test")
    validate_manifest_splits(split)
    all_ids = [eid for name in names for eid in split[name]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Episode leakage across splits")
    scenarios = {}
    for sid, relative_path in split.get("scenario_files", {}).items():
        scenarios[sid] = Scenario.model_validate_json((root / relative_path).read_text(encoding="utf-8"))
    datasets, graph_owners = {}, {}
    cache_hits, cache_misses = 0, 0
    import concurrent.futures
    import os

    for name in names:
        # Manifest parsing is cheap and stays serial (split validation,
        # leakage checks). Only the heavy CSV->window build parallelizes.
        jobs = []
        for episode_id in split[name]:
            folder = root / "episodes" / episode_id
            manifest = json.loads(
                (folder / "episode_manifest.json").read_text(encoding="utf-8")
            )
            sc = scenarios.get(manifest["scenario_id"], scenario)
            if sc is None:
                raise ValueError("No scenario file for episode")
            if manifest.get("split") != name:
                raise ValueError(
                    "Episode manifest disagrees with split assignment"
                )
            if split.get("map_disjoint"):
                key = split["map_fingerprints"][sc.scenario_id]
                if key in graph_owners and graph_owners[key] != name:
                    raise ValueError("Base map leakage across splits")
                graph_owners[key] = name
            jobs.append((episode_id, sc))

        windows = []
        # Process pool: episode builds are GIL-bound CPython (csv + loops),
        # threads measured 1.00x. Ordered map preserves episode order, so
        # downstream batching/shuffling stays deterministic.
        workers = max(1, (os.cpu_count() or 4) // 2)
        ctx_args = [
            (
                str(root),
                episode_id,
                # JSON string: Scenario objects do not pickle; the worker
                # revalidates (cheap, once per episode).
                sc.model_dump_json(),
                name,
                str(cache) if cache is not None else None,
            )
            for episode_id, sc in jobs
        ]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers
        ) as pool:
            for episode_windows, hit in pool.map(_build_episode_job, ctx_args):
                if hit:
                    cache_hits += 1
                else:
                    cache_misses += 1
                windows.extend(episode_windows)
        datasets[name] = WindowDataset(windows)
    if cache is not None:
        print(
            f"Window cache: {cache_hits} hits, {cache_misses} misses "
            f"({cache})",
            flush=True,
        )
    # Streaming scaler fit on training windows only: never materializes the
    # dense [B, L, Emax, F] batch (tens of GB on thousand-episode corpora).
    # Invalid entries are restored to NaN first (zero is real data).
    scaler = FeatureScaler.fit_windows(
        datasets["train"].windows, FEATURE_NAMES
    )
    return {
        name: WindowDataset(ds.windows, scaler=scaler)
        for name, ds in datasets.items()
    }

def write_fixture_tensors(output_dir: Path, dataset: WindowDataset) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = dataset.batch()
    path = output_dir / "fixture_windows.npz"
    np.savez_compressed(path, **{key: np.asarray(value) for key,value in vars(batch).items()},
        feature_names=np.asarray(FEATURE_NAMES), history_minutes=np.asarray([HISTORY_MINUTES]))
    return path
