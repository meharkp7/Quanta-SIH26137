"""Load Step 11 episodes into batched [B, L, E, F] windows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.contracts.scenario import Scenario
from src.learning.scaling import FeatureScaler
from src.learning.schema import FEATURE_NAMES, HISTORY_MINUTES
from src.learning.windows import ForecastWindow, assert_window_causal, build_episode_windows


@dataclass(frozen=True)
class ForecastBatch:
    features: np.ndarray
    feature_mask: np.ndarray
    speed_targets: np.ndarray
    speed_target_mask: np.ndarray
    traversal_targets: np.ndarray
    traversal_target_mask: np.ndarray
    issue_times_s: tuple[int, ...]
    episode_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(int(item) for item in self.features.shape)


class WindowDataset:
    def __init__(
        self,
        windows: list[ForecastWindow],
        *,
        scaler: FeatureScaler | None = None,
        edge_ids: tuple[str, ...] | None = None,
    ) -> None:
        if not windows:
            raise ValueError("WindowDataset requires at least one window")
        self.windows = windows
        self.edge_ids = edge_ids or windows[0].edge_ids
        for window in windows:
            if window.edge_ids != self.edge_ids:
                raise ValueError("edge order is not stable across windows")
            assert_window_causal(window)
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.windows)

    def batch(self, size: int | None = None) -> ForecastBatch:
        selected = self.windows if size is None else self.windows[:size]
        features = np.stack([window.features for window in selected], axis=0)
        if self.scaler is not None:
            features = self.scaler.transform(features)
            features = np.where(np.isfinite(features), features, 0.0)
        return ForecastBatch(
            features=features.astype(np.float32),
            feature_mask=np.stack([window.feature_mask for window in selected], axis=0),
            speed_targets=np.stack([window.speed_targets for window in selected], axis=0),
            speed_target_mask=np.stack(
                [window.speed_target_mask for window in selected], axis=0
            ),
            traversal_targets=np.stack(
                [window.traversal_targets for window in selected], axis=0
            ),
            traversal_target_mask=np.stack(
                [window.traversal_target_mask for window in selected], axis=0
            ),
            issue_times_s=tuple(window.issue_time_s for window in selected),
            episode_ids=tuple(window.episode_id for window in selected),
            edge_ids=self.edge_ids,
        )


def load_pilot_windows(
    pilot_dir: Path,
    scenario: Scenario,
) -> dict[str, WindowDataset]:
    split = json.loads((Path(pilot_dir) / "split_manifest.json").read_text(encoding="utf-8"))
    datasets: dict[str, WindowDataset] = {}
    train_windows: list[ForecastWindow] = []
    for name in ("train", "validation", "test"):
        windows: list[ForecastWindow] = []
        for episode_id in split[name]:
            episode_dir = Path(pilot_dir) / "episodes" / episode_id
            windows.extend(
                build_episode_windows(episode_dir, scenario, split=name)
            )
        if name == "train":
            train_windows = windows
        datasets[name] = WindowDataset(windows, edge_ids=tuple(e.edge_id for e in scenario.edges))

    train_raw = np.stack([window.features for window in train_windows], axis=0)
    scaler = FeatureScaler.fit(train_raw, FEATURE_NAMES)
    return {
        name: WindowDataset(dataset.windows, scaler=scaler, edge_ids=dataset.edge_ids)
        for name, dataset in datasets.items()
    }


def write_fixture_tensors(output_dir: Path, dataset: WindowDataset) -> Path:
    """Write one inspectable [B,L,E,F] pack from already-built windows."""

    output_dir.mkdir(parents=True, exist_ok=True)
    batch = dataset.batch()
    path = output_dir / "fixture_windows.npz"
    np.savez_compressed(
        path,
        features=batch.features,
        feature_mask=batch.feature_mask,
        speed_targets=batch.speed_targets,
        speed_target_mask=batch.speed_target_mask,
        history_minutes=np.array([HISTORY_MINUTES]),
    )
    return path
