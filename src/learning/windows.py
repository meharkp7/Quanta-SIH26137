"""Causal 12-minute history / 5-10-15-minute target windows."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.contracts.core_types import TargetKind
from src.contracts.scenario import Scenario
from src.data.causal_labels import MatureLabel
from src.learning.schema import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    HISTORY_MINUTES,
    HORIZONS_MINUTES,
    INTERVAL_S,
)


@dataclass(frozen=True)
class ForecastWindow:
    episode_id: str
    split: str
    issue_time_s: int
    history_times_s: tuple[int, ...]
    target_times_s: tuple[int, ...]
    edge_ids: tuple[str, ...]
    features: np.ndarray
    feature_mask: np.ndarray
    speed_targets: np.ndarray
    speed_target_mask: np.ndarray
    traversal_targets: np.ndarray
    traversal_target_mask: np.ndarray
    label_available_at_s: np.ndarray
    scenario_id: str = "unknown"
    graph_version: str = "unknown"

    @property
    def valid_edge_count(self) -> int:
        """Number of non-padding edges represented by this window."""
        return len(self.edge_ids)

    def supervision_stats(self) -> dict[str, object]:
        """Return padding-aware, horizon-wise target supervision statistics."""
        denominator = self.valid_edge_count
        horizons = []
        for index, target_minute in enumerate((5, 10, 15)):
            speed_count = int(self.speed_target_mask[:, index].sum())
            traversal_count = int(self.traversal_target_mask[:, index].sum())
            horizons.append({
                "horizon_minutes": target_minute,
                "edge_count": denominator,
                "speed_valid_count": speed_count,
                "speed_coverage": float(speed_count / denominator) if denominator else 0.0,
                "traversal_valid_count": traversal_count,
                "traversal_coverage": float(traversal_count / denominator) if denominator else 0.0,
            })
        return {
            "edge_count": denominator,
            "horizons": horizons,
        }


def build_episode_windows(
    episode_dir: Path,
    scenario: Scenario,
    *,
    split: str,
    interval_s: int = INTERVAL_S,
    history_minutes: int = HISTORY_MINUTES,
    horizons_minutes: Sequence[int] = HORIZONS_MINUTES,
) -> list[ForecastWindow]:
    with (episode_dir / "observations.csv").open(encoding="utf-8", newline="") as handle:
        observations = list(csv.DictReader(handle))
    labels = [
        MatureLabel(**json.loads(line))
        for line in (episode_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((episode_dir / "episode_manifest.json").read_text(encoding="utf-8"))
    episode_id = manifest["episode_id"]
    edge_ids = tuple(edge.edge_id for edge in scenario.edges)
    limits = {edge.edge_id: float(edge.speed_limit_mps) for edge in scenario.edges}
    times = sorted({int(row["observation_time_s"]) for row in observations})
    by_key = {(str(row["edge_id"]), int(row["observation_time_s"])): row for row in observations}
    label_index = {
        (item.edge_id, item.issue_time_s, item.target_time_s, item.target_kind): item
        for item in labels
    }
    issue_times = sorted({item.issue_time_s for item in labels})

    if interval_s != 60:
        raise ValueError("Step 12 requires one-minute observations")
    if manifest.get("scenario_id") != scenario.scenario_id:
        raise ValueError("Episode scenario does not match the supplied graph")
    history_steps = history_minutes
    horizon_s = tuple(int(item) * interval_s for item in horizons_minutes)
    windows: list[ForecastWindow] = []
    for issue in issue_times:
        history = tuple(issue - (history_steps - 1 - index) * interval_s for index in range(history_steps))
        targets = tuple(issue + horizon for horizon in horizon_s)
        if any(time_s < min(times) or time_s > max(times) for time_s in history):
            continue
        if any(time_s not in times for time_s in targets):
            continue
        if history[0] < 0:
            continue

        features = np.full(
            (history_steps, len(edge_ids), FEATURE_COUNT),
            np.nan,
            dtype=np.float32,
        )
        feature_mask = np.zeros((history_steps, len(edge_ids), FEATURE_COUNT), dtype=np.bool_)
        for t_index, time_s in enumerate(history):
            if time_s > issue:
                raise RuntimeError("history timestamp leaked past issue time")
            for e_index, edge_id in enumerate(edge_ids):
                row = by_key.get((edge_id, time_s))
                _fill_features(features, feature_mask, t_index, e_index, row, limits[edge_id])

        speed = np.full((len(edge_ids), len(targets)), np.nan, dtype=np.float32)
        speed_mask = np.zeros((len(edge_ids), len(targets)), dtype=np.bool_)
        traversal = np.full((len(edge_ids), len(targets)), np.nan, dtype=np.float32)
        traversal_mask = np.zeros((len(edge_ids), len(targets)), dtype=np.bool_)
        available = np.full((len(edge_ids), len(targets), 2), np.nan, dtype=np.float32)
        for e_index, edge_id in enumerate(edge_ids):
            for h_index, target in enumerate(targets):
                speed_label = label_index.get(
                    (edge_id, issue, target, TargetKind.SPEED_PROXY.value)
                )
                trav_label = label_index.get(
                    (edge_id, issue, target, TargetKind.REALIZED_TRAVERSAL.value)
                )
                if speed_label and not speed_label.missing and speed_label.value is not None:
                    speed[e_index, h_index] = speed_label.value / limits[edge_id]
                    speed_mask[e_index, h_index] = True
                    available[e_index, h_index, 0] = speed_label.available_at_s or np.nan
                if trav_label and not trav_label.missing and trav_label.value is not None:
                    traversal[e_index, h_index] = trav_label.value
                    traversal_mask[e_index, h_index] = True
                    available[e_index, h_index, 1] = trav_label.available_at_s or np.nan

        windows.append(
            ForecastWindow(
                episode_id=episode_id,
                split=split,
                issue_time_s=issue,
                history_times_s=history,
                target_times_s=targets,
                edge_ids=edge_ids,
                features=features,
                feature_mask=feature_mask,
                speed_targets=speed,
                speed_target_mask=speed_mask,
                traversal_targets=traversal,
                traversal_target_mask=traversal_mask,
                label_available_at_s=available,
                scenario_id=scenario.scenario_id,
                graph_version=scenario.graph_version,
            )
        )
    return windows


def _fill_features(
    features: np.ndarray,
    mask: np.ndarray,
    t_index: int,
    e_index: int,
    row: dict | None,
    speed_limit: float,
) -> None:
    missing = row is None or bool(int(row.get("missing") or 0))
    features[t_index, e_index, 4] = float(missing)
    mask[t_index, e_index, 4] = True
    if row is None:
        return
    # Availability metadata is independent of traffic sensor availability.
    for key, index, divisor in (("observation_age_s", 3, 60.0), ("known_closed", 5, 1.0)):
        raw = row.get(key)
        if raw not in (None, "") and np.isfinite(float(raw)):
            features[t_index, e_index, index] = float(raw) / divisor
            mask[t_index, e_index, index] = True
    if missing:
        return
    for key, index, divisor in (("observed_speed_mps", 0, speed_limit),
                                ("occupancy", 1, 1.0), ("halting_count", 2, 1.0)):
        raw = row.get(key)
        if raw not in (None, "") and np.isfinite(float(raw)):
            features[t_index, e_index, index] = float(raw) / max(divisor, 1e-6)
            mask[t_index, e_index, index] = True


def assert_window_causal(window: ForecastWindow) -> None:
    if max(window.history_times_s) > window.issue_time_s:
        raise AssertionError("visible window contains times after issue_time")
    if min(window.target_times_s) <= window.issue_time_s:
        raise AssertionError("target times must be strictly after issue_time")
    if list(window.target_times_s) != sorted(window.target_times_s):
        raise AssertionError("target times must increase")
    if window.features.shape != (
        len(window.history_times_s),
        len(window.edge_ids),
        FEATURE_COUNT,
    ):
        raise AssertionError("feature shape is not [L, E, F]")

    if len(set(window.edge_ids)) != len(window.edge_ids):
        raise AssertionError("Duplicate edge IDs")
    for channel, targets, valid in ((0, window.speed_targets, window.speed_target_mask),
                                     (1, window.traversal_targets, window.traversal_target_mask)):
        available = window.label_available_at_s[..., channel]
        if np.any(valid & (~np.isfinite(targets) | ~np.isfinite(available))):
            raise AssertionError("Valid labels require finite values and maturity timestamps")
        if np.any(valid & (available < np.asarray(window.target_times_s)[None, :])):
            raise AssertionError("Label maturity precedes target time")
