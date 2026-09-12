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


def build_episode_windows(
    episode_dir: Path,
    scenario: Scenario,
    *,
    split: str,
    interval_s: int = INTERVAL_S,
    history_minutes: int = HISTORY_MINUTES,
    horizons_minutes: Sequence[int] = HORIZONS_MINUTES,
) -> list[ForecastWindow]:
    observations = list(csv.DictReader((episode_dir / "observations.csv").open(encoding="utf-8")))
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
    if row is None or bool(int(row.get("missing") or 0)):
        features[t_index, e_index, 4] = 1.0
        mask[t_index, e_index, 4] = True
        return
    speed = row.get("observed_speed_mps")
    occupancy = row.get("occupancy")
    halt = row.get("halting_count")
    age = float(row.get("observation_age_s") or 0.0)
    closed = float(int(row.get("known_closed") or 0))
    if speed not in (None, ""):
        features[t_index, e_index, 0] = float(speed) / max(speed_limit, 1e-6)
        mask[t_index, e_index, 0] = True
    if occupancy not in (None, ""):
        features[t_index, e_index, 1] = float(occupancy)
        mask[t_index, e_index, 1] = True
    if halt not in (None, ""):
        features[t_index, e_index, 2] = float(halt)
        mask[t_index, e_index, 2] = True
    features[t_index, e_index, 3] = age / 60.0
    features[t_index, e_index, 4] = 0.0
    features[t_index, e_index, 5] = closed
    mask[t_index, e_index, 3] = True
    mask[t_index, e_index, 4] = True
    mask[t_index, e_index, 5] = True


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
