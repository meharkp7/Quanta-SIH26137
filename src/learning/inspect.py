"""Machine-readable causal inspection helpers for Step 12 windows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.learning.loader import WindowDataset
from src.learning.schema import FEATURE_NAMES, HISTORY_MINUTES, HORIZONS_MINUTES
from src.learning.windows import ForecastWindow, assert_window_causal


def inspect_window(window: ForecastWindow) -> dict:
    """Return a compact audit record for one forecast window."""
    if not isinstance(window, ForecastWindow):
        raise TypeError("window must be a ForecastWindow")
    assert_window_causal(window)
    if not window.edge_ids:
        raise ValueError("window must contain at least one edge")

    speed_available = int(window.speed_target_mask.sum())
    traversal_available = int(window.traversal_target_mask.sum())
    edge_count = len(window.edge_ids)
    target_count = edge_count * len(window.target_times_s)

    return {
        "episode_id": window.episode_id,
        "split": window.split,
        "issue_time_s": window.issue_time_s,
        "visible_window_s": list(window.history_times_s),
        "label_times_s": list(window.target_times_s),
        "history_minutes": HISTORY_MINUTES,
        "horizons_minutes": list(HORIZONS_MINUTES),
        "feature_names": list(FEATURE_NAMES),
        "feature_shape": list(window.features.shape),
        "chronology": {
            "window_ends_at_or_before_issue": max(window.history_times_s) <= window.issue_time_s,
            "targets_strictly_after_issue": min(window.target_times_s) > window.issue_time_s,
        },
        "coverage": {
            "edge_count": edge_count,
            "speed_available_count": speed_available,
            "traversal_available_count": traversal_available,
            "speed_available_fraction": speed_available / target_count if target_count else 0.0,
            "traversal_available_fraction": traversal_available / target_count if target_count else 0.0,
        },
        "first_edge": {
            "edge_id": window.edge_ids[0],
            "last_visible_speed_ratio": _finite(window.features[-1, 0, 0]),
            "speed_targets": [_finite(value) for value in window.speed_targets[0]],
            "speed_target_available_at_s": [
                _finite(value) for value in window.label_available_at_s[0, :, 0]
            ],
        },
    }


def write_inspect_example(dataset: WindowDataset, path: Path) -> dict:
    """Persist the first window as deterministic JSON, rejecting empty datasets."""
    if not isinstance(dataset, WindowDataset):
        raise TypeError("dataset must be a WindowDataset")
    if not dataset.windows:
        raise ValueError("cannot inspect an empty WindowDataset")
    example = inspect_window(dataset.windows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(example, indent=2), encoding="utf-8")
    return example


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
