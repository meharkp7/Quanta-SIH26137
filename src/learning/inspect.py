"""One manually inspectable chronology example for Step 12."""

from __future__ import annotations

import json
from pathlib import Path

from src.learning.loader import WindowDataset
from src.learning.schema import FEATURE_NAMES, HISTORY_MINUTES, HORIZONS_MINUTES
from src.learning.windows import ForecastWindow


def inspect_window(window: ForecastWindow) -> dict:
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
            "window_ends_at_or_before_issue": max(window.history_times_s)
            <= window.issue_time_s,
            "targets_strictly_after_issue": min(window.target_times_s)
            > window.issue_time_s,
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
    example = inspect_window(dataset.windows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(example, indent=2), encoding="utf-8")
    return example


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number
