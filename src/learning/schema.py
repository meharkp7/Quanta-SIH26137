"""Frozen Step 12 window schema."""

from __future__ import annotations

HISTORY_MINUTES = 12
HORIZONS_MINUTES = (5, 10, 15)
INTERVAL_S = 60
FEATURE_NAMES = (
    "speed_ratio",
    "occupancy",
    "halting",
    "observation_age_min",
    "missing",
    "known_closed",
)
FEATURE_COUNT = len(FEATURE_NAMES)
