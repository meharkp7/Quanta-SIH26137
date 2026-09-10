"""Empirical disruption calibration from real traffic observations.

The output is a distributional prior, not an incident label. It is used to
parameterize SUMO interventions so that synthetic incidents have magnitudes,
durations and recovery shapes grounded in observed traffic behaviour.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .empirical_traffic import TrafficProfile


def calibrate_event_priors(
    profile: TrafficProfile,
    slowdown_ratio: float = 0.80,
) -> dict:
    durations: list[int] = []
    magnitudes: list[float] = []
    recovery_steps: list[int] = []

    for j, sensor in enumerate(profile.sensor_ids):
        stats = profile.sensor_stats[j]
        baseline = stats.get("median")

        if baseline in (None, 0):
            continue

        active = False
        start = 0
        min_ratio = 1.0
        recover = None

        for i, row in enumerate(profile.values):
            value = row[j]

            ratio = (
                None
                if value is None
                else float(value) / float(baseline)
            )

            if ratio is not None and ratio < slowdown_ratio:
                if not active:
                    active = True
                    start = i
                    min_ratio = ratio
                else:
                    min_ratio = min(min_ratio, ratio)

                continue

            if active:
                duration = i - start

                if duration >= 2:
                    durations.append(duration)

                    magnitudes.append(
                        max(0.0, 1.0 - min_ratio)
                    )

                    recover = i

                    recovery_steps.append(
                        max(
                            0,
                            recover - (start + duration),
                        )
                    )

                active = False
                min_ratio = 1.0

    return {
        "schema_version": "event-prior-1.0",
        "source_name": profile.source_name,
        "source_kind": "empirical_traffic",
        "warning": (
            "These are traffic-shock priors, not "
            "ground-truth incident annotations."
        ),
        "slowdown_ratio_threshold": slowdown_ratio,
        "sample_count": len(durations),
        "duration_steps": _summary(durations),
        "magnitude_fraction": _summary(magnitudes),
        "recovery_steps": _summary(recovery_steps),
        "recommended_use": (
            "sample SUMO incident/closure severity and recovery "
            "distributions; never expose priors as future observations"
        ),
    }


def write_event_priors(
    profile: TrafficProfile,
    output: str | Path,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            calibrate_event_priors(profile),
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def _summary(
    values: list[float | int],
) -> dict:
    if not values:
        return {
            "count": 0,
            "median": None,
            "p10": None,
            "p90": None,
        }

    xs = sorted(
        float(x)
        for x in values
    )

    return {
        "count": len(xs),
        "median": statistics.median(xs),
        "p10": _quantile(xs, 0.10),
        "p90": _quantile(xs, 0.90),
    }


def _quantile(
    values: list[float],
    q: float,
) -> float:
    pos = (len(values) - 1) * q

    lo = int(pos)
    hi = min(
        len(values) - 1,
        lo + 1,
    )

    return values[lo] + (values[hi] - values[lo]) * (pos - lo)