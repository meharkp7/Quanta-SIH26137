"""Build road-edge traffic episodes from empirical traffic observations.

This is a replay/conditioning layer, not a simulator. It maps empirical sensor
trajectories onto a static road graph while preserving the empirical temporal
shape. It is intended for development and distributional validation; SUMO
remains the authoritative source for intervention and vehicle-interaction
truth in the final dataset.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from src.contracts.scenario import Scenario

from .empirical_traffic import TrafficProfile


def generate_real_conditioned_episode(
    scenario: Scenario,
    profile: TrafficProfile,
    output_dir: str | Path,
    *,
    start_index: int = 0,
    duration_steps: int = 288,
    edge_seed: int = 26137,
) -> Path:

    if start_index < 0 or start_index >= len(profile.timestamps):
        raise ValueError(
            "start_index outside empirical profile"
        )

    end = min(
        len(profile.timestamps),
        start_index + duration_steps,
    )

    if end - start_index < 2:
        raise ValueError(
            "episode needs at least two empirical timesteps"
        )

    out = Path(output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(edge_seed)

    # Keep deterministic behaviour explicit.
    _ = rng

    sensors = list(profile.sensor_ids)

    usable = [
        i
        for i, s in enumerate(profile.sensor_stats)
        if s.get("quality") == "usable"
    ]

    if not usable:
        raise ValueError(
            "No usable sensors in empirical profile"
        )

    # Deterministically assign road parents to empirical sensors.
    # Reuse is allowed because sensor networks are smaller than delivery
    # road graphs.
    parent_ids = sorted(
        {
            e.parent_road_id
            for e in scenario.edges
        }
    )

    parent_to_sensor = {
        parent: sensors[
            usable[i % len(usable)]
        ]
        for i, parent in enumerate(parent_ids)
    }

    sensor_index = {
        sensor: i
        for i, sensor in enumerate(sensors)
    }

    stats_by_sensor = {
        s["sensor_id"]: s
        for s in profile.sensor_stats
    }

    truth_rows: list[dict] = []

    for step, source_i in enumerate(
        range(start_index, end)
    ):
        timestamp = profile.timestamps[source_i]

        for edge in scenario.edges:
            sensor = parent_to_sensor[
                edge.parent_road_id
            ]

            j = sensor_index[sensor]

            raw = profile.values[
                source_i
            ][j]

            stat = stats_by_sensor[sensor]

            if (
                raw is None
                or stat["median"] in (None, 0)
            ):
                value = None
                ratio = None
                travel = None

            else:
                value = float(raw)

                ratio = (
                    value
                    / float(stat["median"])
                )

                ratio = max(
                    0.05,
                    min(1.50, ratio),
                )

                travel = (
                    edge.length_m
                    / max(
                        0.1,
                        edge.speed_limit_mps * ratio,
                    )
                )

            truth_rows.append(
                {
                    "episode_time_s": (
                        step * profile.interval_s
                    ),
                    "source_timestamp": timestamp,
                    "edge_id": edge.edge_id,
                    "parent_road_id": edge.parent_road_id,
                    "empirical_sensor_id": sensor,
                    "empirical_value": (
                        ""
                        if value is None
                        else round(value, 6)
                    ),
                    "empirical_speed_ratio": (
                        ""
                        if ratio is None
                        else round(ratio, 6)
                    ),
                    "derived_edge_travel_time_s": (
                        ""
                        if travel is None
                        else round(travel, 6)
                    ),
                    "target_kind": "empirical_replay",
                }
            )

    _write_csv(
        out / "empirical_edge_truth.csv",
        truth_rows,
    )

    _write_json(
        out / "episode_manifest.json",
        {
            "schema_version":
                "real-conditioned-episode-1.0",
            "backend":
                "empirical_replay_v1",
            "source_name":
                profile.source_name,
            "source_observations": [
                profile.timestamps[start_index],
                profile.timestamps[end - 1],
            ],
            "interval_s":
                profile.interval_s,
            "edge_count":
                len(scenario.edges),
            "mapping":
                (
                    "deterministic parent-road to empirical-sensor "
                    "assignment; sensor reuse allowed"
                ),
            "warning":
                (
                    "Derived edge travel times are replay proxies, "
                    "not measured traversals."
                ),
            "sensor_seed":
                edge_seed,
            "anomaly_candidates":
                len(profile.anomaly_events),
        },
    )

    _write_json(
        out / "sensor_mapping.json",
        parent_to_sensor,
    )

    return out


def _write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    path: Path,
    data: object,
) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )