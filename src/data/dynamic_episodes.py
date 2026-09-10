"""Dynamic traffic episode generation for the Quanta-SIH26137 dataset.

This module is intentionally simulator-independent.  It creates a reproducible
*traffic/event scenario specification* and a causal edge-level time series that
can be used to validate the data pipeline before SUMO is installed.  The
records are labelled as ``synthetic_field_v1`` rather than SUMO truth.

The future/hidden side is kept separate from policy-visible observations.  A
future event can exist in ``events.json`` while remaining absent from an
observation until its reveal time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import random
from typing import Iterable

from src.contracts.scenario import Scenario


TRAFFIC_REGIMES = ("normal", "morning_peak", "evening_peak", "corridor_surge")
EVENT_TYPES = ("closure", "incident", "multi_disruption", "scheduled_closure")


@dataclass(frozen=True)
class DynamicEpisodeConfig:
    duration_s: int = 1800
    interval_s: int = 60
    warmup_s: int = 300
    regime: str = "normal"
    event_type: str | None = None
    event_count: int = 1
    event_duration_s: int = 240
    reveal_lead_s: int = 60
    observation_missing_fraction: float = 0.02
    observation_delay_s: int = 0
    speed_noise_fraction: float = 0.025
    occupancy_noise: float = 0.015
    seed: int = 26137
    backend: str = "synthetic_field_v1"

    def __post_init__(self) -> None:
        if self.duration_s <= 0 or self.interval_s <= 0:
            raise ValueError("duration_s and interval_s must be positive")
        if self.duration_s % self.interval_s:
            raise ValueError("duration_s must be divisible by interval_s")
        if self.warmup_s < 0 or self.warmup_s >= self.duration_s:
            raise ValueError("warmup_s must be in [0, duration_s)")
        if self.regime not in TRAFFIC_REGIMES:
            raise ValueError(f"regime must be one of {TRAFFIC_REGIMES}")
        if self.event_type is not None and self.event_type not in EVENT_TYPES:
            raise ValueError(f"event_type must be one of {EVENT_TYPES}")
        if self.event_count < 0:
            raise ValueError("event_count must be non-negative")
        if self.event_duration_s <= 0:
            raise ValueError("event_duration_s must be positive")
        if self.reveal_lead_s < 0:
            raise ValueError("reveal_lead_s must be non-negative")
        if not 0 <= self.observation_missing_fraction < 1:
            raise ValueError("observation_missing_fraction must be in [0, 1)")
        if self.speed_noise_fraction < 0 or self.occupancy_noise < 0:
            raise ValueError("observation noise must be non-negative")


@dataclass(frozen=True)
class DynamicEvent:
    event_id: str
    event_type: str
    generation_time_s: int
    reveal_time_s: int
    effect_start_s: int
    effect_end_s: int
    affected_parent_road_ids: tuple[str, ...]
    severity: float


def generate_episode(
    scenario: Scenario,
    output_dir: str | Path,
    config: DynamicEpisodeConfig = DynamicEpisodeConfig(),
    episode_id: str | None = None,
) -> Path:
    """Generate one causal dynamic episode from an existing static scenario."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episode_id = episode_id or f"ep-{scenario.scenario_id.replace(':', '-')}-{config.seed}"
    events = _generate_events(scenario, config, episode_id)
    truth_rows = _generate_truth(scenario, config, events)
    observation_rows = _generate_observations(scenario, config, events, truth_rows)

    _write_csv(output / "edge_truth.csv", truth_rows)
    _write_csv(output / "observations.csv", observation_rows)
    _write_json(output / "events.json", [asdict(e) for e in events])
    _write_json(output / "episode_manifest.json", {
        "schema_version": "dynamic-1.0",
        "episode_id": episode_id,
        "scenario_id": scenario.scenario_id,
        "graph_version": scenario.graph_version,
        "backend": config.backend,
        "target_kind": "synthetic_field_travel_time",
        "time": {"duration_s": config.duration_s, "interval_s": config.interval_s, "warmup_s": config.warmup_s},
        "regime": config.regime,
        "seed": config.seed,
        "events": len(events),
        "causal_visibility": {
            "future_effect_times_hidden": True,
            "future_labels_hidden": True,
            "hidden_events_in_observations": False,
        },
    })
    return output


def _generate_events(scenario: Scenario, config: DynamicEpisodeConfig, episode_id: str) -> list[DynamicEvent]:
    if config.event_type is None or config.event_count == 0:
        return []
    rng = random.Random(config.seed + 101)
    parents = sorted({e.parent_road_id for e in scenario.edges})
    if not parents:
        return []
    # Prefer arterials/collectors because disruptions there create meaningful
    # network-level effects. Fall back to any physical road if needed.
    preferred = [e.parent_road_id for e in scenario.edges if e.road_class.value in {"arterial", "collector"}]
    pool = sorted(set(preferred)) or parents
    chosen = rng.sample(pool, min(config.event_count, len(pool)))
    usable_start = max(config.warmup_s + 60, config.duration_s // 4)
    latest_start = max(usable_start, config.duration_s - config.event_duration_s - 60)
    events: list[DynamicEvent] = []
    for i, parent in enumerate(chosen):
        start = rng.randint(usable_start, latest_start)
        end = min(config.duration_s, start + config.event_duration_s)
        severity = rng.uniform(0.65, 1.0)
        event_type = config.event_type
        if event_type == "multi_disruption":
            event_type = "incident" if i % 2 == 0 else "closure"
        reveal = max(0, start - config.reveal_lead_s)
        events.append(DynamicEvent(
            event_id=f"{episode_id}-EV{i:03d}",
            event_type=event_type,
            generation_time_s=0,
            reveal_time_s=reveal,
            effect_start_s=start,
            effect_end_s=end,
            affected_parent_road_ids=(parent,),
            severity=severity,
        ))
    return sorted(events, key=lambda e: (e.effect_start_s, e.event_id))


def _generate_truth(scenario: Scenario, config: DynamicEpisodeConfig, events: list[DynamicEvent]) -> list[dict]:
    rng = random.Random(config.seed + 211)
    rows: list[dict] = []
    for t in range(0, config.duration_s + 1, config.interval_s):
        phase = _daily_profile(t, config.duration_s, config.regime)
        for edge in scenario.edges:
            event = _active_event(edge.parent_road_id, t, events)
            base_ratio = _edge_load_factor(scenario, edge.edge_id)
            congestion = min(0.92, max(0.0, 0.10 + 0.62 * phase * base_ratio))
            if event:
                if event.event_type in {"closure", "scheduled_closure"}:
                    speed_ratio = 0.0
                    closed = True
                    congestion = 1.0
                else:
                    speed_ratio = max(0.08, 1.0 - min(0.88, congestion + 0.60 * event.severity))
                    closed = False
                    congestion = min(1.0, congestion + 0.60 * event.severity)
            else:
                speed_ratio = max(0.08, 1.0 - 0.72 * congestion)
                closed = False
            speed = edge.speed_limit_mps * speed_ratio
            travel_time = None if closed else edge.length_m / max(speed, 0.1)
            occupancy = min(1.0, max(0.0, 0.10 + 0.78 * congestion + rng.gauss(0, 0.01)))
            halting = max(0, int(round(8 * congestion + rng.gauss(0, 0.7))))
            rows.append({
                "timestamp_s": t,
                "edge_id": edge.edge_id,
                "parent_road_id": edge.parent_road_id,
                "true_speed_mps": round(speed, 5),
                "true_speed_ratio": round(speed_ratio, 5),
                "true_travel_time_s": None if travel_time is None else round(travel_time, 5),
                "congestion_index": round(congestion, 5),
                "occupancy": round(occupancy, 5),
                "halting_count": halting,
                "is_closed": closed,
                "active_event_id": event.event_id if event else "",
                "target_kind": "synthetic_field_travel_time",
            })
    return rows


def _generate_observations(scenario: Scenario, config: DynamicEpisodeConfig, events: list[DynamicEvent], truth_rows: list[dict]) -> list[dict]:
    rng = random.Random(config.seed + 307)
    out: list[dict] = []
    for row in truth_rows:
        t = int(row["timestamp_s"])
        # A deterministic missingness process, independent of event selection.
        missing = rng.random() < config.observation_missing_fraction
        if missing:
            out.append({
                "observation_time_s": t,
                "edge_id": row["edge_id"],
                "observed_speed_mps": "",
                "observed_travel_time_s": "",
                "occupancy": "",
                "halting_count": "",
                "observation_age_s": config.observation_delay_s,
                "missing": 1,
                "known_closed": 0,
                "target_available_at_s": "",
            })
            continue
        speed = float(row["true_speed_mps"])
        if not row["is_closed"]:
            speed *= max(0.0, 1.0 + rng.gauss(0, config.speed_noise_fraction))
        else:
            speed = 0.0
        visible_event = next((e for e in events if e.event_id == row["active_event_id"] and e.reveal_time_s <= t), None)
        known_closed = bool(visible_event and visible_event.event_type in {"closure", "scheduled_closure"})
        out.append({
            "observation_time_s": t,
            "edge_id": row["edge_id"],
            "observed_speed_mps": round(speed, 5),
            "observed_travel_time_s": row["true_travel_time_s"],
            "occupancy": round(min(1.0, max(0.0, float(row["occupancy"]) + rng.gauss(0, config.occupancy_noise))), 5),
            "halting_count": row["halting_count"],
            "observation_age_s": config.observation_delay_s,
            "missing": 0,
            "known_closed": int(known_closed),
            # This is intentionally a label availability timestamp, not a value
            # exposed to a forecasting model at issue time.
            "target_available_at_s": t + config.interval_s,
        })
    return out


def _daily_profile(t: int, duration: int, regime: str) -> float:
    # Smooth synthetic demand field. Values are deliberately dimensionless;
    # SUMO episodes later replace this with measured traffic states.
    x = t / max(duration, 1)
    if regime == "normal":
        return 0.20 + 0.08 * math.sin(2 * math.pi * x)
    if regime == "morning_peak":
        return 0.18 + 0.72 * math.exp(-((x - 0.35) / 0.16) ** 2)
    if regime == "evening_peak":
        return 0.18 + 0.72 * math.exp(-((x - 0.68) / 0.16) ** 2)
    return 0.18 + 0.70 * math.exp(-((x - 0.52) / 0.11) ** 2)


def _edge_load_factor(scenario: Scenario, edge_id: str) -> float:
    edge = next(e for e in scenario.edges if e.edge_id == edge_id)
    klass = edge.road_class.value
    class_factor = {"local": 0.55, "collector": 0.80, "arterial": 1.0}.get(klass, 0.70)
    return min(1.35, class_factor * (1.0 + 0.12 * edge.lane_count))


def _active_event(parent: str, t: int, events: Iterable[DynamicEvent]) -> DynamicEvent | None:
    for event in events:
        if parent in event.affected_parent_road_ids and event.effect_start_s <= t < event.effect_end_s:
            return event
    return None


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
