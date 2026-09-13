"""Step 11 — causal traffic episodes and matured forecasting labels.

Six pilot episodes are generated with independent RNG streams for traffic,
incidents, observation noise and background trajectories.  Observations never
contain future event times or unmatured labels.  Issued forecasts are stored
separately from later-available labels.

The default backend builds a measured-style field plus integrated vehicle
traversals so the loader can be developed without SUMO.  When SUMO is
available, ``backend="sumo"`` records TraCI speeds and edge entry/exit times
and writes the same files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Callable, Iterable, Mapping, Sequence

from src.contracts.core_types import TargetKind
from src.contracts.forecast import Forecast
from src.contracts.scenario import Scenario
from src.data.causal_labels import (
    MatureLabel,
    mature_speed_proxy_label,
    mature_traversal_label,
    minute_bucket,
)
from src.data.dynamic_episodes import (
    DynamicEpisodeConfig,
    DynamicEvent,
    _active_event,
    _daily_profile,
    _edge_load_factor,
    _generate_events,
)

HORIZONS_S = (300, 600, 900)
HISTORY_MINUTES = 12
PILOT_SPECS = (
    {"episode_id": "ep-001", "regime": "normal", "event_type": None, "split": "train"},
    {"episode_id": "ep-002", "regime": "morning_peak", "event_type": "incident", "split": "train"},
    {"episode_id": "ep-003", "regime": "evening_peak", "event_type": None, "split": "train"},
    {"episode_id": "ep-004", "regime": "corridor_surge", "event_type": "incident", "split": "train"},
    {"episode_id": "ep-005", "regime": "morning_peak", "event_type": "closure", "split": "validation"},
    {"episode_id": "ep-006", "regime": "normal", "event_type": "multi_disruption", "split": "test"},
)


@dataclass(frozen=True)
class StreamSeeds:
    """Independent environment RNG streams. One global seed is not enough."""

    traffic: int
    incident: int
    observation: int
    trajectory: int


def stream_seeds(base: int) -> StreamSeeds:
    return StreamSeeds(
        traffic=int(base) + 211,
        incident=int(base) + 101,
        observation=int(base) + 307,
        trajectory=int(base) + 419,
    )


@dataclass
class CausalEpisodeResult:
    episode_dir: Path
    episode_id: str
    split: str
    wall_clock_s: float
    disk_bytes: int
    label_coverage: dict


def generate_causal_pilot(
    scenario: Scenario,
    output_dir: str | Path,
    *,
    duration_s: int = 2400,
    interval_s: int = 60,
    warmup_s: int = 300,
    base_seed: int = 26137,
    backend: str = "causal_field_v1",
    on_sumo_episode: Callable[..., Path] | None = None,
) -> dict:
    """Write six split episodes and a coverage report. Split happens first."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    episodes_dir = root / "episodes"
    split = {
        "train": [spec["episode_id"] for spec in PILOT_SPECS if spec["split"] == "train"],
        "validation": [spec["episode_id"] for spec in PILOT_SPECS if spec["split"] == "validation"],
        "test": [spec["episode_id"] for spec in PILOT_SPECS if spec["split"] == "test"],
        "rule": "split by episode before overlapping windows",
        "scenario_id": scenario.scenario_id,
        "graph_version": scenario.graph_version,
    }
    _write_json(root / "split_manifest.json", split)

    results: list[CausalEpisodeResult] = []
    for index, spec in enumerate(PILOT_SPECS):
        seed = base_seed + 17 * index
        config = DynamicEpisodeConfig(
            duration_s=duration_s,
            interval_s=interval_s,
            warmup_s=warmup_s,
            regime=spec["regime"],
            event_type=spec["event_type"],
            event_count=1 if spec["event_type"] else 0,
            event_duration_s=240,
            reveal_lead_s=60,
            observation_missing_fraction=0.03,
            seed=seed,
            backend=backend,
        )
        episode_dir = episodes_dir / spec["episode_id"]
        if backend == "sumo" and on_sumo_episode is not None:
            events = _generate_events(scenario, config, spec["episode_id"])
            on_sumo_episode(
                scenario=scenario,
                output_dir=episode_dir,
                config=config,
                episode_id=spec["episode_id"],
                split=spec["split"],
                events=events,
            )
            result = _finalize_existing_episode(
                scenario, episode_dir, spec["episode_id"], spec["split"], config
            )
        else:
            result = generate_causal_episode(
                scenario,
                episode_dir,
                config=config,
                episode_id=spec["episode_id"],
                split=spec["split"],
            )
        results.append(result)

    coverage = _pilot_coverage(results, time.perf_counter() - started)
    _write_json(root / "coverage.json", coverage)

    # Runtime timing is intentionally excluded from the reproducibility hash.
    # Episode artifacts themselves must remain byte-stable for a fixed seed.
    deterministic = {}
    for result in results:
        for artifact in sorted(result.episode_dir.iterdir()):
            if artifact.is_file():
                deterministic[str(artifact.relative_to(root))] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    deterministic["split_manifest.json"] = hashlib.sha256((root / "split_manifest.json").read_bytes()).hexdigest()
    _write_json(root / "reproducibility.json", {
        "schema_version": "reproducibility-1.0",
        "seed": base_seed,
        "fixed_seed_artifacts_byte_stable": True,
        "hash_algorithm": "sha256",
        "artifact_sha256": deterministic,
        "excluded_from_hash": ["coverage.json"],
    })
    return coverage


def generate_causal_episode(
    scenario: Scenario,
    output_dir: str | Path,
    *,
    config: DynamicEpisodeConfig | None = None,
    episode_id: str,
    split: str,
) -> CausalEpisodeResult:
    started = time.perf_counter()
    config = config or DynamicEpisodeConfig()
    seeds = stream_seeds(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    events = _generate_events(
        scenario,
        DynamicEpisodeConfig(
            **{**config.__dict__, "seed": seeds.incident - 101}
        ),
        episode_id,
    )
    truth = _truth_rows(scenario, config, events, seeds.traffic)
    observations = _observation_rows(config, events, truth, seeds.observation)
    trajectories = _simulate_traversals(scenario, config, truth, events, seeds.trajectory)
    labels = _build_labels(
        episode_id=episode_id,
        scenario=scenario,
        config=config,
        observations=observations,
        trajectories=trajectories,
    )
    forecasts = _issue_forecasts(
        scenario=scenario,
        episode_id=episode_id,
        config=config,
        observations=observations,
    )

    _write_csv(output / "edge_truth.csv", truth)
    _write_csv(output / "observations.csv", observations)
    _write_csv(output / "trajectories.csv", trajectories)
    _write_jsonl(output / "labels.jsonl", [asdict(item) for item in labels])
    _write_jsonl(output / "issued_forecasts.jsonl", forecasts)
    _write_json(output / "events.json", [asdict(event) for event in events])
    coverage = _episode_coverage(labels, observations, trajectories)
    _write_json(
        output / "episode_manifest.json",
        {
            "schema_version": "causal-1.0",
            "episode_id": episode_id,
            "scenario_id": scenario.scenario_id,
            "graph_version": scenario.graph_version,
            "split": split,
            "backend": config.backend,
            "interval_s": config.interval_s,
            "duration_s": config.duration_s,
            "warmup_s": config.warmup_s,
            "regime": config.regime,
            "seeds": asdict(seeds),
            "horizons_s": list(HORIZONS_S),
            "history_minutes": HISTORY_MINUTES,
            "target_kinds": [
                TargetKind.SPEED_PROXY.value,
                TargetKind.REALIZED_TRAVERSAL.value,
            ],
            "causal_visibility": {
                "future_effect_times_hidden": True,
                "future_labels_hidden": True,
                "issued_forecasts_separate": True,
                "sparse_edges_are_missing": True,
            },
            "coverage": coverage,
        },
    )
    disk = _directory_size(output)
    return CausalEpisodeResult(
        episode_dir=output,
        episode_id=episode_id,
        split=split,
        wall_clock_s=time.perf_counter() - started,
        disk_bytes=disk,
        label_coverage=coverage,
    )


def load_observation_index(rows: Sequence[Mapping]) -> dict[tuple[str, int], dict]:
    index: dict[tuple[str, int], dict] = {}
    for row in rows:
        time_s = int(row["observation_time_s"])
        index[(str(row["edge_id"]), time_s)] = dict(row)
    return index


def _truth_rows(
    scenario: Scenario,
    config: DynamicEpisodeConfig,
    events: list[DynamicEvent],
    traffic_seed: int,
) -> list[dict]:
    rng = random.Random(traffic_seed)
    rows: list[dict] = []
    for t in range(0, config.duration_s + 1, config.interval_s):
        phase = _daily_profile(t, config.duration_s, config.regime)
        corridor = 0.0
        if config.regime == "corridor_surge":
            corridor = 0.35 * max(0.0, 1.0 - abs((t / max(config.duration_s, 1)) - 0.5) / 0.18)
        for edge in scenario.edges:
            event = _active_event(edge.parent_road_id, t, events)
            recovering = _recovering(edge.parent_road_id, t, events)
            load = _edge_load_factor(scenario, edge.edge_id)
            congestion = min(0.94, max(0.0, 0.10 + 0.58 * phase * load + corridor))
            if recovering:
                congestion = max(0.08, congestion * 0.65)
            closed = False
            if event and event.event_type in {"closure", "scheduled_closure"}:
                speed_ratio = 0.0
                closed = True
                congestion = 1.0
            elif event:
                speed_ratio = max(0.08, 1.0 - min(0.88, congestion + 0.55 * event.severity))
                congestion = min(1.0, congestion + 0.55 * event.severity)
            else:
                speed_ratio = max(0.08, 1.0 - 0.72 * congestion)
            speed = edge.speed_limit_mps * speed_ratio
            travel = None if closed else edge.length_m / max(speed, 0.1)
            occupancy = min(1.0, max(0.0, 0.10 + 0.78 * congestion + rng.gauss(0, 0.01)))
            rows.append(
                {
                    "timestamp_s": t,
                    "edge_id": edge.edge_id,
                    "parent_road_id": edge.parent_road_id,
                    "true_speed_mps": round(speed, 5),
                    "true_speed_ratio": round(speed_ratio, 5),
                    "true_travel_time_s": None if travel is None else round(travel, 5),
                    "congestion_index": round(congestion, 5),
                    "occupancy": round(occupancy, 5),
                    "halting_count": max(0, int(round(8 * congestion + rng.gauss(0, 0.6)))),
                    "is_closed": closed,
                    "active_event_id": event.event_id if event else "",
                }
            )
    return rows


def _recovering(parent: str, t: int, events: Iterable[DynamicEvent]) -> bool:
    for event in events:
        if parent not in event.affected_parent_road_ids:
            continue
        if event.effect_end_s <= t < event.effect_end_s + 180:
            return True
    return False


def _observation_rows(
    config: DynamicEpisodeConfig,
    events: list[DynamicEvent],
    truth_rows: list[dict],
    observation_seed: int,
) -> list[dict]:
    rng = random.Random(observation_seed)
    out: list[dict] = []
    for row in truth_rows:
        t = int(row["timestamp_s"])
        missing = rng.random() < config.observation_missing_fraction
        if missing:
            out.append(
                {
                    "observation_time_s": t,
                    "edge_id": row["edge_id"],
                    "observed_speed_mps": "",
                    "observed_travel_time_s": "",
                    "occupancy": "",
                    "halting_count": "",
                    "observation_age_s": config.observation_delay_s,
                    "missing": 1,
                    "known_closed": 0,
                }
            )
            continue
        speed = float(row["true_speed_mps"])
        if row["is_closed"]:
            speed = 0.0
        else:
            speed *= max(0.0, 1.0 + rng.gauss(0, config.speed_noise_fraction))
        visible = next(
            (
                event
                for event in events
                if event.event_id == row["active_event_id"]
                and event.reveal_time_s <= t
            ),
            None,
        )
        known_closed = bool(
            visible and visible.event_type in {"closure", "scheduled_closure"}
        )
        travel = row["true_travel_time_s"]
        out.append(
            {
                "observation_time_s": t,
                "edge_id": row["edge_id"],
                "observed_speed_mps": round(speed, 5),
                "observed_travel_time_s": travel if travel not in (None, "") else "",
                "occupancy": round(
                    min(1.0, max(0.0, float(row["occupancy"]) + rng.gauss(0, config.occupancy_noise))),
                    5,
                ),
                "halting_count": row["halting_count"],
                "observation_age_s": config.observation_delay_s,
                "missing": 0,
                "known_closed": int(known_closed),
            }
        )
    return out


def _simulate_traversals(
    scenario: Scenario,
    config: DynamicEpisodeConfig,
    truth_rows: list[dict],
    events: list[DynamicEvent],
    trajectory_seed: int,
) -> list[dict]:
    """Integrate background + delivery probes through the speed field."""

    rng = random.Random(trajectory_seed)
    speed_at = {
        (row["edge_id"], int(row["timestamp_s"])): row
        for row in truth_rows
    }
    outgoing: dict[str, list] = {}
    for edge in scenario.edges:
        outgoing.setdefault(edge.from_node, []).append(edge)
    nodes = [node.node_id for node in scenario.nodes]
    paths: list[list] = []
    if scenario.requests and scenario.fleet:
        depot = scenario.fleet[0].depot_node_id
        for job in scenario.requests:
            path = _walk(outgoing, depot, job.access_node_id, rng)
            if path:
                paths.append(path)
    for _ in range(max(4, len(scenario.edges) // 2)):
        src, dst = rng.sample(nodes, 2)
        path = _walk(outgoing, src, dst, rng)
        if path:
            paths.append(path)

    rows: list[dict] = []
    trip_id = 0
    launch_times = list(range(config.warmup_s, max(config.warmup_s + 1, config.duration_s - 120), 120))
    if not launch_times:
        launch_times = [float(config.warmup_s)]
    for path in paths:
        for launch in launch_times:
            t = float(launch) + rng.random() * 20.0
            trip_id += 1
            for edge in path:
                if t >= config.duration_s:
                    break
                bucket = minute_bucket(t, config.interval_s)
                state = speed_at.get((edge.edge_id, bucket))
                if state is None or state["is_closed"]:
                    break
                speed = max(0.1, float(state["true_speed_mps"]))
                duration = float(edge.length_m) / speed
                exit_time = t + duration
                rows.append(
                    {
                        "trip_id": f"T{trip_id:04d}",
                        "edge_id": edge.edge_id,
                        "entry_time_s": round(t, 3),
                        "exit_time_s": round(exit_time, 3),
                        "duration_s": round(duration, 3),
                        "entry_bucket_s": bucket,
                        "target_kind": TargetKind.REALIZED_TRAVERSAL.value,
                    }
                )
                t = exit_time
    return rows


def _walk(outgoing, source: str, target: str, rng: random.Random):
    if source == target:
        return []
    frontier = [(source, [])]
    seen = {source}
    while frontier:
        node, path = frontier.pop(0)
        neighbors = list(outgoing.get(node, ()))
        rng.shuffle(neighbors)
        for edge in neighbors:
            if edge.to_node in seen:
                continue
            nxt = path + [edge]
            if edge.to_node == target:
                return nxt
            seen.add(edge.to_node)
            frontier.append((edge.to_node, nxt))
    return []


def _issue_times(config: DynamicEpisodeConfig) -> list[int]:
    first = config.warmup_s + HISTORY_MINUTES * config.interval_s
    last = config.duration_s - max(HORIZONS_S)
    if last < first:
        return []
    return list(range(first, last + 1, config.interval_s))


def _build_labels(
    *,
    episode_id: str,
    scenario: Scenario,
    config: DynamicEpisodeConfig,
    observations: list[dict],
    trajectories: list[dict],
) -> list[MatureLabel]:
    index = load_observation_index(observations)
    labels: list[MatureLabel] = []
    for issue in _issue_times(config):
        for horizon in HORIZONS_S:
            target = issue + horizon
            for edge in scenario.edges:
                labels.append(
                    mature_speed_proxy_label(
                        episode_id=episode_id,
                        edge_id=edge.edge_id,
                        issue_time_s=issue,
                        target_time_s=target,
                        observation_by_bucket=index,
                        interval_s=config.interval_s,
                    )
                )
                labels.append(
                    mature_traversal_label(
                        episode_id=episode_id,
                        edge_id=edge.edge_id,
                        issue_time_s=issue,
                        target_time_s=target,
                        traversals=trajectories,
                        interval_s=config.interval_s,
                    )
                )
    return labels


def _issue_forecasts(
    *,
    scenario: Scenario,
    episode_id: str,
    config: DynamicEpisodeConfig,
    observations: list[dict],
) -> list[dict]:
    """Persistence forecasts issued at t. Labels are not stored here."""

    index = load_observation_index(observations)
    edge_ids = tuple(edge.edge_id for edge in scenario.edges)
    # Persistence means the latest *available* observation at or before the
    # issue time, not merely an exact timestamp. This matters when an edge has
    # a missing sample at the issue bucket.
    history: dict[str, list[tuple[int, float]]] = {}
    for row in observations:
        if bool(int(row.get("missing") or 0)) or row.get("observed_speed_mps") in (None, ""):
            continue
        history.setdefault(str(row["edge_id"]), []).append(
            (int(row["observation_time_s"]), float(row["observed_speed_mps"]))
        )
    for values in history.values():
        values.sort()

    def latest_valid(edge_id: str, issue_time: int) -> float | None:
        values = history.get(edge_id, ())
        candidate = None
        for timestamp, value in values:
            if timestamp > issue_time:
                break
            candidate = value
        return candidate

    records = []
    for issue in _issue_times(config):
        targets = tuple(issue + horizon for horizon in HORIZONS_S)
        prediction = []
        valid = []
        for edge_id in edge_ids:
            value = latest_valid(edge_id, issue)
            row = tuple(value for _ in targets)
            prediction.append(row)
            valid.append(tuple(value is not None for _ in targets))
        forecast = Forecast(
            scenario_id=scenario.scenario_id,
            episode_id=episode_id,
            forecast_version="persistence-v0",
            issued_at_s=float(issue),
            target_times_s=tuple(float(t) for t in targets),
            edge_ids=edge_ids,
            prediction=tuple(prediction),
            valid_mask=tuple(valid),
            target_kind=TargetKind.SPEED_PROXY,
            target_unit="m/s",
            model_version="persistence-v0",
        )
        records.append(forecast.model_dump(mode="json"))
    return records


def _episode_coverage(
    labels: Sequence[MatureLabel],
    observations: Sequence[dict],
    trajectories: Sequence[dict],
) -> dict:
    def _stats(kind: str) -> dict:
        subset = [item for item in labels if item.target_kind == kind]
        valid = sum(1 for item in subset if not item.missing and item.value is not None)
        return {
            "count": len(subset),
            "valid": valid,
            "missing": len(subset) - valid,
            "coverage": (valid / len(subset)) if subset else 0.0,
            "all_have_availability_when_valid": all(
                item.available_at_s is not None
                for item in subset
                if not item.missing and item.value is not None
            ),
        }

    return {
        "observation_rows": len(observations),
        "trajectory_rows": len(trajectories),
        "issued_forecast_slots": len({(item.issue_time_s) for item in labels}),
        TargetKind.SPEED_PROXY.value: _stats(TargetKind.SPEED_PROXY.value),
        TargetKind.REALIZED_TRAVERSAL.value: _stats(
            TargetKind.REALIZED_TRAVERSAL.value
        ),
    }


def _pilot_coverage(results: Sequence[CausalEpisodeResult], wall_s: float) -> dict:
    return {
        "episodes": len(results),
        "wall_clock_s": round(wall_s, 3),
        "disk_bytes": sum(item.disk_bytes for item in results),
        "per_episode": [
            {
                "episode_id": item.episode_id,
                "split": item.split,
                "wall_clock_s": round(item.wall_clock_s, 3),
                "disk_bytes": item.disk_bytes,
                "coverage": item.label_coverage,
            }
            for item in results
        ],
        "split_before_windows": True,
    }


def _finalize_existing_episode(
    scenario: Scenario,
    episode_dir: Path,
    episode_id: str,
    split: str,
    config: DynamicEpisodeConfig,
) -> CausalEpisodeResult:
    observations = list(csv.DictReader((episode_dir / "observations.csv").open(encoding="utf-8")))
    trajectories = list(csv.DictReader((episode_dir / "trajectories.csv").open(encoding="utf-8")))
    labels = _build_labels(
        episode_id=episode_id,
        scenario=scenario,
        config=config,
        observations=observations,
        trajectories=trajectories,
    )
    forecasts = _issue_forecasts(
        scenario=scenario,
        episode_id=episode_id,
        config=config,
        observations=observations,
    )
    _write_jsonl(episode_dir / "labels.jsonl", [asdict(item) for item in labels])
    _write_jsonl(episode_dir / "issued_forecasts.jsonl", forecasts)

    # SUMO-backed episodes are finalized from observed TraCI measurements.
    # Recreate the episode manifest here so the generated artifacts have the
    # same contract as the synthetic backend and remain auditable/replayable.
    _write_json(
        episode_dir / "episode_manifest.json",
        {
            "schema_version": "causal-1.0",
            "episode_id": episode_id,
            "scenario_id": scenario.scenario_id,
            "graph_version": scenario.graph_version,
            "split": split,
            "backend": config.backend,
            "interval_s": config.interval_s,
            "duration_s": config.duration_s,
            "warmup_s": config.warmup_s,
            "regime": config.regime,
            "seed": config.seed,
            "seeds": asdict(stream_seeds(config.seed)),
            "horizons_s": list(HORIZONS_S),
            "history_minutes": HISTORY_MINUTES,
            "target_kinds": [
                TargetKind.SPEED_PROXY.value,
                TargetKind.REALIZED_TRAVERSAL.value,
            ],
            "causal_visibility": {
                "future_effect_times_hidden": True,
                "future_labels_hidden": True,
                "issued_forecasts_separate": True,
                "sparse_edges_are_missing": True,
                "labels_derived_from_observed_sumo_outcomes": True,
            },
            "events_source": "declared_event_tape_applied_in_sumo",
            "trajectory_source": "sumo_traci_vehicle_edge_transitions",
        },
    )

    coverage = _episode_coverage(labels, observations, trajectories)
    return CausalEpisodeResult(
        episode_dir=episode_dir,
        episode_id=episode_id,
        split=split,
        wall_clock_s=0.0,
        disk_bytes=_directory_size(episode_dir),
        label_coverage=coverage,
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
