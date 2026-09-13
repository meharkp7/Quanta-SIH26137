"""Record 1-second TraCI samples into Step 11 observation / trajectory logs."""

from __future__ import annotations

from collections import defaultdict
from math import floor
from pathlib import Path
from typing import Any

from src.contracts.core_types import TargetKind
from src.contracts.scenario import Scenario
from src.data.causal_episodes import _write_csv, _write_json
from src.data.dynamic_episodes import DynamicEpisodeConfig
from src.sim.traci_adaptor import SimStepOutput


class CausalSumoLogger:
    """Aggregate TraCI steps into minute observations and entry/exit traces."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        interval_s: int = 60,
        observation_missing_fraction: float = 0.0,
    ) -> None:
        self.scenario = scenario
        self.interval_s = interval_s
        self.edge_ids = [edge.edge_id for edge in scenario.edges]
        self._edge_acc: dict[int, dict[str, list[tuple[float, float, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._truth_acc: dict[int, dict[str, list[tuple[float, float, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._vehicle_edge: dict[str, tuple[str, float]] = {}
        self.trajectories: list[dict] = []
        self.latest_time_s = 0.0
        self._known_closed_by_bucket: dict[int, set[str]] = defaultdict(set)

    #def on_step(self, step: SimStepOutput) -> None:
    def on_step(
            self,
            step: SimStepOutput,
            *,
            known_closed_edges: set[str] | None = None,
    ) -> None:
        known_closed_edges = known_closed_edges or set()
        t = float(step.sim_time_s)
        self.latest_time_s = t
        bucket = int(floor(t / self.interval_s) * self.interval_s)
        self._known_closed_by_bucket[bucket].update(known_closed_edges)

        by_edge = {item.edge_id: item for item in step.edges}
        for edge_id in self.edge_ids:
            item = by_edge.get(edge_id)
            if item is None:
                continue
            sample = (item.mean_speed_mps, item.occupancy, item.halting_count)
            self._edge_acc[bucket][edge_id].append(sample)
            self._truth_acc[bucket][edge_id].append(sample)

        for vehicle in step.vehicles:
            current = vehicle.edge_id
            previous = self._vehicle_edge.get(vehicle.vehicle_id)
            if previous is None:
                if current:
                    self._vehicle_edge[vehicle.vehicle_id] = (current, t)
                continue
            prev_edge, entry = previous
            if current != prev_edge:
                if prev_edge in self.edge_ids:
                    self.trajectories.append(
                        {
                            "trip_id": vehicle.vehicle_id,
                            "edge_id": prev_edge,
                            "entry_time_s": round(entry, 3),
                            "exit_time_s": round(t, 3),
                            "duration_s": round(t - entry, 3),
                            "entry_bucket_s": int(floor(entry / self.interval_s) * self.interval_s),
                            "target_kind": TargetKind.REALIZED_TRAVERSAL.value,
                        }
                    )
                if current:
                    self._vehicle_edge[vehicle.vehicle_id] = (current, t)
                else:
                    self._vehicle_edge.pop(vehicle.vehicle_id, None)

    def write(
        self,
        output_dir: Path,
        *,
        episode_id: str,
        config: DynamicEpisodeConfig,
        events: list[dict] | None = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        limits = {edge.edge_id: edge.speed_limit_mps for edge in self.scenario.edges}
        parents = {edge.edge_id: edge.parent_road_id for edge in self.scenario.edges}
        lengths = {edge.edge_id: edge.length_m for edge in self.scenario.edges}
        observations: list[dict] = []
        truth: list[dict] = []
        buckets = sorted(set(self._edge_acc) | set(self._truth_acc))
        for bucket in buckets:
            for edge_id in self.edge_ids:
                samples = self._edge_acc.get(bucket, {}).get(edge_id, [])
                if not samples:
                    observations.append(
                        {
                            "observation_time_s": bucket,
                            "edge_id": edge_id,
                            "observed_speed_mps": "",
                            "observed_travel_time_s": "",
                            "occupancy": "",
                            "halting_count": "",
                            "observation_age_s": 0,
                            "missing": 1,
                            "known_closed": 0,
                        }
                    )
                    truth.append(
                        {
                            "timestamp_s": bucket,
                            "edge_id": edge_id,
                            "parent_road_id": parents[edge_id],
                            "true_speed_mps": "",
                            "true_speed_ratio": "",
                            "true_travel_time_s": "",
                            "congestion_index": "",
                            "occupancy": "",
                            "halting_count": "",
                            "is_closed": False,
                            "active_event_id": "",
                        }
                    )
                    continue
                speed = sum(item[0] for item in samples) / len(samples)
                occupancy = sum(item[1] for item in samples) / len(samples)
                halt = int(round(sum(item[2] for item in samples) / len(samples)))
                # closed = speed <= 0.05
                # travel = "" if closed else round(lengths[edge_id] / max(speed, 0.1), 5)
                observed_closed = edge_id in self._known_closed_by_bucket.get(bucket, set())
                truth_closed = speed <= 0.05 or observed_closed
                travel = "" if observed_closed or speed <= 0.05 else round(
                    lengths[edge_id] / max(speed, 0.1), 5
                )
                ratio = speed / max(limits[edge_id], 1e-6)
                observations.append(
                    {
                        "observation_time_s": bucket,
                        "edge_id": edge_id,
                        # "observed_speed_mps": round(speed, 5),
                        # "observed_travel_time_s": travel,
                        "observed_speed_mps": None if observed_closed else round(speed, 5),
                        "observed_travel_time_s": None if observed_closed else travel,
                        "occupancy": round(min(1.0, max(0.0, occupancy)), 5),
                        "halting_count": halt,
                        "observation_age_s": 0,
                        "missing": 0,
                        #"known_closed": int(closed),
                        "known_closed": int(observed_closed),
                    }
                )
                truth.append(
                    {
                        "timestamp_s": bucket,
                        "edge_id": edge_id,
                        "parent_road_id": parents[edge_id],
                        "true_speed_mps": round(speed, 5),
                        "true_speed_ratio": round(ratio, 5),
                        "true_travel_time_s": (
                            "" if speed <= 0.05 else round(lengths[edge_id] / max(speed, 0.1), 5)
                        ),
                        "congestion_index": round(max(0.0, 1.0 - ratio), 5),
                        "occupancy": round(min(1.0, max(0.0, occupancy)), 5),
                        "halting_count": halt,
                        "is_closed": truth_closed,
                        "active_event_id": "",
                    }
                )
        _write_csv(output_dir / "observations.csv", observations)
        _write_csv(output_dir / "edge_truth.csv", truth)
        #_write_csv(output_dir / "trajectories.csv", self.trajectories)
        _write_csv(
            output_dir / "trajectories.csv",
            sorted(
                self.trajectories,
                key=lambda row: (
                    row["entry_bucket_s"],
                    row["entry_time_s"],
                    row["exit_time_s"],
                    row["trip_id"],
                    row["edge_id"],
                ),
            ),
        )
        _write_json(output_dir / "events.json", events or [])
        _write_json(
            output_dir / "sumo_logger.json",
            {
                "episode_id": episode_id,
                "backend": "sumo",
                "latest_time_s": self.latest_time_s,
                "interval_s": self.interval_s,
                "trajectory_count": len(self.trajectories),
            },
        )
