"""Causal completed-minute observations and measured edge traversals."""
from collections import defaultdict
from math import ceil, floor, isfinite
from pathlib import Path
from src.contracts.core_types import TargetKind
from src.data.causal_episodes import _write_csv, _write_json

class CausalSumoLogger:
    def __init__(self, scenario, *, interval_s=60, observation_missing_fraction=0.0,
                 closed_edge_speed_zero: bool = False):
        self.scenario = scenario
        self.interval_s = interval_s
        self.edge_ids = [e.edge_id for e in scenario.edges]
        self._samples = defaultdict(lambda: defaultdict(list))
        self._closed = {}
        self._vehicle_edge = {}
        self.trajectories = []
        self.latest_time_s = 0.0
        # corpus_v2: emit closed-edge buckets as non-missing speed-0 rows
        # (see DynamicEpisodeConfig.emit_closed_edge_speed_zero). Default False
        # preserves legacy behavior (closed edges with no samples are missing).
        self._closed_edge_speed_zero = bool(closed_edge_speed_zero)

    def _exit(self, vid, time_s):
        previous = self._vehicle_edge.pop(vid, None)
        if previous is None:
            return
        edge, entry = previous
        if edge in self.edge_ids and time_s > entry:
            self.trajectories.append(dict(trip_id=vid, edge_id=edge, entry_time_s=entry,
                exit_time_s=time_s, duration_s=time_s-entry,
                entry_bucket_s=int(floor(entry/self.interval_s)*self.interval_s),
                target_kind=TargetKind.REALIZED_TRAVERSAL.value))

    def on_step(self, step, *, known_closed_edges=None):
        t = float(step.sim_time_s)
        self.latest_time_s = t
        # Tick 1..60 is published at 60, never at 0. Partial final buckets are omitted.
        end = int(ceil(t/self.interval_s)*self.interval_s)
        self._closed[end] = set(known_closed_edges or ())
        self._samples[end]  # Ensure wholly unobserved minutes are represented.
        for edge in step.edges:
            # SUMO returns free-flow speed on an empty lane: this is not a measurement.
            if edge.vehicle_count > 0 and isfinite(edge.mean_speed_mps) and edge.mean_speed_mps >= 0:
                self._samples[end][edge.edge_id].append((t, edge.mean_speed_mps, edge.occupancy, edge.halting_count))
        for vehicle in step.vehicles:
            previous = self._vehicle_edge.get(vehicle.vehicle_id)
            if previous is not None and previous[0] != vehicle.edge_id:
                self._exit(vehicle.vehicle_id, t)
            if vehicle.edge_id and vehicle.vehicle_id not in self._vehicle_edge:
                self._vehicle_edge[vehicle.vehicle_id] = (vehicle.edge_id, t)
        for vid in step.arrived_vehicle_ids:
            self._exit(vid, t)

    def write(self, output_dir: Path, *, episode_id, config, events=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        observations, truth = [], []
        last_seen = {}
        for end in sorted(self._samples):
            if end > self.latest_time_s:
                continue
            for edge in self.scenario.edges:
                samples = self._samples[end].get(edge.edge_id, [])
                closed = edge.edge_id in self._closed.get(end, ())
                speed = sum(s[1] for s in samples)/len(samples) if samples else None
                occupancy = sum(s[2] for s in samples)/len(samples) if samples else None
                halt = sum(s[3] for s in samples)/len(samples) if samples else None
                if samples:
                    last_seen[edge.edge_id] = samples[-1][0]
                age = end-last_seen.get(edge.edge_id, 0)
                travel = edge.length_m/speed if speed is not None and speed > 0.05 and not closed else None
                # corpus_v2 closed-edge semantics: zero flow on a closed road is
                # measured intervention truth at minute-bucket granularity, so a
                # closed bucket is recorded as observed speed-0 (the downstream
                # speed_proxy label matures to value 0.0 at this bucket).
                # Occupancy/halting/travel stay empty when no vehicle was
                # sampled -- they are sensor measurements, not intervention
                # truth, and the loader treats empty fields as masked features.
                # Legacy behavior (flag off): closed buckets keep the measured
                # speed (None when unsampled) and missing=int(not samples).
                if closed and self._closed_edge_speed_zero:
                    row_missing, row_speed = 0, 0.0
                else:
                    row_missing = int(not samples)
                    row_speed = speed if not closed else None
                observations.append(dict(observation_time_s=end, edge_id=edge.edge_id,
                    observed_speed_mps=row_speed, observed_travel_time_s=travel,
                    occupancy=occupancy, halting_count=halt, observation_age_s=age,
                    missing=row_missing, known_closed=int(closed)))
                truth.append(dict(timestamp_s=end, edge_id=edge.edge_id,
                    parent_road_id=edge.parent_road_id, true_speed_mps=speed,
                    true_speed_ratio=speed/edge.speed_limit_mps if speed is not None else None,
                    true_travel_time_s=travel, congestion_index=None,
                    occupancy=occupancy, halting_count=halt, is_closed=closed, active_event_id=""))
        _write_csv(output_dir / "observations.csv", observations)
        _write_csv(output_dir / "edge_truth.csv", truth)
        _write_csv(output_dir / "trajectories.csv", self.trajectories)
        _write_json(output_dir / "events.json", events or [])
