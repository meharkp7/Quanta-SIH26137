"""
Step 16.6 — Causal deadline and service-risk representation.

Builds request-level and vehicle-level operational risk features from the
currently visible Observation. No future simulator outcomes are consumed.

Risk is expressed relative to the current observation clock:
- release readiness
- remaining service-window slack
- normalized urgency
- onboard/remaining workload
- deadline pressure

The representation is model-independent and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

JOB_FEATURE_NAMES: tuple[str, ...] = (
    "released",
    "time_until_release_normalized",
    "time_until_earliest_service_normalized",
    "time_until_latest_service_normalized",
    "service_window_width_normalized",
    "urgency",
    "demand_normalized",
    "service_duration_normalized",
    "status_pending",
    "status_in_progress",
    "status_served",
    "status_cancelled",
)

VEHICLE_RISK_FEATURE_NAMES: tuple[str, ...] = (
    "remaining_load_normalized",
    "onboard_job_count_normalized",
    "committed_job_count_normalized",
    "distance_remaining_normalized",
    "deadline_pressure",
    "workload_pressure",
)

_STATUS_COLUMNS = {
    "pending": "status_pending",
    "in_progress": "status_in_progress",
    "served": "status_served",
    "cancelled": "status_cancelled",
}


@dataclass(frozen=True)
class ServiceRiskState:
    """Immutable causal service-risk representation."""

    job_ids: tuple[str, ...]
    job_features: np.ndarray
    vehicle_ids: tuple[str, ...]
    vehicle_features: np.ndarray
    job_id_to_index: Mapping[str, int]
    vehicle_id_to_index: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.job_features.shape != (
            len(self.job_ids),
            len(JOB_FEATURE_NAMES),
        ):
            raise ValueError("job_features shape is inconsistent with job_ids")

        if self.vehicle_features.shape != (
            len(self.vehicle_ids),
            len(VEHICLE_RISK_FEATURE_NAMES),
        ):
            raise ValueError("vehicle_features shape is inconsistent with vehicle_ids")

        if not np.all(np.isfinite(self.job_features)):
            raise ValueError("job_features contains NaN or infinite values")
        if not np.all(np.isfinite(self.vehicle_features)):
            raise ValueError("vehicle_features contains NaN or infinite values")

        if dict(self.job_id_to_index) != {
            value: i for i, value in enumerate(self.job_ids)
        }:
            raise ValueError("job_id_to_index is inconsistent")

        if dict(self.vehicle_id_to_index) != {
            value: i for i, value in enumerate(self.vehicle_ids)
        }:
            raise ValueError("vehicle_id_to_index is inconsistent")

        self.job_features.setflags(write=False)
        self.vehicle_features.setflags(write=False)

    @property
    def job_count(self) -> int:
        return len(self.job_ids)

    @property
    def vehicle_count(self) -> int:
        return len(self.vehicle_ids)


class ServiceRiskBuilder:
    """Build causal deadline/service-risk features from one Observation."""

    def __init__(
        self,
        scenario,
        *,
        time_normalization_s: float = 300.0,
        demand_normalization: float | None = None,
        distance_normalization_m: float | None = None,
    ) -> None:
        if time_normalization_s <= 0:
            raise ValueError("time_normalization_s must be positive")

        self.scenario = scenario
        self.time_normalization_s = float(time_normalization_s)

        visible_demands = [float(r.demand) for r in scenario.requests]
        self.demand_normalization = float(
            demand_normalization
            if demand_normalization is not None
            else max(max(visible_demands, default=1.0), 1.0)
        )
        self.distance_normalization_m = float(
            distance_normalization_m
            if distance_normalization_m is not None
            else max(
                [float(e.length_m) for e in scenario.edges] or [1.0]
            )
        )
        if self.demand_normalization <= 0 or self.distance_normalization_m <= 0:
            raise ValueError("normalization references must be positive")

    def build(self, observation) -> ServiceRiskState:
        if str(observation.scenario_id) != str(self.scenario.scenario_id):
            raise ValueError("observation scenario_id does not match Scenario")

        now = float(observation.observation_time_s)
        jobs = tuple(sorted(observation.visible_jobs, key=lambda j: str(j.request_id)))
        fleet = tuple(sorted(observation.fleet, key=lambda v: str(v.vehicle_id)))

        request_by_id = {
            str(request.request_id): request
            for request in self.scenario.requests
        }

        job_features = np.zeros(
            (len(jobs), len(JOB_FEATURE_NAMES)), dtype=np.float32
        )

        for i, job in enumerate(jobs):
            request = request_by_id.get(str(job.request_id))
            if request is None:
                raise ValueError(
                    f"visible job {job.request_id!r} does not exist in Scenario"
                )

            release = float(job.release_s)
            earliest = float(job.earliest_service_start_s)
            latest = float(job.latest_service_start_s)

            job_features[i, JOB_FEATURE_NAMES.index("released")] = float(release <= now)
            job_features[i, JOB_FEATURE_NAMES.index("time_until_release_normalized")] = self._future_time(
                release, now
            )
            job_features[i, JOB_FEATURE_NAMES.index("time_until_earliest_service_normalized")] = self._future_time(
                earliest, now
            )
            job_features[i, JOB_FEATURE_NAMES.index("time_until_latest_service_normalized")] = self._future_time(
                latest, now
            )

            width = max(0.0, latest - earliest)
            job_features[i, JOB_FEATURE_NAMES.index("service_window_width_normalized")] = self._clip_time(width)

            # Urgency rises as the latest service time approaches or passes.
            # It is causal: it depends only on the known deadline and current clock.
            deadline_remaining = latest - now
            job_features[i, JOB_FEATURE_NAMES.index("urgency")] = self._urgency(
                deadline_remaining
            )

            job_features[i, JOB_FEATURE_NAMES.index("demand_normalized")] = np.clip(
                float(job.demand) / self.demand_normalization, 0.0, 1.0
            )
            job_features[i, JOB_FEATURE_NAMES.index("service_duration_normalized")] = self._clip_time(
                float(job.service_duration_s)
            )

            status = str(job.status).lower()
            status_column = _STATUS_COLUMNS.get(status)
            if status_column is not None:
                job_features[i, JOB_FEATURE_NAMES.index(status_column)] = 1.0

        vehicle_features = np.zeros(
            (len(fleet), len(VEHICLE_RISK_FEATURE_NAMES)), dtype=np.float32
        )

        for i, vehicle in enumerate(fleet):
            vehicle_features[i, VEHICLE_RISK_FEATURE_NAMES.index("remaining_load_normalized")] = (
                np.clip(
                    float(vehicle.remaining_load) / self._vehicle_capacity(vehicle.vehicle_id),
                    0.0,
                    1.0,
                )
            )
            vehicle_features[i, VEHICLE_RISK_FEATURE_NAMES.index("onboard_job_count_normalized")] = (
                np.clip(len(vehicle.onboard_request_ids) / max(len(jobs), 1), 0.0, 1.0)
            )
            vehicle_features[i, VEHICLE_RISK_FEATURE_NAMES.index("committed_job_count_normalized")] = (
                np.clip(len(vehicle.committed_request_ids) / max(len(jobs), 1), 0.0, 1.0)
            )
            vehicle_features[i, VEHICLE_RISK_FEATURE_NAMES.index("distance_remaining_normalized")] = (
                np.clip(
                    float(vehicle.distance_remaining_m) / self.distance_normalization_m,
                    0.0,
                    10.0,
                )
            )

            assigned = set(str(x) for x in vehicle.committed_request_ids)
            pressures = []
            for job in jobs:
                if str(job.request_id) in assigned:
                    pressures.append(self._urgency(float(job.latest_service_start_s) - now))
            vehicle_features[i, VEHICLE_RISK_FEATURE_NAMES.index("deadline_pressure")] = (
                max(pressures, default=0.0)
            )
            vehicle_features[i, VEHICLE_RISK_FEATURE_NAMES.index("workload_pressure")] = np.clip(
                (
                    len(vehicle.onboard_request_ids)
                    + len(vehicle.committed_request_ids)
                )
                / max(len(jobs) * 2, 1),
                0.0,
                1.0,
            )

        return ServiceRiskState(
            job_ids=tuple(str(j.request_id) for j in jobs),
            job_features=job_features,
            vehicle_ids=tuple(str(v.vehicle_id) for v in fleet),
            vehicle_features=vehicle_features,
            job_id_to_index={
                str(j.request_id): i for i, j in enumerate(jobs)
            },
            vehicle_id_to_index={
                str(v.vehicle_id): i for i, v in enumerate(fleet)
            },
        )

    def _vehicle_capacity(self, vehicle_id: object) -> float:
        for vehicle in self.scenario.fleet:
            if str(vehicle.vehicle_id) == str(vehicle_id):
                return max(float(vehicle.capacity), 1e-9)
        raise ValueError(f"vehicle {vehicle_id!r} does not exist in Scenario")

    def _future_time(self, target: float, now: float) -> float:
        return self._clip_time(max(0.0, target - now))

    def _clip_time(self, seconds: float) -> float:
        return float(
            np.clip(seconds / self.time_normalization_s, 0.0, 10.0)
        )

    def _urgency(self, deadline_remaining: float) -> float:
        if deadline_remaining <= 0:
            return 1.0
        # 1 at deadline, smoothly decreasing toward 0 as slack grows.
        return float(
            np.clip(
                1.0 - deadline_remaining / self.time_normalization_s,
                0.0,
                1.0,
            )
        )
