from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from ._base import Contract
from .core_types import (
    CustomerId,
    EpisodeId,
    GraphVersion,
    RoadEdgeId,
    RoadNodeId,
    TimeS,
    VehicleId,
)


class EdgeObservation(Contract):
    edge_id: RoadEdgeId

    observed_speed_mps: float | None = Field(default=None, ge=0)
    observed_travel_time_s: float | None = Field(default=None, ge=0)

    observation_age_s: TimeS = Field(ge=0)

    missing: bool
    known_closed: bool

    occupancy: float | None = Field(default=None, ge=0, le=1)
    halting_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _missing_cannot_masquerade_as_zero(self) -> "EdgeObservation":
        # This is the concrete rule behind CONTRACTS.md's "It must not
        # contain ... future traffic truth" / missingness handling in the
        # master plan: `missing=True` must mean the numeric fields carry no
        # asserted value at all, not a silently-substituted 0.0.
        if self.missing and (
            self.observed_speed_mps is not None
            or self.observed_travel_time_s is not None
        ):
            raise ValueError(
                f"edge {self.edge_id!r}: missing=True but a numeric "
                "observation value is populated -- missing observations "
                "must be represented as None, never as 0.0"
            )
        if self.known_closed and self.observed_speed_mps not in (None, 0.0):
            raise ValueError(
                f"edge {self.edge_id!r}: known_closed=True but "
                f"observed_speed_mps={self.observed_speed_mps!r} "
                "(closed edges should report speed as None or 0.0, not a "
                "positive traversal speed)"
            )
        return self


class VehicleObservation(Contract):
    vehicle_id: VehicleId

    current_edge_id: RoadEdgeId | None
    current_node_id: RoadNodeId | None  # was `str` in V1 -- inconsistent
    # with scenario.Vehicle.current_node_id, which was already RoadNodeId.

    distance_remaining_m: float = Field(ge=0)

    onboard_request_ids: tuple[CustomerId, ...]
    remaining_load: float = Field(ge=0)

    executed_prefix_edge_ids: tuple[RoadEdgeId, ...]

    # Commitments already made and therefore not freely mutable.
    committed_request_ids: tuple[CustomerId, ...]

    @model_validator(mode="after")
    def _committed_is_subset_of_onboard_or_explicit(self) -> "VehicleObservation":
        # Commitments must be a real subset bookkeeping decision, not an
        # accidental disjoint list -- every committed request for THIS
        # vehicle should also appear onboard once it is picked up. We only
        # enforce the direction that is always true regardless of pickup
        # timing: no duplicate IDs within either tuple.
        for name, ids in (
            ("onboard_request_ids", self.onboard_request_ids),
            ("committed_request_ids", self.committed_request_ids),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(
                    f"vehicle {self.vehicle_id!r}: duplicate IDs in {name}"
                )
        return self


class VisibleJob(Contract):
    request_id: CustomerId

    demand: float = Field(ge=0)
    release_s: float = Field(ge=0)
    earliest_service_start_s: float = Field(ge=0)
    latest_service_start_s: float = Field(ge=0)
    service_duration_s: float = Field(ge=0)

    status: str

    @model_validator(mode="after")
    def _window_consistent(self) -> "VisibleJob":
        if self.latest_service_start_s < self.earliest_service_start_s:
            raise ValueError(
                f"job {self.request_id!r}: latest before earliest service "
                "start"
            )
        return self


class VisibleEvent(Contract):
    event_id: str

    # Only information that has become visible by observation_time_s.
    revealed_at_s: TimeS = Field(ge=0)
    event_type: str

    affected_parent_road_ids: tuple[str, ...]

    # A future effect_start_s is legitimate: an incident can be *announced*
    # before it *begins* (e.g. a scheduled closure). The causality boundary
    # that actually must never be crossed is on `revealed_at_s`, checked at
    # the Observation level below -- NOT on effect_start_s. (This corrects
    # the V1.0 review's suggested invariant.)
    effect_start_s: TimeS | None = None


class Observation(Contract):
    scenario_id: str
    episode_id: EpisodeId

    observation_time_s: TimeS = Field(ge=0)
    graph_version: GraphVersion

    edge_observations: tuple[EdgeObservation, ...]
    visible_jobs: tuple[VisibleJob, ...]
    fleet: tuple[VehicleObservation, ...]
    visible_events: tuple[VisibleEvent, ...]

    # Explicit list of jobs the policy can currently act upon.
    pending_request_ids: tuple[CustomerId, ...]

    # State version allows asynchronous candidates to be revalidated.
    state_version: str

    @field_validator("visible_jobs")
    @classmethod
    def _unique_job_ids(cls, v: tuple[VisibleJob, ...]) -> tuple[VisibleJob, ...]:
        ids = [j.request_id for j in v]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate request_id in visible_jobs")
        return v

    @field_validator("fleet")
    @classmethod
    def _unique_vehicle_ids(
        cls, v: tuple[VehicleObservation, ...]
    ) -> tuple[VehicleObservation, ...]:
        ids = [f.vehicle_id for f in v]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate vehicle_id in fleet")
        return v

    @model_validator(mode="after")
    def _causality_and_reference_checks(self) -> "Observation":
        # 1. The visibility rule, precisely: an event's *reveal* time can
        #    never be in the future relative to this observation. Its
        #    effect_start_s may legitimately be in the future.
        for event in self.visible_events:
            if event.revealed_at_s > self.observation_time_s:
                raise ValueError(
                    f"event {event.event_id!r}: revealed_at_s "
                    f"({event.revealed_at_s}) is after observation_time_s "
                    f"({self.observation_time_s}) -- this is the exact "
                    "leakage CONTRACTS.md's Visibility Rule forbids"
                )

        # 2. pending_request_ids must refer to jobs the observation actually
        #    exposes -- a dangling pending ID would let a policy "act on"
        #    a job it cannot see the details of.
        visible_ids = {j.request_id for j in self.visible_jobs}
        dangling = [rid for rid in self.pending_request_ids if rid not in visible_ids]
        if dangling:
            raise ValueError(
                f"pending_request_ids references jobs not present in "
                f"visible_jobs: {dangling}"
            )

        # 3. pending_request_ids has no duplicates.
        if len(self.pending_request_ids) != len(set(self.pending_request_ids)):
            raise ValueError("duplicate IDs in pending_request_ids")

        return self
