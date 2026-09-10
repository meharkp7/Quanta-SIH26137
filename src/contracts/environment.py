from __future__ import annotations

from pydantic import Field, model_validator

from ._base import Contract
from .core_types import EventId, RoadEdgeId, TimeS

# NOTE: this module must never be imported by observation.py, and nothing
# in observation.py should ever be constructed FROM an object defined here.
# That is the actual enforcement mechanism behind CONTRACTS.md's Visibility
# Rule at the module-boundary level: policy-visible code has no import path
# to future/hidden truth, so leaking it requires an explicit, reviewable
# act (manually copying a field across), not an accidental one.


class HiddenEvent(Contract):
    event_id: EventId

    generation_time_s: TimeS = Field(ge=0)
    reveal_time_s: TimeS = Field(ge=0)

    effect_start_s: TimeS = Field(ge=0)
    effect_end_s: TimeS | None = None

    event_type: str

    affected_parent_road_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _timeline_is_ordered(self) -> "HiddenEvent":
        if self.reveal_time_s < self.generation_time_s:
            raise ValueError(
                f"event {self.event_id!r}: reveal_time_s precedes "
                "generation_time_s"
            )
        if self.effect_end_s is not None and self.effect_end_s < self.effect_start_s:
            raise ValueError(
                f"event {self.event_id!r}: effect_end_s precedes "
                "effect_start_s"
            )
        return self


class EdgeTruth(Contract):
    edge_id: RoadEdgeId

    timestamp_s: TimeS = Field(ge=0)

    true_speed_mps: float | None = Field(default=None, ge=0)
    true_travel_time_s: float | None = Field(default=None, ge=0)

    is_closed: bool

    @model_validator(mode="after")
    def _closed_edges_have_no_positive_speed(self) -> "EdgeTruth":
        if self.is_closed and self.true_speed_mps not in (None, 0.0):
            raise ValueError(
                f"edge {self.edge_id!r} is_closed=True but "
                f"true_speed_mps={self.true_speed_mps!r}"
            )
        return self


class EnvironmentTruth(Contract):
    episode_id: str

    hidden_events: tuple[HiddenEvent, ...]
    edge_truth: tuple[EdgeTruth, ...]

    simulator_seed: int
