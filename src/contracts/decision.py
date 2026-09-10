from __future__ import annotations

from pydantic import Field, model_validator

from ._base import Contract
from .core_types import CustomerId, DecisionId, ScopeAction, TimeS, VehicleId


class ScopeDecision(Contract):
    scenario_id: str
    state_version: str

    decision_time_s: TimeS = Field(ge=0)

    # `requested_action`/`executed_action` were already restricted to the
    # five legal values by being typed `ScopeAction` rather than `str` in
    # V1 -- this was correct and is unchanged here. What V1 was missing is
    # everything below.
    requested_action: ScopeAction
    executed_action: ScopeAction

    affected_vehicle_ids: tuple[VehicleId, ...]
    mutable_request_ids: tuple[CustomerId, ...]

    budget_seconds: float = Field(ge=0)

    overridden: bool
    override_reason: str | None

    selection_reason: str

    # Policy provenance.
    selected_by: str

    decision_version: str
    decision_id: DecisionId | None = None

    @model_validator(mode="after")
    def _override_bookkeeping_is_honest(self) -> "ScopeDecision":
        if self.overridden:
            if self.requested_action == self.executed_action:
                raise ValueError(
                    "overridden=True but requested_action == executed_action "
                    "-- an override must actually change what gets executed, "
                    "or it is not an override"
                )
            if not self.override_reason:
                raise ValueError(
                    "overridden=True requires a non-empty override_reason "
                    "-- CONTRACTS.md requires safety overrides to be "
                    "explicit and auditable"
                )
        else:
            if self.requested_action != self.executed_action:
                raise ValueError(
                    "requested_action != executed_action but overridden=False "
                    "-- any deviation from the requested action must be "
                    "recorded as an override"
                )
            if self.override_reason:
                raise ValueError(
                    "override_reason is set but overridden=False -- this "
                    "would credit an overridden proposal as if the actor "
                    "independently chose the executed action"
                )
        return self

    @model_validator(mode="after")
    def _keep_has_no_mutable_scope(self) -> "ScopeDecision":
        # KEEP means "no QPSO call" per the master plan's scope table. A
        # nonempty mutable scope under an executed KEEP would silently
        # contradict that.
        if self.executed_action == ScopeAction.KEEP and self.mutable_request_ids:
            raise ValueError(
                "executed_action=KEEP but mutable_request_ids is nonempty"
            )
        return self

    @model_validator(mode="after")
    def _no_duplicate_ids(self) -> "ScopeDecision":
        for name, ids in (
            ("affected_vehicle_ids", self.affected_vehicle_ids),
            ("mutable_request_ids", self.mutable_request_ids),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate IDs in {name}")
        return self
