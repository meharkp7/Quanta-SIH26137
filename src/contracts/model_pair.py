from __future__ import annotations

from pydantic import Field, model_validator

from ._base import Contract
from ._immutable import ImmutableStrMap
from .core_types import (
    ForecastVersion,
    ModelPromotionStatus,
    PolicyVersion,
    SchemaVersion,
    ScalerVersion,
)


class ModelPair(Contract):
    pair_version: str

    forecast_version: ForecastVersion
    policy_version: PolicyVersion

    schema_version: SchemaVersion
    scaler_version: ScalerVersion

    action_semantics_version: str
    reward_version: str
    qpso_version: str
    sumo_version: str

    training_cutoff_time_s: float = Field(ge=0)
    parent_pair_version: str | None

    validation_status: ModelPromotionStatus
    validation_summary: str

    created_at_s: float = Field(ge=0)

    # Explicit rollback target.
    rollback_pair_version: str | None

    metadata: ImmutableStrMap = Field(default_factory=ImmutableStrMap)

    @model_validator(mode="after")
    def _pair_is_not_its_own_parent_or_rollback(self) -> "ModelPair":
        if self.parent_pair_version == self.pair_version:
            raise ValueError(
                f"pair {self.pair_version!r} lists itself as parent_pair_version"
            )
        if self.rollback_pair_version == self.pair_version:
            raise ValueError(
                f"pair {self.pair_version!r} lists itself as "
                "rollback_pair_version"
            )
        return self

    @model_validator(mode="after")
    def _accepted_pairs_are_not_missing_a_rollback_target(self) -> "ModelPair":
        # A pair that has been ACCEPTED for live use with no rollback
        # target and no parent is, by definition, un-rollback-able --
        # acceptable only for the very first released pair.
        if (
            self.validation_status == ModelPromotionStatus.ACCEPTED
            and self.rollback_pair_version is None
            and self.parent_pair_version is not None
        ):
            raise ValueError(
                f"pair {self.pair_version!r} is ACCEPTED, has a parent "
                f"({self.parent_pair_version!r}), but declares no "
                "rollback_pair_version -- an accepted non-initial release "
                "must be rollback-able"
            )
        return self
