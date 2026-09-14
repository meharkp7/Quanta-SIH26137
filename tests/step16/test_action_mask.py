from __future__ import annotations

import numpy as np
import pytest

from src.contracts.decision import ScopeAction
from src.learning.action_mask import (
    ACTION_MASK_FEATURE_NAMES,
    ACTION_TO_INDEX,
    ActionFeasibilityMaskBuilder,
    ActionMaskConfig,
    validate_action_index,
)
from src.runtime.scope_actions import JobImpact


def _job(
    request_id: str,
    vehicle_id: str = "V1",
    *,
    zone_id: str = "Z1",
    affected: bool = False,
    slack: float = 600.0,
) -> JobImpact:
    return JobImpact(
        request_id=request_id,
        vehicle_id=vehicle_id,
        zone_id=zone_id,
        affected=affected,
        deadline_slack_s=slack,
    )


def test_no_mutable_jobs_allows_only_keep():
    mask = ActionFeasibilityMaskBuilder().build(
        jobs=(),
    )

    assert mask.available_actions == (
        ScopeAction.KEEP,
    )

    assert mask.allows(ScopeAction.KEEP)
    assert not mask.allows(ScopeAction.LOCAL)
    assert not mask.allows(ScopeAction.VEHICLE)
    assert not mask.allows(ScopeAction.REGIONAL)
    assert not mask.allows(ScopeAction.GLOBAL)

    assert mask.canonical_action == ScopeAction.KEEP


def test_affected_mutable_work_masks_keep():
    mask = ActionFeasibilityMaskBuilder().build(
        jobs=(
            _job(
                "J1",
                affected=True,
            ),
        ),
        affected_vehicle_ids=("V1",),
        affected_zone_ids=("Z1",),
    )

    assert not mask.allows(ScopeAction.KEEP)
    assert mask.any_available
    assert mask.allows(ScopeAction.LOCAL)
    assert mask.allows(ScopeAction.VEHICLE)
    assert mask.allows(ScopeAction.REGIONAL)
    assert mask.allows(ScopeAction.GLOBAL)


def test_unaffected_work_can_leave_keep_available():
    mask = ActionFeasibilityMaskBuilder().build(
        jobs=(
            _job(
                "J1",
                affected=False,
            ),
        ),
        affected_vehicle_ids=(),
        affected_zone_ids=(),
    )

    assert mask.allows(ScopeAction.KEEP)
    assert not mask.allows(ScopeAction.LOCAL)
    assert not mask.allows(ScopeAction.VEHICLE)
    assert not mask.allows(ScopeAction.REGIONAL)
    assert mask.allows(ScopeAction.GLOBAL)


def test_local_requires_affected_work():
    builder = ActionFeasibilityMaskBuilder(
        config=ActionMaskConfig(
            require_affected_for_local=True,
            mask_duplicate_scopes=False,
        )
    )

    mask = builder.build(
        jobs=(
            _job(
                "J1",
                affected=False,
            ),
        ),
        affected_vehicle_ids=(),
        affected_zone_ids=(),
    )

    assert not mask.allows(ScopeAction.LOCAL)
    assert mask.allows(ScopeAction.GLOBAL)


def test_legal_mutability_filters_actions():
    mask = ActionFeasibilityMaskBuilder().build(
        jobs=(
            _job(
                "J1",
                affected=True,
            ),
            _job(
                "J2",
                affected=True,
            ),
        ),
        affected_vehicle_ids=("V1",),
        affected_zone_ids=("Z1",),
        legally_mutable_request_ids=("J2",),
    )

    assert mask.scope_request_ids[
        ACTION_TO_INDEX[ScopeAction.GLOBAL]
    ] == ("J2",)

    assert mask.scope_request_ids[
        ACTION_TO_INDEX[ScopeAction.LOCAL]
    ] == ("J2",)


def test_legal_vehicle_filter_is_respected():
    mask = ActionFeasibilityMaskBuilder().build(
        jobs=(
            _job(
                "J1",
                "V1",
                affected=True,
            ),
            _job(
                "J2",
                "V2",
                affected=True,
            ),
        ),
        affected_vehicle_ids=("V1", "V2"),
        legally_mutable_vehicle_ids=("V2",),
    )

    assert mask.scope_request_ids[
        ACTION_TO_INDEX[ScopeAction.GLOBAL]
    ] == ("J2",)


def test_duplicate_scope_masking_is_deterministic():
    mask = ActionFeasibilityMaskBuilder(
        config=ActionMaskConfig(
            mask_duplicate_scopes=True,
        )
    ).build(
        jobs=(
            _job(
                "J1",
                affected=True,
            ),
        ),
        affected_vehicle_ids=("V1",),
        affected_zone_ids=("Z1",),
    )

    # On a one-job fixture LOCAL, VEHICLE, REGIONAL and GLOBAL can collapse
    # to the same exact operational scope. LOCAL is the deterministic
    # representative; broader duplicate labels are masked.
    assert mask.canonical_action == ScopeAction.LOCAL
    assert mask.allows(ScopeAction.LOCAL)
    assert not mask.allows(ScopeAction.VEHICLE)
    assert not mask.allows(ScopeAction.REGIONAL)
    assert not mask.allows(ScopeAction.GLOBAL)


def test_duplicate_scope_masking_can_be_disabled():
    mask = ActionFeasibilityMaskBuilder(
        config=ActionMaskConfig(
            mask_duplicate_scopes=False,
        )
    ).build(
        jobs=(
            _job(
                "J1",
                affected=True,
            ),
        ),
        affected_vehicle_ids=("V1",),
        affected_zone_ids=("Z1",),
    )

    assert mask.allows(ScopeAction.LOCAL)
    assert mask.allows(ScopeAction.VEHICLE)
    assert mask.allows(ScopeAction.REGIONAL)
    assert mask.allows(ScopeAction.GLOBAL)


def test_mask_is_fixed_five_action_float_vector():
    mask = ActionFeasibilityMaskBuilder().build(
        jobs=(),
    )

    assert mask.mask.shape == (5,)
    assert mask.mask.dtype == np.float32
    assert mask.mask.flags.writeable is False
    assert ACTION_MASK_FEATURE_NAMES == (
        "keep",
        "local",
        "vehicle",
        "regional",
        "global",
    )


def test_boolean_and_float_exports_are_defensive_copies():
    mask = ActionFeasibilityMaskBuilder().build(
        jobs=(),
    )

    boolean = mask.as_bool()
    floating = mask.as_float()

    boolean[0] = False
    floating[0] = 0.0

    assert mask.allows(ScopeAction.KEEP)


def test_invalid_sampled_action_is_rejected():
    mask = ActionFeasibilityMaskBuilder().build(
        jobs=(),
    )

    assert (
        validate_action_index(
            0,
            mask,
        )
        == ScopeAction.KEEP
    )

    with pytest.raises(
        ValueError,
        match="masked as infeasible",
    ):
        validate_action_index(
            1,
            mask,
        )


def test_invalid_action_index_is_rejected():
    mask = ActionFeasibilityMaskBuilder().build(
        jobs=(),
    )

    with pytest.raises(
        ValueError,
        match="must be in",
    ):
        validate_action_index(
            5,
            mask,
        )


def test_mask_never_becomes_all_zero():
    # GLOBAL is retained as the final replanning escape hatch.
    mask = ActionFeasibilityMaskBuilder().build(
        jobs=(
            _job(
                "J1",
                affected=True,
            ),
        ),
        affected_vehicle_ids=("V1",),
        affected_zone_ids=("Z1",),
        legally_mutable_request_ids=("J1",),
        legally_mutable_vehicle_ids=("V1",),
    )

    assert mask.any_available
    assert np.any(mask.mask == 1.0)


def test_action_order_matches_project_contract():
    mask = ActionFeasibilityMaskBuilder(
        config=ActionMaskConfig(
            mask_duplicate_scopes=False,
        )
    ).build(
        jobs=(
            _job(
                "J1",
                affected=True,
            ),
            _job(
                "J2",
                vehicle_id="V2",
                affected=True,
                zone_id="Z2",
            ),
        ),
        affected_vehicle_ids=("V1", "V2"),
        affected_zone_ids=("Z1", "Z2"),
    )

    assert ACTION_TO_INDEX == {
        ScopeAction.KEEP: 0,
        ScopeAction.LOCAL: 1,
        ScopeAction.VEHICLE: 2,
        ScopeAction.REGIONAL: 3,
        ScopeAction.GLOBAL: 4,
    }

    assert mask.indices() == (
        1,
        2,
        3,
        4,
    )
