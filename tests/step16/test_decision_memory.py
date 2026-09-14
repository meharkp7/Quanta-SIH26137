from __future__ import annotations

import numpy as np
import pytest

from src.contracts.core_types import ScopeAction
from src.contracts.decision import ScopeDecision
from src.learning.state.decision_memory import (
    DECISION_MEMORY_FEATURE_NAMES,
    DecisionMemory,
)


def _decision(
    *,
    scenario_id: str = "scenario-test",
    state_version: str = "state-1",
    decision_time_s: float = 60.0,
    requested_action: ScopeAction = ScopeAction.LOCAL,
    executed_action: ScopeAction = ScopeAction.LOCAL,
    affected_vehicle_ids: tuple[str, ...] = ("V1",),
    mutable_request_ids: tuple[str, ...] = ("J1", "J2"),
    budget_seconds: float = 10.0,
    overridden: bool = False,
    override_reason: str | None = None,
    selection_reason: str = "test selection",
    selected_by: str = "rule",
    decision_version: str = "1",
    decision_id: str | None = "D1",
) -> ScopeDecision:
    return ScopeDecision(
        scenario_id=scenario_id,
        state_version=state_version,
        decision_time_s=decision_time_s,
        requested_action=requested_action,
        executed_action=executed_action,
        affected_vehicle_ids=affected_vehicle_ids,
        mutable_request_ids=mutable_request_ids,
        budget_seconds=budget_seconds,
        overridden=overridden,
        override_reason=override_reason,
        selection_reason=selection_reason,
        selected_by=selected_by,
        decision_version=decision_version,
        decision_id=decision_id,
    )


# ============================================================================
# Basic lifecycle
# ============================================================================


def test_empty_memory_has_no_records():
    memory = DecisionMemory(history_length=4)

    assert memory.is_empty
    assert memory.size == 0
    assert memory.records() == ()


def test_append_preserves_decision_contract():
    memory = DecisionMemory(history_length=4)

    decision = _decision()

    record = memory.append(decision)

    assert record.scenario_id == "scenario-test"
    assert record.state_version == "state-1"
    assert record.requested_action == "LOCAL"
    assert record.executed_action == "LOCAL"
    assert record.affected_vehicle_ids == ("V1",)
    assert record.mutable_request_ids == ("J1", "J2")
    assert record.budget_seconds == 10.0
    assert record.overridden is False
    assert record.decision_version == "1"
    assert record.decision_id == "D1"


def test_first_decision_has_zero_temporal_churn():
    memory = DecisionMemory(history_length=4)

    record = memory.append(_decision())

    assert record.time_since_previous_decision_s == 0.0
    assert record.requested_action_changed is False
    assert record.executed_action_changed is False
    assert record.mutable_request_overlap == 0.0
    assert record.affected_vehicle_overlap == 0.0
    assert record.mutable_scope_changed is False


# ============================================================================
# Temporal behaviour
# ============================================================================


def test_time_since_previous_decision_is_causal():
    memory = DecisionMemory(history_length=4)

    memory.append(
        _decision(
            decision_time_s=60.0,
            decision_version="1",
        )
    )

    second = memory.append(
        _decision(
            decision_time_s=120.0,
            decision_version="2",
        )
    )

    assert second.time_since_previous_decision_s == 60.0


def test_non_monotonic_decision_time_is_rejected():
    memory = DecisionMemory(history_length=4)

    memory.append(
        _decision(
            decision_time_s=100.0,
            decision_version="1",
        )
    )

    with pytest.raises(
        ValueError,
        match="decision_time_s must strictly increase",
    ):
        memory.append(
            _decision(
                decision_time_s=100.0,
                decision_version="2",
            )
        )


def test_numeric_decision_version_must_increase():
    memory = DecisionMemory(history_length=4)

    memory.append(
        _decision(
            decision_time_s=60.0,
            decision_version="5",
        )
    )

    with pytest.raises(
        ValueError,
        match="decision_version must strictly increase",
    ):
        memory.append(
            _decision(
                decision_time_s=120.0,
                decision_version="5",
            )
        )


def test_non_numeric_decision_versions_are_allowed():
    memory = DecisionMemory(history_length=4)

    memory.append(
        _decision(
            decision_time_s=60.0,
            decision_version="decision-a",
        )
    )

    record = memory.append(
        _decision(
            decision_time_s=120.0,
            decision_version="decision-b",
        )
    )

    assert record.decision_version == "decision-b"


def test_scenario_change_requires_reset():
    memory = DecisionMemory(history_length=4)

    memory.append(
        _decision(
            scenario_id="scenario-a",
            decision_time_s=60.0,
            decision_version="1",
        )
    )

    with pytest.raises(
        ValueError,
        match="scenario_id changed without reset",
    ):
        memory.append(
            _decision(
                scenario_id="scenario-b",
                decision_time_s=120.0,
                decision_version="2",
            )
        )


# ============================================================================
# Scope churn
# ============================================================================


def test_action_changes_are_recorded():
    memory = DecisionMemory(history_length=4)

    memory.append(
        _decision(
            requested_action=ScopeAction.LOCAL,
            executed_action=ScopeAction.LOCAL,
            decision_time_s=60.0,
            decision_version="1",
        )
    )

    record = memory.append(
        _decision(
            requested_action=ScopeAction.GLOBAL,
            executed_action=ScopeAction.GLOBAL,
            decision_time_s=120.0,
            decision_version="2",
        )
    )

    assert record.requested_action_changed is True
    assert record.executed_action_changed is True


def test_scope_overlap_is_computed_deterministically():
    memory = DecisionMemory(history_length=4)

    memory.append(
        _decision(
            affected_vehicle_ids=("V1", "V2"),
            mutable_request_ids=("J1", "J2", "J3"),
            decision_time_s=60.0,
            decision_version="1",
        )
    )

    record = memory.append(
        _decision(
            affected_vehicle_ids=("V2", "V3"),
            mutable_request_ids=("J2", "J3", "J4"),
            decision_time_s=120.0,
            decision_version="2",
        )
    )

    assert record.affected_vehicle_overlap == pytest.approx(1.0 / 3.0)
    assert record.mutable_request_overlap == pytest.approx(2.0 / 4.0)
    assert record.mutable_scope_changed is True


def test_identical_scope_has_full_overlap():
    memory = DecisionMemory(history_length=4)

    decision = _decision(
        affected_vehicle_ids=("V1", "V2"),
        mutable_request_ids=("J1", "J2"),
        decision_time_s=60.0,
        decision_version="1",
    )

    memory.append(decision)

    record = memory.append(
        _decision(
            affected_vehicle_ids=("V1", "V2"),
            mutable_request_ids=("J1", "J2"),
            decision_time_s=120.0,
            decision_version="2",
        )
    )

    assert record.affected_vehicle_overlap == 1.0
    assert record.mutable_request_overlap == 1.0
    assert record.mutable_scope_changed is False


# ============================================================================
# Overrides
# ============================================================================


def test_override_information_is_preserved():
    memory = DecisionMemory(history_length=4)

    record = memory.append(
        _decision(
            requested_action=ScopeAction.KEEP,
            executed_action=ScopeAction.LOCAL,
            overridden=True,
            override_reason="KEEP was unsafe",
            decision_time_s=60.0,
            decision_version="1",
        )
    )

    assert record.requested_action == "KEEP"
    assert record.executed_action == "LOCAL"
    assert record.overridden is True
    assert record.override_reason == "KEEP was unsafe"


# ============================================================================
# Bounded history
# ============================================================================


def test_history_is_bounded():
    memory = DecisionMemory(history_length=3)

    for index in range(5):
        memory.append(
            _decision(
                decision_time_s=float((index + 1) * 60),
                decision_version=str(index + 1),
                decision_id=f"D{index + 1}",
            )
        )

    assert memory.size == 3

    records = memory.records()

    assert [record.decision_id for record in records] == [
        "D3",
        "D4",
        "D5",
    ]


# ============================================================================
# Batch encoding
# ============================================================================


def test_empty_batch_is_explicitly_left_padded():
    memory = DecisionMemory(history_length=4)

    batch = memory.as_batch()

    assert batch.length == 4
    assert batch.valid_count == 0
    assert batch.padding_count == 4

    assert np.array_equal(
        batch.valid_mask,
        np.zeros(4, dtype=np.float32),
    )

    assert np.array_equal(
        batch.features,
        np.zeros(
            (4, len(DECISION_MEMORY_FEATURE_NAMES)),
            dtype=np.float32,
        ),
    )


def test_batch_contains_left_padding_and_real_records():
    memory = DecisionMemory(history_length=4)

    memory.append(
        _decision(
            decision_time_s=60.0,
            decision_version="1",
        )
    )

    batch = memory.as_batch()

    assert batch.length == 4
    assert batch.valid_count == 1
    assert batch.padding_count == 3

    assert np.array_equal(
        batch.valid_mask,
        np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )

    assert batch.decision_times_s[-1] == 60.0


def test_feature_vector_has_stable_contract():
    memory = DecisionMemory(
        history_length=4,
        max_budget_seconds=20.0,
        max_decision_interval_seconds=120.0,
    )

    memory.append(
        _decision(
            requested_action=ScopeAction.LOCAL,
            executed_action=ScopeAction.LOCAL,
            affected_vehicle_ids=("V1", "V2"),
            mutable_request_ids=("J1", "J2", "J3"),
            budget_seconds=10.0,
            decision_time_s=60.0,
            decision_version="1",
        )
    )

    batch = memory.as_batch()

    features = batch.features[-1]

    assert features.shape == (
        len(DECISION_MEMORY_FEATURE_NAMES),
    )

    # has_decision
    assert features[0] == 1.0

    # requested LOCAL
    assert features[2] == 1.0

    # executed LOCAL
    assert features[7] == 1.0

    # no override
    assert features[11] == 0.0

    # 2 / 10 affected vehicles
    assert features[12] == pytest.approx(0.2)

    # 3 / 50 mutable requests
    assert features[13] == pytest.approx(0.06)

    # 10 / 20 seconds
    assert features[14] == pytest.approx(0.5)

    # first decision has no previous interval
    assert features[15] == 0.0


def test_second_decision_exposes_replanning_interval_and_churn():
    memory = DecisionMemory(
        history_length=4,
        max_decision_interval_seconds=120.0,
    )

    memory.append(
        _decision(
            requested_action=ScopeAction.LOCAL,
            executed_action=ScopeAction.LOCAL,
            affected_vehicle_ids=("V1",),
            mutable_request_ids=("J1", "J2"),
            decision_time_s=60.0,
            decision_version="1",
        )
    )

    memory.append(
        _decision(
            requested_action=ScopeAction.GLOBAL,
            executed_action=ScopeAction.GLOBAL,
            affected_vehicle_ids=("V1", "V2"),
            mutable_request_ids=("J2", "J3"),
            decision_time_s=120.0,
            decision_version="2",
        )
    )

    batch = memory.as_batch()
    features = batch.features[-1]

    # GLOBAL requested
    assert features[5] == 1.0

    # GLOBAL executed
    assert features[10] == 1.0

    # 60 / 120 seconds
    assert features[15] == pytest.approx(0.5)

    # requested action changed
    assert features[16] == 1.0

    # executed action changed
    assert features[17] == 1.0

    # J2 overlap: 1 / 3
    assert features[18] == pytest.approx(1.0 / 3.0)

    # V1 overlap: 1 / 2
    assert features[19] == pytest.approx(0.5)

    # mutable scope changed
    assert features[20] == 1.0


# ============================================================================
# Immutability
# ============================================================================


def test_batch_arrays_are_read_only():
    memory = DecisionMemory(history_length=4)

    memory.append(_decision())

    batch = memory.as_batch()

    with pytest.raises(ValueError):
        batch.features[0, 0] = 99.0

    with pytest.raises(ValueError):
        batch.valid_mask[0] = 1.0

    with pytest.raises(ValueError):
        batch.decision_times_s[0] = 99.0


def test_records_are_immutable():
    memory = DecisionMemory(history_length=4)

    record = memory.append(_decision())

    with pytest.raises(AttributeError):
        record.executed_action = "GLOBAL"


# ============================================================================
# Reset
# ============================================================================


def test_reset_clears_temporal_and_identity_state():
    memory = DecisionMemory(history_length=4)

    memory.reset(
        episode_id="episode-a",
        scenario_id="scenario-a",
    )

    memory.append(
        _decision(
            scenario_id="scenario-a",
            decision_time_s=100.0,
            decision_version="10",
        )
    )

    memory.reset(
        episode_id="episode-b",
        scenario_id="scenario-b",
    )

    assert memory.is_empty
    assert memory.episode_id == "episode-b"
    assert memory.scenario_id == "scenario-b"

    # The new episode may restart its own clock/version sequence.
    record = memory.append(
        _decision(
            scenario_id="scenario-b",
            decision_time_s=10.0,
            decision_version="1",
        )
    )

    assert record.sequence_index == 0