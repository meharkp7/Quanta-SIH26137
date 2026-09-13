from __future__ import annotations

import pytest

from src.contracts.decision import ScopeAction
from src.runtime.scope_actions import JobImpact
from src.runtime.safety_override import (
    SafetyOverrideContext,
    SafetyOverridePolicy,
)


def _job(
    request_id: str,
    vehicle_id: str = "V1",
    zone_id: str = "Z1",
    *,
    affected: bool = False,
    slack: float = 100.0,
) -> JobImpact:
    return JobImpact(
        request_id=request_id,
        vehicle_id=vehicle_id,
        zone_id=zone_id,
        affected=affected,
        deadline_slack_s=slack,
        route_overlap_fraction=1.0 if affected else 0.0,
    )


def test_safe_request_is_not_overridden():
    policy = SafetyOverridePolicy()

    result = policy.apply(
        ScopeAction.GLOBAL,
        jobs=(_job("J1"),),
        context=SafetyOverrideContext(),
    )

    assert not result.overridden
    assert result.requested_action == ScopeAction.GLOBAL
    assert result.executed_action == ScopeAction.GLOBAL
    assert result.override_reason is None


def test_keep_is_overridden_when_affected_mutable_work_exists():
    policy = SafetyOverridePolicy()

    result = policy.apply(
        ScopeAction.KEEP,
        jobs=(
            _job("J1", affected=True),
            _job("J2"),
        ),
        context=SafetyOverrideContext(
            affected_mutable_request_ids=("J1",),
            affected_vehicle_ids=("V1",),
        ),
    )

    assert result.overridden
    assert result.requested_action == ScopeAction.KEEP
    assert result.executed_action == ScopeAction.LOCAL
    # LOCAL is defined by the Step-15 scope selector as the top impact-ranked
    # mutable work across affected vehicles, not as "affected jobs only".
    assert set(result.executed_selection.request_ids) == {"J1", "J2"}
    assert "AFFECTED_MUTABLE_WORK" in result.reason_codes


def test_keep_escalates_beyond_local_when_more_than_local_cap_is_affected():
    policy = SafetyOverridePolicy(local_max_jobs=1)

    jobs = tuple(
        _job(f"J{i}", affected=True)
        for i in range(1, 4)
    )

    result = policy.apply(
        ScopeAction.KEEP,
        jobs=jobs,
        context=SafetyOverrideContext(
            affected_mutable_request_ids=("J1", "J2", "J3"),
            affected_vehicle_ids=("V1",),
        ),
    )

    # One vehicle contains all affected work, so VEHICLE is the smallest
    # scope capable of covering all hard-affected jobs.
    assert result.overridden
    assert result.executed_action == ScopeAction.VEHICLE
    assert set(result.executed_selection.request_ids) == {"J1", "J2", "J3"}


def test_requested_local_is_overridden_if_it_does_not_cover_hard_work():
    policy = SafetyOverridePolicy(local_max_jobs=1)

    jobs = (
        _job("J1", vehicle_id="V1", affected=True),
        _job("J2", vehicle_id="V2", affected=True),
    )

    result = policy.apply(
        ScopeAction.LOCAL,
        jobs=jobs,
        context=SafetyOverrideContext(
            affected_mutable_request_ids=("J1", "J2"),
            affected_vehicle_ids=("V1", "V2"),
        ),
    )

    assert result.overridden
    assert result.executed_action == ScopeAction.REGIONAL
    assert set(result.executed_selection.request_ids) == {"J1", "J2"}


def test_requested_scope_is_kept_if_it_already_covers_hard_work():
    policy = SafetyOverridePolicy()

    jobs = (
        _job("J1", affected=True),
        _job("J2"),
    )

    result = policy.apply(
        ScopeAction.LOCAL,
        jobs=jobs,
        context=SafetyOverrideContext(
            affected_mutable_request_ids=("J1",),
            affected_vehicle_ids=("V1",),
        ),
    )

    assert not result.overridden
    assert result.executed_action == ScopeAction.LOCAL
    # The requested LOCAL scope already contains the hard-affected J1.
    # LOCAL may also contain other mutable work selected by the Step-15
    # impact-ranking rule.
    assert set(result.executed_selection.request_ids) == {"J1", "J2"}


def test_deadline_risk_is_audited():
    policy = SafetyOverridePolicy()

    result = policy.apply(
        ScopeAction.KEEP,
        jobs=(_job("J1", affected=True, slack=1.0),),
        context=SafetyOverrideContext(
            affected_mutable_request_ids=("J1",),
            deadline_critical_request_ids=("J1",),
            affected_vehicle_ids=("V1",),
        ),
    )

    assert result.overridden
    assert "HARD_DEADLINE_RISK" in result.reason_codes
    assert "AFFECTED_MUTABLE_WORK" in result.reason_codes


def test_known_closure_alone_cannot_fabricate_a_replan_scope():
    policy = SafetyOverridePolicy()

    with pytest.raises(RuntimeError, match="could not construct"):
        policy.apply(
            ScopeAction.KEEP,
            jobs=(_job("J1"),),
            context=SafetyOverrideContext(
                known_closed_edge_ids=("E1",),
                mandatory_replan=True,
            ),
        )


def test_duplicate_context_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate IDs"):
        SafetyOverrideContext(
            affected_mutable_request_ids=("J1", "J1"),
        )


def test_deadline_ids_must_be_affected():
    with pytest.raises(ValueError, match="subset"):
        SafetyOverrideContext(
            affected_mutable_request_ids=("J1",),
            deadline_critical_request_ids=("J2",),
        )


def test_scope_decision_contract_is_materialized_honestly():
    policy = SafetyOverridePolicy()

    result = policy.apply(
        ScopeAction.KEEP,
        jobs=(_job("J1", affected=True),),
        context=SafetyOverrideContext(
            affected_mutable_request_ids=("J1",),
            affected_vehicle_ids=("V1",),
        ),
    )

    decision = policy.to_scope_decision(
        result,
        scenario_id="S3_BASE",
        state_version="S3_BASE:t0:graph:open",
        decision_time_s=60.0,
        budget_seconds=0.25,
        selected_by="ppo_policy",
        decision_version="step16.11-v1",
        decision_id="D1",
    )

    assert decision.requested_action == ScopeAction.KEEP
    assert decision.executed_action == ScopeAction.LOCAL
    assert decision.overridden is True
    assert decision.override_reason
    assert decision.mutable_request_ids == ("J1",)


def test_non_overridden_materialization_preserves_requested_action():
    policy = SafetyOverridePolicy()

    result = policy.apply(
        ScopeAction.LOCAL,
        jobs=(_job("J1", affected=True),),
        context=SafetyOverrideContext(
            affected_mutable_request_ids=("J1",),
            affected_vehicle_ids=("V1",),
        ),
    )

    decision = policy.to_scope_decision(
        result,
        scenario_id="S3_BASE",
        state_version="S3_BASE:t0:graph:open",
        decision_time_s=60.0,
        budget_seconds=0.25,
        selected_by="ppo_policy",
        decision_version="step16.11-v1",
    )

    assert decision.requested_action == ScopeAction.LOCAL
    assert decision.executed_action == ScopeAction.LOCAL
    assert decision.overridden is False
    assert decision.override_reason is None


def test_override_does_not_claim_qpso_or_route_feasibility():
    policy = SafetyOverridePolicy()

    result = policy.apply(
        ScopeAction.KEEP,
        jobs=(_job("J1", affected=True),),
        context=SafetyOverrideContext(
            affected_mutable_request_ids=("J1",),
            affected_vehicle_ids=("V1",),
            known_closed_edge_ids=("E1",),
        ),
    )

    assert result.executed_action == ScopeAction.LOCAL
    # The safety layer only selected the scope; it did not create a route.
    assert result.executed_selection.request_ids == ("J1",)