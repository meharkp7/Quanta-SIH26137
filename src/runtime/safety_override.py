"""
Step 16.11 — independent, auditable safety-override layer.

Architectural boundary
----------------------
The policy/controller requests one of the five scope actions.  This module
does NOT:
    - run PPO,
    - run QPSO,
    - validate a physical route,
    - inspect future simulator truth,
    - invent event information.

It receives explicit, already-derived safety facts from the current causal
state and may replace the requested action only when a declared hard-safety
condition requires it.

The layer deliberately returns an immutable decision *before* the routing
optimizer.  The existing ScopeDecision contract remains the authoritative
runtime audit contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.contracts.decision import ScopeAction, ScopeDecision
from src.runtime.scope_actions import (
    JobImpact,
    ScopeActionConfig,
    ScopeActionSelector,
    ScopeSelection,
)


ACTION_ORDER: tuple[ScopeAction, ...] = (
    ScopeAction.LOCAL,
    ScopeAction.VEHICLE,
    ScopeAction.REGIONAL,
    ScopeAction.GLOBAL,
)


@dataclass(frozen=True)
class SafetyOverrideContext:
    """
    Causal hard-safety facts available at the current decision point.

    These are facts/flags produced by upstream state/impact analysis.  The
    override layer does not infer them from future information.

    `affected_mutable_request_ids` identifies mutable jobs for which the
    current plan cannot safely be retained.

    `affected_vehicle_ids` identifies vehicles whose current assignment is
    implicated by the safety condition.

    `known_closed_edge_ids` is intentionally informational here.  The route
    validator remains authoritative for physical path legality.  A non-empty
    value is useful as an explicit audit reason when an active plan contains
    a known closure.

    `deadline_critical_request_ids` identifies mutable jobs whose current
    service plan has a declared hard deadline risk.

    `mandatory_replan` is reserved for a hard operational/legal condition
    already established by the current state.  It must not be used merely
    because a larger scope is predicted to perform better.
    """

    affected_mutable_request_ids: tuple[str, ...] = ()
    affected_vehicle_ids: tuple[str, ...] = ()
    known_closed_edge_ids: tuple[str, ...] = ()
    deadline_critical_request_ids: tuple[str, ...] = ()
    mandatory_replan: bool = False

    def __post_init__(self) -> None:
        for name, values in (
            ("affected_mutable_request_ids", self.affected_mutable_request_ids),
            ("affected_vehicle_ids", self.affected_vehicle_ids),
            ("known_closed_edge_ids", self.known_closed_edge_ids),
            ("deadline_critical_request_ids", self.deadline_critical_request_ids),
        ):
            normalized = tuple(str(value) for value in values)
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"duplicate IDs in {name}")
            if any(not value for value in normalized):
                raise ValueError(f"empty ID in {name}")

        affected = set(self.affected_mutable_request_ids)
        critical = set(self.deadline_critical_request_ids)

        if not critical.issubset(affected):
            raise ValueError(
                "deadline_critical_request_ids must be a subset of "
                "affected_mutable_request_ids"
            )

    @property
    def hard_reason_codes(self) -> tuple[str, ...]:
        """Stable machine-readable reasons for an override."""

        reasons: list[str] = []

        if self.mandatory_replan:
            reasons.append("MANDATORY_REPLAN")

        if self.known_closed_edge_ids:
            reasons.append("KNOWN_CLOSED_EDGE_ON_ACTIVE_NETWORK")

        if self.affected_mutable_request_ids:
            reasons.append("AFFECTED_MUTABLE_WORK")

        if self.deadline_critical_request_ids:
            reasons.append("HARD_DEADLINE_RISK")

        return tuple(reasons)

    @property
    def requires_override(self) -> bool:
        return bool(self.hard_reason_codes)


@dataclass(frozen=True)
class SafetyOverrideResult:
    """
    Immutable result of the safety layer.

    `requested_action` is the policy/controller proposal.
    `executed_action` is the action that may safely proceed to scope/QPSO.
    """

    requested_action: ScopeAction
    executed_action: ScopeAction

    requested_selection: ScopeSelection
    executed_selection: ScopeSelection

    overridden: bool
    override_reason: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.overridden != (
            self.requested_action != self.executed_action
        ):
            raise ValueError(
                "overridden must exactly match requested/executed action change"
            )

        if self.overridden and not self.override_reason:
            raise ValueError(
                "an overridden result requires a non-empty override_reason"
            )

        if not self.overridden and self.override_reason is not None:
            raise ValueError(
                "non-overridden result cannot carry override_reason"
            )

        if self.executed_selection.action != self.executed_action:
            raise ValueError(
                "executed_selection.action must equal executed_action"
            )

        if self.requested_selection.action != self.requested_action:
            raise ValueError(
                "requested_selection.action must equal requested_action"
            )


class SafetyOverridePolicy:
    """
    Permanent rule-based safety layer between policy selection and execution.

    Escalation is conservative:
        KEEP -> LOCAL -> VEHICLE -> REGIONAL -> GLOBAL

    The layer never declares a physical route feasible.  It only decides
    whether the requested *scope* is too small to respond to an already-known
    hard condition.
    """

    def __init__(
        self,
        *,
        selector: ScopeActionSelector | None = None,
        local_max_jobs: int = 10,
    ) -> None:
        config = ScopeActionConfig(
            local_max_jobs=int(local_max_jobs),
            safety_override_to_global=False,
            keep_requires_no_affected_mutable_jobs=False,
        )
        self.selector = selector or ScopeActionSelector(config=config)

    def apply(
        self,
        requested_action: ScopeAction,
        *,
        jobs: Sequence[JobImpact],
        context: SafetyOverrideContext,
        affected_zone_ids: Iterable[str] = (),
        legally_mutable_request_ids: Iterable[str] | None = None,
        legally_mutable_vehicle_ids: Iterable[str] | None = None,
    ) -> SafetyOverrideResult:
        """
        Apply the independent safety policy to a requested action.

        A request is overridden only when:
          1. a hard safety condition exists, and
          2. the requested scope does not already cover all hard-affected
             mutable work.

        No future labels, future events, or simulator outcomes are consulted.
        """

        requested_selection = self.selector.select(
            requested_action,
            jobs=jobs,
            affected_vehicle_ids=context.affected_vehicle_ids,
            affected_zone_ids=affected_zone_ids,
            legally_mutable_request_ids=legally_mutable_request_ids,
            legally_mutable_vehicle_ids=legally_mutable_vehicle_ids,
        )

        # KEEP is the one action for which affected mutable work is directly
        # contradictory to safe execution.
        requested_covers_risk = self._selection_covers_hard_work(
            requested_selection,
            context,
        )

        if not context.requires_override or requested_covers_risk:
            return SafetyOverrideResult(
                requested_action=requested_action,
                executed_action=requested_action,
                requested_selection=requested_selection,
                executed_selection=requested_selection,
                overridden=False,
                override_reason=None,
                reason_codes=(),
            )

        safe_selection = self._find_smallest_safe_scope(
            jobs=jobs,
            context=context,
            affected_zone_ids=affected_zone_ids,
            legally_mutable_request_ids=legally_mutable_request_ids,
            legally_mutable_vehicle_ids=legally_mutable_vehicle_ids,
        )

        if safe_selection is None:
            raise RuntimeError(
                "Safety override could not construct any legal mutable scope "
                "for a hard safety condition. Refusing to silently execute "
                "the unsafe request."
            )

        reason = self._build_reason(
            requested_action=requested_action,
            executed_action=safe_selection.action,
            context=context,
        )

        return SafetyOverrideResult(
            requested_action=requested_action,
            executed_action=safe_selection.action,
            requested_selection=requested_selection,
            executed_selection=safe_selection,
            overridden=True,
            override_reason=reason,
            reason_codes=context.hard_reason_codes,
        )

    def to_scope_decision(
        self,
        result: SafetyOverrideResult,
        *,
        scenario_id: str,
        state_version: str,
        decision_time_s: float,
        budget_seconds: float,
        selected_by: str,
        decision_version: str,
        decision_id: str | None = None,
    ) -> ScopeDecision:
        """
        Materialize the repository's authoritative ScopeDecision contract.

        The Pydantic contract performs the final consistency checks for
        requested/executed action and override bookkeeping.
        """

        return ScopeDecision(
            scenario_id=str(scenario_id),
            state_version=str(state_version),
            decision_time_s=float(decision_time_s),
            requested_action=result.requested_action,
            executed_action=result.executed_action,
            affected_vehicle_ids=tuple(
                result.executed_selection.affected_vehicle_ids
            ),
            mutable_request_ids=tuple(
                result.executed_selection.request_ids
            ),
            budget_seconds=float(budget_seconds),
            overridden=result.overridden,
            override_reason=result.override_reason,
            selection_reason=result.executed_selection.reason,
            selected_by=str(selected_by),
            decision_version=str(decision_version),
            decision_id=decision_id,
        )

    def _find_smallest_safe_scope(
        self,
        *,
        jobs: Sequence[JobImpact],
        context: SafetyOverrideContext,
        affected_zone_ids: Iterable[str],
        legally_mutable_request_ids: Iterable[str] | None,
        legally_mutable_vehicle_ids: Iterable[str] | None,
    ) -> ScopeSelection | None:
        """
        Choose the smallest action whose concrete scope covers every hard
        affected mutable request.

        The order is intentionally deterministic and independent of PPO.
        """

        hard_requests = set(context.affected_mutable_request_ids)

        # If the context says there is a mandatory replan but names no
        # affected jobs, there is no honest way to fabricate a scope.
        if not hard_requests:
            return None

        for action in ACTION_ORDER:
            selection = self.selector.select(
                action,
                jobs=jobs,
                affected_vehicle_ids=context.affected_vehicle_ids,
                affected_zone_ids=affected_zone_ids,
                legally_mutable_request_ids=legally_mutable_request_ids,
                legally_mutable_vehicle_ids=legally_mutable_vehicle_ids,
            )

            if hard_requests.issubset(set(selection.request_ids)):
                return selection

        return None

    @staticmethod
    def _selection_covers_hard_work(
        selection: ScopeSelection,
        context: SafetyOverrideContext,
    ) -> bool:
        hard_requests = set(context.affected_mutable_request_ids)

        if not hard_requests:
            return selection.action != ScopeAction.KEEP or not context.requires_override

        return hard_requests.issubset(set(selection.request_ids))

    @staticmethod
    def _build_reason(
        *,
        requested_action: ScopeAction,
        executed_action: ScopeAction,
        context: SafetyOverrideContext,
    ) -> str:
        codes = ", ".join(context.hard_reason_codes)

        return (
            f"Requested {requested_action.value} was overridden to "
            f"{executed_action.value} by the permanent safety policy. "
            f"Reason codes: {codes}."
        )