"""
Step 16.10 — Explicit action-feasibility masking.

The PPO controller has exactly five scope actions:

    KEEP / LOCAL / VEHICLE / REGIONAL / GLOBAL

This module determines which actions are structurally selectable from the
currently visible, legally mutable work.

Important boundary
------------------
This is NOT the route validator.

The mask does not attempt to prove that a QPSO route is feasible. Physical
road legality, capacity, commitments, time windows and final route validity
remain validator responsibilities.

The mask exists so a PPO implementation that supports action masking can
avoid selecting an action whose defined Step-15 scope is empty or unsafe.

A separate safety layer remains mandatory even when a masked PPO policy is
used. This follows the project contract: requested and executed actions are
distinct and safety overrides are auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.contracts.decision import ScopeAction
from src.runtime.scope_actions import (
    JobImpact,
    ScopeActionConfig,
    ScopeActionSelector,
)


ACTION_MASK_FEATURE_NAMES: tuple[str, ...] = (
    "keep",
    "local",
    "vehicle",
    "regional",
    "global",
)


ACTION_TO_INDEX: dict[ScopeAction, int] = {
    ScopeAction.KEEP: 0,
    ScopeAction.LOCAL: 1,
    ScopeAction.VEHICLE: 2,
    ScopeAction.REGIONAL: 3,
    ScopeAction.GLOBAL: 4,
}

INDEX_TO_ACTION: dict[int, ScopeAction] = {
    index: action
    for action, index in ACTION_TO_INDEX.items()
}


@dataclass(frozen=True)
class ActionMask:
    """
    Immutable five-action feasibility mask.

    ``mask[i] == 1`` means the actor may propose that action.
    ``mask[i] == 0`` means the action is structurally unavailable.

    ``available_actions`` contains actions in canonical action order.
    ``canonical_action`` is the deterministic representative when several
    feasible actions produce exactly the same scope.
    """

    mask: np.ndarray
    available_actions: tuple[ScopeAction, ...]
    canonical_action: ScopeAction | None

    # Exact scopes used to derive the mask. This makes the decision boundary
    # auditable without requiring another selector pass.
    scope_request_ids: tuple[tuple[str, ...], ...]
    scope_vehicle_ids: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if self.mask.shape != (5,):
            raise ValueError(
                f"mask must have shape (5,), got {self.mask.shape}"
            )

        if self.mask.dtype.kind not in "biuf":
            raise TypeError("mask must be numeric")

        if not np.all(np.isfinite(self.mask)):
            raise ValueError("mask contains NaN or infinite values")

        if np.any((self.mask != 0.0) & (self.mask != 1.0)):
            raise ValueError("mask must be binary")

        if len(self.scope_request_ids) != 5:
            raise ValueError("scope_request_ids must contain five entries")

        if len(self.scope_vehicle_ids) != 5:
            raise ValueError("scope_vehicle_ids must contain five entries")

        expected = tuple(
            INDEX_TO_ACTION[i]
            for i in range(5)
            if self.mask[i] == 1.0
        )

        if self.available_actions != expected:
            raise ValueError(
                "available_actions does not match mask"
            )

        if self.canonical_action is not None:
            if self.mask[
                ACTION_TO_INDEX[self.canonical_action]
            ] != 1.0:
                raise ValueError(
                    "canonical_action must be feasible"
                )

        self.mask.setflags(write=False)

    @property
    def action_count(self) -> int:
        return len(self.available_actions)

    @property
    def any_available(self) -> bool:
        return bool(self.available_actions)

    @property
    def all_available(self) -> bool:
        return self.action_count == 5

    def allows(self, action: ScopeAction | int) -> bool:
        """Return whether an action is currently selectable."""

        if isinstance(action, int):
            index = int(action)
            if index < 0 or index >= 5:
                raise ValueError(
                    f"action index must be in [0, 4], got {action}"
                )
        else:
            try:
                index = ACTION_TO_INDEX[action]
            except KeyError as exc:
                raise ValueError(
                    f"unsupported ScopeAction: {action!r}"
                ) from exc

        return bool(self.mask[index] == 1.0)

    def indices(self) -> tuple[int, ...]:
        """Return feasible action indices in canonical order."""

        return tuple(
            ACTION_TO_INDEX[action]
            for action in self.available_actions
        )

    def as_bool(self) -> np.ndarray:
        """Return a defensive boolean copy for masking-capable PPO code."""

        return self.mask.astype(bool, copy=True)

    def as_float(self) -> np.ndarray:
        """Return a defensive float32 copy for neural-policy interfaces."""

        return self.mask.astype(np.float32, copy=True)


@dataclass(frozen=True)
class ActionMaskConfig:
    """
    Configuration for structural action masking.

    ``mask_duplicate_scopes`` prevents multiple action labels from representing
    the same exact operational scope on tiny instances. The most specific
    action wins deterministically.

    ``require_affected_for_local`` follows the Step-15 definition: LOCAL is
    intended to operate on impact-ranked affected mutable jobs.

    ``require_affected_for_vehicle`` delegates to the existing Step-15
    selector configuration.
    """

    local_max_jobs: int = 10
    mask_duplicate_scopes: bool = False
    require_affected_for_local: bool = True
    require_affected_for_vehicle: bool = True

    def __post_init__(self) -> None:
        if self.local_max_jobs <= 0:
            raise ValueError(
                "local_max_jobs must be positive"
            )


class ActionFeasibilityMaskBuilder:
    """
    Build the five-action mask from the existing Step-15 scope selector.

    No future information is accepted by this API. Inputs must represent the
    currently visible/legally mutable decision state.
    """

    # Specificity priority used only when duplicate operational scopes occur.
    # Earlier actions are retained; broader duplicate labels are masked.
    _SPECIFICITY_ORDER: tuple[ScopeAction, ...] = (
        ScopeAction.LOCAL,
        ScopeAction.VEHICLE,
        ScopeAction.REGIONAL,
        ScopeAction.GLOBAL,
    )

    def __init__(
        self,
        *,
        config: ActionMaskConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ActionMaskConfig()
        )

        self.selector = ScopeActionSelector(
            config=ScopeActionConfig(
                local_max_jobs=self.config.local_max_jobs,
                require_affected_vehicle_for_vehicle=(
                    self.config.require_affected_for_vehicle
                ),
                # The mask answers feasibility; KEEP safety is handled
                # independently by the runtime safety layer.
                safety_override_to_global=False,
                keep_requires_no_affected_mutable_jobs=True,
            )
        )

    def build(
        self,
        *,
        jobs: Sequence[JobImpact],
        affected_vehicle_ids: Iterable[str] = (),
        affected_zone_ids: Iterable[str] = (),
        legally_mutable_request_ids: Iterable[str] | None = None,
        legally_mutable_vehicle_ids: Iterable[str] | None = None,
    ) -> ActionMask:
        """
        Build an explicit five-action feasibility mask.

        The selector is run independently for each action. This is deliberate:
        the mask is derived from the same scope semantics that the runtime will
        later execute, rather than duplicating those semantics in a second
        hand-written implementation.
        """

        jobs = tuple(jobs)
        affected_vehicle_ids = tuple(
            str(value)
            for value in affected_vehicle_ids
        )
        affected_zone_ids = tuple(
            str(value)
            for value in affected_zone_ids
        )

        selections = {}

        for action in ScopeAction:
            selections[action] = self.selector.select(
                action,
                jobs=jobs,
                affected_vehicle_ids=affected_vehicle_ids,
                affected_zone_ids=affected_zone_ids,
                legally_mutable_request_ids=(
                    legally_mutable_request_ids
                ),
                legally_mutable_vehicle_ids=(
                    legally_mutable_vehicle_ids
                ),
            )

        mask = np.zeros(
            5,
            dtype=np.float32,
        )

        # KEEP is feasible only when the current mutable work does not make
        # keeping the current assignment unsafe.
        keep_selection = selections[ScopeAction.KEEP]
        eligible = self._eligible_jobs(
            jobs=jobs,
            legally_mutable_request_ids=legally_mutable_request_ids,
            legally_mutable_vehicle_ids=legally_mutable_vehicle_ids,
        )
        affected_mutable = self._affected_mutable_jobs(
            eligible=eligible,
            affected_vehicle_ids=affected_vehicle_ids,
        )

        if not (
            self.selector.config.keep_requires_no_affected_mutable_jobs
            and affected_mutable
        ):
            mask[ACTION_TO_INDEX[ScopeAction.KEEP]] = 1.0

        # A non-KEEP action is structurally feasible when its defined scope
        # contains at least one mutable request.
        for action in self._SPECIFICITY_ORDER:
            selection = selections[action]
            if selection.request_ids:
                mask[ACTION_TO_INDEX[action]] = 1.0

        # If the configuration requests LOCAL to be genuinely incident-driven,
        # reject a non-empty LOCAL scope when no mutable work is affected.
        if self.config.require_affected_for_local:
            local = selections[ScopeAction.LOCAL]
            if local.request_ids and not affected_mutable:
                mask[ACTION_TO_INDEX[ScopeAction.LOCAL]] = 0.0

        # Duplicate operational scopes are redundant action labels. Retain the
        # first (most specific) action and mask later identical scopes.
        if self.config.mask_duplicate_scopes:
            self._mask_duplicate_scopes(
                mask=mask,
                selections=selections,
            )

        # Safety invariant: never return an all-zero mask.
        # GLOBAL is the final legal replanning escape hatch when mutable work
        # exists; KEEP is the only valid action when there is no mutable work.
        if not np.any(mask):
            if eligible:
                mask[ACTION_TO_INDEX[ScopeAction.GLOBAL]] = 1.0
            else:
                mask[ACTION_TO_INDEX[ScopeAction.KEEP]] = 1.0

        available_actions = tuple(
            INDEX_TO_ACTION[index]
            for index in range(5)
            if mask[index] == 1.0
        )

        canonical = self._canonical_action(
            mask=mask,
            selections=selections,
        )

        return ActionMask(
            mask=mask,
            available_actions=available_actions,
            canonical_action=canonical,
            scope_request_ids=tuple(
                tuple(selections[INDEX_TO_ACTION[i]].request_ids)
                for i in range(5)
            ),
            scope_vehicle_ids=tuple(
                tuple(selections[INDEX_TO_ACTION[i]].vehicle_ids)
                for i in range(5)
            ),
        )

    @staticmethod
    def _eligible_jobs(
        *,
        jobs: Sequence[JobImpact],
        legally_mutable_request_ids: Iterable[str] | None,
        legally_mutable_vehicle_ids: Iterable[str] | None,
    ) -> tuple[JobImpact, ...]:
        request_ids = (
            None
            if legally_mutable_request_ids is None
            else {str(value) for value in legally_mutable_request_ids}
        )
        vehicle_ids = (
            None
            if legally_mutable_vehicle_ids is None
            else {str(value) for value in legally_mutable_vehicle_ids}
        )

        result = []
        seen_requests: set[str] = set()

        for job in jobs:
            request_id = str(job.request_id)
            vehicle_id = str(job.vehicle_id)

            if request_ids is not None and request_id not in request_ids:
                continue

            if vehicle_ids is not None and vehicle_id not in vehicle_ids:
                continue

            if request_id in seen_requests:
                continue

            seen_requests.add(request_id)
            result.append(job)

        return tuple(result)

    @staticmethod
    def _affected_mutable_jobs(
        *,
        eligible: Sequence[JobImpact],
        affected_vehicle_ids: Iterable[str],
    ) -> tuple[JobImpact, ...]:
        affected_vehicles = {
            str(value)
            for value in affected_vehicle_ids
        }

        return tuple(
            job
            for job in eligible
            if job.affected
            or str(job.vehicle_id) in affected_vehicles
        )

    def _mask_duplicate_scopes(
        self,
        *,
        mask: np.ndarray,
        selections: dict[ScopeAction, object],
    ) -> None:
        seen: dict[
            tuple[tuple[str, ...], tuple[str, ...]],
            ScopeAction,
        ] = {}

        for action in self._SPECIFICITY_ORDER:
            index = ACTION_TO_INDEX[action]

            if mask[index] != 1.0:
                continue

            selection = selections[action]
            signature = (
                tuple(selection.request_ids),
                tuple(selection.vehicle_ids),
            )

            if signature[0] == ():
                continue

            if signature in seen:
                mask[index] = 0.0
            else:
                seen[signature] = action

    @staticmethod
    def _canonical_action(
        *,
        mask: np.ndarray,
        selections: dict[ScopeAction, object],
    ) -> ScopeAction | None:
        # KEEP is the canonical choice only when it is the sole feasible action.
        if (
            mask[ACTION_TO_INDEX[ScopeAction.KEEP]] == 1.0
            and np.sum(mask) == 1
        ):
            return ScopeAction.KEEP

        for action in (
            ScopeAction.LOCAL,
            ScopeAction.VEHICLE,
            ScopeAction.REGIONAL,
            ScopeAction.GLOBAL,
        ):
            if mask[ACTION_TO_INDEX[action]] == 1.0:
                return action

        return None


def validate_action_index(
    action_index: int,
    mask: ActionMask,
) -> ScopeAction:
    """
    Convert and validate a sampled PPO action.

    This is intentionally separate from the neural-policy implementation.
    A policy implementation must still apply the mask before sampling.
    """

    if not isinstance(action_index, (int, np.integer)):
        raise TypeError(
            "action_index must be an integer"
        )

    action_index = int(action_index)

    if action_index < 0 or action_index >= 5:
        raise ValueError(
            f"action_index must be in [0, 4], got {action_index}"
        )

    action = INDEX_TO_ACTION[action_index]

    if not mask.allows(action):
        raise ValueError(
            f"action {action.value!r} is masked as infeasible"
        )

    return action
