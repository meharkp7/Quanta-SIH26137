"""
Step 16.8 — Causal decision/replanning memory.

This module stores the controller's recent operational decisions and derives
causal replanning-context features from those decisions.

It is intentionally separate from TemporalMemory:

    TemporalMemory
        = what the observed system state looked like over time

    DecisionMemory
        = what the controller requested/executed and how its replanning scope
          evolved over time

Design requirements:
    - causal only
    - bounded history
    - deterministic ordering
    - explicit padding mask
    - immutable records
    - defensive copies
    - strict decision-time monotonicity
    - strict decision-version monotonicity where numeric
    - scenario/episode identity consistency
    - exact requested vs executed action preservation
    - explicit override provenance
    - deterministic scope-churn metrics
    - no reward or future simulator truth
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np

from src.contracts.decision import ScopeDecision


# ============================================================================
# Feature contract
# ============================================================================

DECISION_MEMORY_FEATURE_NAMES: tuple[str, ...] = (
    "has_decision",
    "requested_keep",
    "requested_local",
    "requested_vehicle",
    "requested_regional",
    "requested_global",
    "executed_keep",
    "executed_local",
    "executed_vehicle",
    "executed_regional",
    "executed_global",
    "was_overridden",
    "affected_vehicle_fraction",
    "mutable_request_fraction",
    "budget_normalized",
    "time_since_previous_decision_normalized",
    "requested_action_changed",
    "executed_action_changed",
    "mutable_request_overlap",
    "affected_vehicle_overlap",
    "mutable_scope_changed",
    "decision_sequence_fraction",
)


# ============================================================================
# Immutable record
# ============================================================================


@dataclass(frozen=True)
class DecisionMemoryRecord:
    """
    One immutable causal decision record.

    All values describe information available when the decision was made.
    """

    scenario_id: str
    state_version: str

    decision_time_s: float

    requested_action: str
    executed_action: str

    affected_vehicle_ids: tuple[str, ...]
    mutable_request_ids: tuple[str, ...]

    budget_seconds: float

    overridden: bool
    override_reason: str | None

    selection_reason: str
    selected_by: str

    decision_version: str
    decision_id: str | None

    # Derived causal quantities.
    time_since_previous_decision_s: float = 0.0
    requested_action_changed: bool = False
    executed_action_changed: bool = False
    mutable_request_overlap: float = 0.0
    affected_vehicle_overlap: float = 0.0
    mutable_scope_changed: bool = False

    sequence_index: int = 0

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")

        if not self.state_version:
            raise ValueError("state_version must be non-empty")

        if not isfinite(float(self.decision_time_s)):
            raise ValueError("decision_time_s must be finite")

        if self.decision_time_s < 0:
            raise ValueError("decision_time_s must be non-negative")

        if not self.requested_action:
            raise ValueError("requested_action must be non-empty")

        if not self.executed_action:
            raise ValueError("executed_action must be non-empty")

        if not isfinite(float(self.budget_seconds)):
            raise ValueError("budget_seconds must be finite")

        if self.budget_seconds < 0:
            raise ValueError("budget_seconds must be non-negative")

        if self.time_since_previous_decision_s < 0:
            raise ValueError(
                "time_since_previous_decision_s must be non-negative"
            )

        for name, value in (
            ("mutable_request_overlap", self.mutable_request_overlap),
            ("affected_vehicle_overlap", self.affected_vehicle_overlap),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

        if self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative")

        if len(self.affected_vehicle_ids) != len(
            set(self.affected_vehicle_ids)
        ):
            raise ValueError("duplicate affected vehicle IDs")

        if len(self.mutable_request_ids) != len(
            set(self.mutable_request_ids)
        ):
            raise ValueError("duplicate mutable request IDs")


# ============================================================================
# Fixed-size batch
# ============================================================================


@dataclass(frozen=True)
class DecisionMemoryBatch:
    """
    Fixed-length oldest -> newest decision history.

    Padding is explicit and appears on the left.
    """

    records: tuple[DecisionMemoryRecord | None, ...]

    valid_mask: np.ndarray
    decision_times_s: np.ndarray
    features: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.records)

        if self.valid_mask.shape != (count,):
            raise ValueError(
                f"valid_mask must have shape {(count,)}; "
                f"got {self.valid_mask.shape}"
            )

        if self.decision_times_s.shape != (count,):
            raise ValueError(
                f"decision_times_s must have shape {(count,)}; "
                f"got {self.decision_times_s.shape}"
            )

        expected_shape = (
            count,
            len(DECISION_MEMORY_FEATURE_NAMES),
        )

        if self.features.shape != expected_shape:
            raise ValueError(
                f"features must have shape {expected_shape}; "
                f"got {self.features.shape}"
            )

        for name, array in (
            ("valid_mask", self.valid_mask),
            ("decision_times_s", self.decision_times_s),
            ("features", self.features),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or infinite values")

        if np.any((self.valid_mask < 0.0) | (self.valid_mask > 1.0)):
            raise ValueError("valid_mask must be binary")

        self.valid_mask.setflags(write=False)
        self.decision_times_s.setflags(write=False)
        self.features.setflags(write=False)

    @property
    def length(self) -> int:
        return len(self.records)

    @property
    def valid_count(self) -> int:
        return int(np.sum(self.valid_mask))

    @property
    def padding_count(self) -> int:
        return self.length - self.valid_count

    @property
    def latest(self) -> DecisionMemoryRecord | None:
        if not self.records:
            return None

        return self.records[-1]


# ============================================================================
# Decision memory
# ============================================================================


class DecisionMemory:
    """
    Bounded causal history of ScopeDecision objects.

    The memory belongs to one episode. A reset is required before changing
    episode/scenario identity.
    """

    def __init__(
        self,
        *,
        history_length: int = 12,
        max_budget_seconds: float = 60.0,
        max_decision_interval_seconds: float = 300.0,
    ) -> None:
        if int(history_length) <= 0:
            raise ValueError("history_length must be positive")

        if max_budget_seconds <= 0:
            raise ValueError("max_budget_seconds must be positive")

        if max_decision_interval_seconds <= 0:
            raise ValueError(
                "max_decision_interval_seconds must be positive"
            )

        self.history_length = int(history_length)
        self.max_budget_seconds = float(max_budget_seconds)
        self.max_decision_interval_seconds = float(
            max_decision_interval_seconds
        )

        self._records: list[DecisionMemoryRecord] = []

        self._scenario_id: str | None = None
        self._episode_id: str | None = None

        self._last_decision_time_s: float | None = None
        self._last_numeric_decision_version: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        episode_id: str | None = None,
        scenario_id: str | None = None,
    ) -> None:
        """Clear the memory and optionally establish episode identity."""

        self._records.clear()

        self._scenario_id = (
            str(scenario_id)
            if scenario_id is not None
            else None
        )

        self._episode_id = (
            str(episode_id)
            if episode_id is not None
            else None
        )

        self._last_decision_time_s = None
        self._last_numeric_decision_version = None

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def is_empty(self) -> bool:
        return not self._records

    @property
    def scenario_id(self) -> str | None:
        return self._scenario_id

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        decision: ScopeDecision,
    ) -> DecisionMemoryRecord:
        """
        Append one causal routing decision.

        The decision is copied into an immutable internal representation.
        No reward, realized route outcome, or future simulator state is
        accepted by this interface.
        """

        if not isinstance(decision, ScopeDecision):
            raise TypeError(
                "decision must be a ScopeDecision"
            )

        scenario_id = str(decision.scenario_id)
        decision_time_s = float(decision.decision_time_s)

        self._validate_identity(scenario_id)
        self._validate_time(decision_time_s)
        self._validate_decision_version(decision.decision_version)

        previous = self._records[-1] if self._records else None

        affected_vehicle_ids = _unique_strings(
            decision.affected_vehicle_ids
        )
        mutable_request_ids = _unique_strings(
            decision.mutable_request_ids
        )

        time_since_previous = (
            0.0
            if previous is None
            else decision_time_s - previous.decision_time_s
        )

        requested_changed = (
            False
            if previous is None
            else (
                _action_value(decision.requested_action)
                != previous.requested_action
            )
        )

        executed_changed = (
            False
            if previous is None
            else (
                _action_value(decision.executed_action)
                != previous.executed_action
            )
        )

        request_overlap = (
            0.0
            if previous is None
            else _jaccard(
                mutable_request_ids,
                previous.mutable_request_ids,
            )
        )

        vehicle_overlap = (
            0.0
            if previous is None
            else _jaccard(
                affected_vehicle_ids,
                previous.affected_vehicle_ids,
            )
        )

        scope_changed = (
            False
            if previous is None
            else (
                mutable_request_ids
                != previous.mutable_request_ids
                or affected_vehicle_ids
                != previous.affected_vehicle_ids
            )
        )

        record = DecisionMemoryRecord(
            scenario_id=scenario_id,
            state_version=str(decision.state_version),
            decision_time_s=decision_time_s,
            requested_action=_action_value(decision.requested_action),
            executed_action=_action_value(decision.executed_action),
            affected_vehicle_ids=affected_vehicle_ids,
            mutable_request_ids=mutable_request_ids,
            budget_seconds=float(decision.budget_seconds),
            overridden=bool(decision.overridden),
            override_reason=(
                None
                if decision.override_reason is None
                else str(decision.override_reason)
            ),
            selection_reason=str(decision.selection_reason),
            selected_by=str(decision.selected_by),
            decision_version=str(decision.decision_version),
            decision_id=(
                None
                if decision.decision_id is None
                else str(decision.decision_id)
            ),
            time_since_previous_decision_s=time_since_previous,
            requested_action_changed=requested_changed,
            executed_action_changed=executed_changed,
            mutable_request_overlap=request_overlap,
            affected_vehicle_overlap=vehicle_overlap,
            mutable_scope_changed=scope_changed,
            sequence_index=(
                0
                if previous is None
                else previous.sequence_index + 1
            ),
        )

        self._records.append(record)

        if len(self._records) > self.history_length:
            del self._records[
                : len(self._records) - self.history_length
            ]

        self._last_decision_time_s = decision_time_s

        numeric_version = _numeric_version(decision.decision_version)

        if numeric_version is not None:
            self._last_numeric_decision_version = numeric_version

        return record

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def records(self) -> tuple[DecisionMemoryRecord, ...]:
        """Return immutable oldest -> newest history."""

        return tuple(self._records)

    def as_batch(self) -> DecisionMemoryBatch:
        """
        Return a fixed-size left-padded batch.
        """

        padding_count = self.history_length - len(self._records)

        records: tuple[DecisionMemoryRecord | None, ...] = (
            (None,) * padding_count
            + tuple(self._records)
        )

        valid_mask = np.asarray(
            [
                0.0 if record is None else 1.0
                for record in records
            ],
            dtype=np.float32,
        )

        decision_times = np.asarray(
            [
                0.0 if record is None
                else record.decision_time_s
                for record in records
            ],
            dtype=np.float64,
        )

        features = np.asarray(
            [
                (
                    np.zeros(
                        len(DECISION_MEMORY_FEATURE_NAMES),
                        dtype=np.float32,
                    )
                    if record is None
                    else self._record_features(record)
                )
                for record in records
            ],
            dtype=np.float32,
        )

        return DecisionMemoryBatch(
            records=records,
            valid_mask=valid_mask,
            decision_times_s=decision_times,
            features=features,
        )

    # ------------------------------------------------------------------
    # Feature encoding
    # ------------------------------------------------------------------

    def _record_features(
        self,
        record: DecisionMemoryRecord,
    ) -> np.ndarray:
        requested = _action_one_hot(record.requested_action)
        executed = _action_one_hot(record.executed_action)

        # Counts are normalized against conservative fixed scales rather
        # than the current scenario size so the feature space is stable.
        affected_vehicle_fraction = float(
            np.clip(
                len(record.affected_vehicle_ids) / 10.0,
                0.0,
                1.0,
            )
        )

        mutable_request_fraction = float(
            np.clip(
                len(record.mutable_request_ids) / 50.0,
                0.0,
                1.0,
            )
        )

        budget_normalized = float(
            np.clip(
                record.budget_seconds / self.max_budget_seconds,
                0.0,
                10.0,
            )
        )

        interval_normalized = float(
            np.clip(
                record.time_since_previous_decision_s
                / self.max_decision_interval_seconds,
                0.0,
                10.0,
            )
        )

        sequence_fraction = float(
            np.clip(
                record.sequence_index
                / max(self.history_length - 1, 1),
                0.0,
                1.0,
            )
        )

        result = np.asarray(
            [
                1.0,
                *requested,
                *executed,
                float(record.overridden),
                affected_vehicle_fraction,
                mutable_request_fraction,
                budget_normalized,
                interval_normalized,
                float(record.requested_action_changed),
                float(record.executed_action_changed),
                record.mutable_request_overlap,
                record.affected_vehicle_overlap,
                float(record.mutable_scope_changed),
                sequence_fraction,
            ],
            dtype=np.float32,
        )

        result.setflags(write=False)
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_identity(self, scenario_id: str) -> None:
        if self._scenario_id is None:
            self._scenario_id = scenario_id
        elif scenario_id != self._scenario_id:
            raise ValueError(
                "scenario_id changed without reset; "
                f"previous={self._scenario_id!r}, "
                f"current={scenario_id!r}"
            )

    def _validate_time(self, decision_time_s: float) -> None:
        if not isfinite(decision_time_s):
            raise ValueError("decision_time_s must be finite")

        if decision_time_s < 0:
            raise ValueError(
                "decision_time_s must be non-negative"
            )

        if (
            self._last_decision_time_s is not None
            and decision_time_s <= self._last_decision_time_s
        ):
            raise ValueError(
                "decision_time_s must strictly increase; "
                f"previous={self._last_decision_time_s}, "
                f"current={decision_time_s}"
            )

    def _validate_decision_version(
        self,
        decision_version: str,
    ) -> None:
        version = _numeric_version(decision_version)

        if version is None:
            return

        if (
            self._last_numeric_decision_version is not None
            and version <= self._last_numeric_decision_version
        ):
            raise ValueError(
                "numeric decision_version must strictly increase; "
                f"previous={self._last_numeric_decision_version}, "
                f"current={version}"
            )


# ============================================================================
# Helpers
# ============================================================================


def _unique_strings(
    values: Iterable[object],
) -> tuple[str, ...]:
    """Deduplicate while preserving first-seen deterministic order."""

    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        value = str(value)

        if value in seen:
            continue

        seen.add(value)
        result.append(value)

    return tuple(result)


def _jaccard(
    left: Iterable[str],
    right: Iterable[str],
) -> float:
    """
    Jaccard similarity between two immutable ID collections.

    Two empty scopes are considered identical and therefore receive 1.0.
    """

    left_set = set(left)
    right_set = set(right)

    if not left_set and not right_set:
        return 1.0

    union = left_set | right_set

    if not union:
        return 1.0

    return float(len(left_set & right_set) / len(union))


def _numeric_version(value: object) -> int | None:
    """
    Extract a numeric decision version when possible.

    Examples:
        "1"       -> 1
        "decision-3" -> None

    Non-numeric repository version strings remain valid; their ordering is
    already represented by the strictly increasing decision timestamps.
    """

    text = str(value).strip()

    if not text:
        return None

    if text.isdigit():
        return int(text)

    return None

def _action_value(action: object) -> str:
    """
    Return the canonical string value of a ScopeAction.

    Supports enum instances as well as already-normalized strings.
    """
    value = getattr(action, "value", action)
    return str(value).strip().upper()

def _action_one_hot(action: str | None) -> tuple[float, ...]:
    normalized = (
        str(action).strip().upper()
        if action is not None
        else None
    )

    return (
        float(normalized == "KEEP"),
        float(normalized == "LOCAL"),
        float(normalized == "VEHICLE"),
        float(normalized == "REGIONAL"),
        float(normalized == "GLOBAL"),
    )