"""
Step 16.7 — Causal temporal memory.

Maintains a bounded, immutable history of state representations for one
episode. This is a temporal DATA CONTRACT, not a Transformer.

The memory stores:
    - GraphState
    - MultiScaleTrafficState
    - EventState
    - ServiceRiskState
    - observation timestamp
    - state version
    - recent decision context

Design requirements:
    - causal only
    - fixed maximum history
    - deterministic ordering
    - explicit padding mask
    - explicit time deltas
    - strict timestamp monotonicity
    - strict state-version monotonicity
    - immutable snapshots
    - deterministic reset
    - no future simulator truth
    - no mutation of previously stored states

A later Transformer can consume the resulting TemporalBatch without needing
to know anything about the runtime history-management rules.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


# ============================================================================
# Decision context
# ============================================================================

DECISION_FEATURE_NAMES: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class TemporalDecisionContext:
    """
    Compact causal representation of the most recent routing decision.

    The object deliberately stores only decision metadata. It does not store
    optimizer internals, future outcomes, rewards, or simulator truth.
    """

    requested_action: str | None = None
    executed_action: str | None = None

    affected_vehicle_count: int = 0
    total_vehicle_count: int = 0

    mutable_request_count: int = 0
    total_request_count: int = 0

    budget_seconds: float = 0.0
    overridden: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("affected_vehicle_count", self.affected_vehicle_count),
            ("total_vehicle_count", self.total_vehicle_count),
            ("mutable_request_count", self.mutable_request_count),
            ("total_request_count", self.total_request_count),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.affected_vehicle_count > self.total_vehicle_count:
            raise ValueError(
                "affected_vehicle_count cannot exceed total_vehicle_count"
            )

        if self.mutable_request_count > self.total_request_count:
            raise ValueError(
                "mutable_request_count cannot exceed total_request_count"
            )

        if not isfinite(float(self.budget_seconds)):
            raise ValueError("budget_seconds must be finite")

        if self.budget_seconds < 0:
            raise ValueError("budget_seconds must be non-negative")

        for action in (
            self.requested_action,
            self.executed_action,
        ):
            if action is not None and not str(action).strip():
                raise ValueError("action cannot be an empty string")

    @classmethod
    def empty(cls) -> "TemporalDecisionContext":
        return cls()

    @property
    def has_decision(self) -> bool:
        return (
            self.requested_action is not None
            or self.executed_action is not None
        )

    def as_features(
        self,
        *,
        max_budget_seconds: float = 60.0,
    ) -> np.ndarray:
        """
        Convert the decision context into a deterministic numeric vector.

        This vector is intentionally small. The detailed routing state lives
        in the other temporal components.
        """

        if max_budget_seconds <= 0:
            raise ValueError("max_budget_seconds must be positive")

        requested = _action_one_hot(self.requested_action)
        executed = _action_one_hot(self.executed_action)

        vehicle_fraction = (
            self.affected_vehicle_count
            / max(self.total_vehicle_count, 1)
        )

        request_fraction = (
            self.mutable_request_count
            / max(self.total_request_count, 1)
        )

        result = np.asarray(
            [
                float(self.has_decision),
                *requested,
                *executed,
                float(self.overridden),
                float(np.clip(vehicle_fraction, 0.0, 1.0)),
                float(np.clip(request_fraction, 0.0, 1.0)),
                float(
                    np.clip(
                        self.budget_seconds / max_budget_seconds,
                        0.0,
                        10.0,
                    )
                ),
            ],
            dtype=np.float32,
        )

        result.setflags(write=False)
        return result


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


# ============================================================================
# Temporal snapshot
# ============================================================================


@dataclass(frozen=True)
class TemporalSnapshot:
    """
    One immutable point-in-time state.

    All component objects are copied before insertion so later caller-side
    mutation cannot alter the temporal history.
    """

    observation_time_s: float
    state_version: int

    graph_state: Any
    multiscale_traffic_state: Any
    event_state: Any
    service_risk_state: Any

    decision_context: TemporalDecisionContext

    delta_time_s: float = 0.0
    is_padding: bool = False

    def __post_init__(self) -> None:
        if not isfinite(float(self.observation_time_s)):
            raise ValueError("observation_time_s must be finite")

        if self.observation_time_s < 0:
            raise ValueError(
                "observation_time_s must be non-negative"
            )

        if int(self.state_version) < 0:
            raise ValueError("state_version must be non-negative")

        if not isfinite(float(self.delta_time_s)):
            raise ValueError("delta_time_s must be finite")

        if self.delta_time_s < 0:
            raise ValueError("delta_time_s must be non-negative")

        if self.is_padding and (
            self.graph_state is not None
            or self.multiscale_traffic_state is not None
            or self.event_state is not None
            or self.service_risk_state is not None
        ):
            raise ValueError(
                "padding snapshot cannot contain state objects"
            )

    @classmethod
    def padding(cls) -> "TemporalSnapshot":
        """
        Explicit padding entry.

        Padding has no timestamp or state semantics. The zero values exist
        solely so a fixed-size tensor batch can be constructed.
        """

        return cls(
            observation_time_s=0.0,
            state_version=0,
            graph_state=None,
            multiscale_traffic_state=None,
            event_state=None,
            service_risk_state=None,
            decision_context=TemporalDecisionContext.empty(),
            delta_time_s=0.0,
            is_padding=True,
        )


# ============================================================================
# Temporal batch
# ============================================================================


@dataclass(frozen=True)
class TemporalBatch:
    """
    Fixed-length causal history ready for a temporal model.

    Entries are ordered oldest -> newest.

    valid_mask:
        1 for real observations
        0 for explicit left-padding

    delta_time_s:
        elapsed time since the previous REAL observation.
        Padding entries are zero.
    """

    snapshots: tuple[TemporalSnapshot, ...]

    valid_mask: np.ndarray
    observation_times_s: np.ndarray
    state_versions: np.ndarray
    delta_time_s: np.ndarray

    decision_features: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.snapshots)

        for name, array in (
            ("valid_mask", self.valid_mask),
            ("observation_times_s", self.observation_times_s),
            ("state_versions", self.state_versions),
            ("delta_time_s", self.delta_time_s),
        ):
            if array.shape != (count,):
                raise ValueError(
                    f"{name} must have shape {(count,)}; "
                    f"got {array.shape}"
                )

        expected_decision_shape = (
            count,
            len(DECISION_FEATURE_NAMES),
        )

        if self.decision_features.shape != expected_decision_shape:
            raise ValueError(
                "decision_features has shape "
                f"{self.decision_features.shape}; expected "
                f"{expected_decision_shape}"
            )

        for name, array in (
            ("valid_mask", self.valid_mask),
            ("observation_times_s", self.observation_times_s),
            ("state_versions", self.state_versions),
            ("delta_time_s", self.delta_time_s),
            ("decision_features", self.decision_features),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"{name} contains NaN or infinite values"
                )

        if np.any(
            (self.valid_mask < 0)
            | (self.valid_mask > 1)
        ):
            raise ValueError("valid_mask must be binary")

        self.valid_mask.setflags(write=False)
        self.observation_times_s.setflags(write=False)
        self.state_versions.setflags(write=False)
        self.delta_time_s.setflags(write=False)
        self.decision_features.setflags(write=False)

    @property
    def length(self) -> int:
        return len(self.snapshots)

    @property
    def valid_count(self) -> int:
        return int(np.sum(self.valid_mask))

    @property
    def padding_count(self) -> int:
        return self.length - self.valid_count

    @property
    def latest(self) -> TemporalSnapshot | None:
        if not self.snapshots:
            return None

        snapshot = self.snapshots[-1]
        return None if snapshot.is_padding else snapshot


# ============================================================================
# Temporal memory
# ============================================================================


class TemporalMemory:
    """
    Bounded causal history for one episode.

    Example:

        memory = TemporalMemory(history_length=12)

        memory.append(
            graph_state=graph_state,
            multiscale_traffic_state=traffic_state,
            event_state=event_state,
            service_risk_state=service_state,
            observation_time_s=60.0,
            state_version=1,
        )

        batch = memory.as_batch()

    The oldest observation is discarded once capacity is reached.
    """

    def __init__(
        self,
        *,
        history_length: int = 12,
        max_budget_seconds: float = 60.0,
    ) -> None:
        if int(history_length) <= 0:
            raise ValueError("history_length must be positive")

        if max_budget_seconds <= 0:
            raise ValueError(
                "max_budget_seconds must be positive"
            )

        self.history_length = int(history_length)
        self.max_budget_seconds = float(max_budget_seconds)

        self._snapshots: list[TemporalSnapshot] = []

        self._episode_id: str | None = None
        self._scenario_id: str | None = None
        self._graph_version: str | None = None

        self._last_time_s: float | None = None
        self._last_state_version: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        episode_id: str | None = None,
        scenario_id: str | None = None,
        graph_version: str | None = None,
    ) -> None:
        """
        Clear the history and optionally establish episode identity.

        Reset is the only supported way to start a new episode.
        """

        self._snapshots.clear()

        self._episode_id = (
            str(episode_id)
            if episode_id is not None
            else None
        )

        self._scenario_id = (
            str(scenario_id)
            if scenario_id is not None
            else None
        )

        self._graph_version = (
            str(graph_version)
            if graph_version is not None
            else None
        )

        self._last_time_s = None
        self._last_state_version = None

    @property
    def size(self) -> int:
        return len(self._snapshots)

    @property
    def is_empty(self) -> bool:
        return not self._snapshots

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    @property
    def scenario_id(self) -> str | None:
        return self._scenario_id

    @property
    def graph_version(self) -> str | None:
        return self._graph_version

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        graph_state: Any,
        multiscale_traffic_state: Any,
        event_state: Any,
        service_risk_state: Any,
        observation_time_s: float,
        state_version: int,
        decision_context: TemporalDecisionContext | None = None,
    ) -> TemporalSnapshot:
        """
        Append one causal observation.

        The method requires monotonically increasing observation time and
        state version. Replays and time-travel are rejected.
        """

        time_s = float(observation_time_s)
        version = int(state_version)

        self._validate_time(time_s)
        self._validate_state_version(version)

        self._validate_component_identity(
            graph_state=graph_state,
            multiscale_traffic_state=multiscale_traffic_state,
            event_state=event_state,
            service_risk_state=service_risk_state,
        )

        if decision_context is None:
            decision_context = TemporalDecisionContext.empty()

        if not isinstance(
            decision_context,
            TemporalDecisionContext,
        ):
            raise TypeError(
                "decision_context must be TemporalDecisionContext"
            )

        delta = (
            0.0
            if self._last_time_s is None
            else time_s - self._last_time_s
        )

        snapshot = TemporalSnapshot(
            observation_time_s=time_s,
            state_version=version,
            graph_state=_immutable_copy(graph_state),
            multiscale_traffic_state=_immutable_copy(
                multiscale_traffic_state
            ),
            event_state=_immutable_copy(event_state),
            service_risk_state=_immutable_copy(
                service_risk_state
            ),
            decision_context=deepcopy(decision_context),
            delta_time_s=delta,
            is_padding=False,
        )

        self._snapshots.append(snapshot)

        if len(self._snapshots) > self.history_length:
            del self._snapshots[
                : len(self._snapshots) - self.history_length
            ]

        self._last_time_s = time_s
        self._last_state_version = version

        return snapshot

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def snapshots(self) -> tuple[TemporalSnapshot, ...]:
        """
        Return the causal history oldest -> newest.

        The returned tuple cannot mutate the memory itself.
        """

        return tuple(self._snapshots)

    def as_batch(self) -> TemporalBatch:
        """
        Return a fixed-length left-padded temporal batch.

        If history_length=12 and only 3 observations exist:

            [PAD, PAD, ..., PAD, obs1, obs2, obs3]

        This avoids pretending that nonexistent observations are real data.
        """

        padding_count = self.history_length - len(self._snapshots)

        snapshots = (
            tuple(
                TemporalSnapshot.padding()
                for _ in range(padding_count)
            )
            + tuple(self._snapshots)
        )

        valid_mask = np.asarray(
            [
                0.0 if snapshot.is_padding else 1.0
                for snapshot in snapshots
            ],
            dtype=np.float32,
        )

        observation_times = np.asarray(
            [
                0.0 if snapshot.is_padding
                else snapshot.observation_time_s
                for snapshot in snapshots
            ],
            dtype=np.float64,
        )

        state_versions = np.asarray(
            [
                0 if snapshot.is_padding
                else snapshot.state_version
                for snapshot in snapshots
            ],
            dtype=np.int64,
        )

        delta_times = np.asarray(
            [
                0.0 if snapshot.is_padding
                else snapshot.delta_time_s
                for snapshot in snapshots
            ],
            dtype=np.float64,
        )

        decision_features = np.asarray(
            [
                np.zeros(
                    len(DECISION_FEATURE_NAMES),
                    dtype=np.float32,
                )
                if snapshot.is_padding
                else snapshot.decision_context.as_features(
                    max_budget_seconds=self.max_budget_seconds
                )
                for snapshot in snapshots
            ],
            dtype=np.float32,
        )

        return TemporalBatch(
            snapshots=snapshots,
            valid_mask=valid_mask,
            observation_times_s=observation_times,
            state_versions=state_versions,
            delta_time_s=delta_times,
            decision_features=decision_features,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_time(self, time_s: float) -> None:
        if not isfinite(time_s):
            raise ValueError(
                "observation_time_s must be finite"
            )

        if time_s < 0:
            raise ValueError(
                "observation_time_s must be non-negative"
            )

        if self._last_time_s is not None and time_s <= self._last_time_s:
            raise ValueError(
                "observation_time_s must strictly increase; "
                f"previous={self._last_time_s}, current={time_s}"
            )

    def _validate_state_version(self, version: int) -> None:
        if version < 0:
            raise ValueError(
                "state_version must be non-negative"
            )

        if (
            self._last_state_version is not None
            and version <= self._last_state_version
        ):
            raise ValueError(
                "state_version must strictly increase; "
                f"previous={self._last_state_version}, "
                f"current={version}"
            )

    def _validate_component_identity(
        self,
        *,
        graph_state: Any,
        multiscale_traffic_state: Any,
        event_state: Any,
        service_risk_state: Any,
    ) -> None:
        """
        Verify that all state components belong to the same scenario/graph.

        State objects from the current Step 16 layers expose these fields.
        """

        if graph_state is None:
            raise ValueError("graph_state cannot be None")

        if multiscale_traffic_state is None:
            raise ValueError(
                "multiscale_traffic_state cannot be None"
            )

        if event_state is None:
            raise ValueError("event_state cannot be None")

        if service_risk_state is None:
            raise ValueError(
                "service_risk_state cannot be None"
            )

        scenario_id = getattr(
            graph_state,
            "scenario_id",
            None,
        )

        graph_version = getattr(
            graph_state,
            "graph_version",
            None,
        )

        if scenario_id is not None:
            scenario_id = str(scenario_id)

            if (
                self._scenario_id is not None
                and scenario_id != self._scenario_id
            ):
                raise ValueError(
                    "graph_state scenario_id changed within temporal memory"
                )

            if self._scenario_id is None:
                self._scenario_id = scenario_id

        if graph_version is not None:
            graph_version = str(graph_version)

            if (
                self._graph_version is not None
                and graph_version != self._graph_version
            ):
                raise ValueError(
                    "graph_version changed within temporal memory"
                )

            if self._graph_version is None:
                self._graph_version = graph_version

        # Multi-scale state must refer to the same graph node universe.
        graph_node_ids = getattr(
            graph_state,
            "node_ids",
            None,
        )

        traffic_node_ids = getattr(
            multiscale_traffic_state,
            "node_ids",
            None,
        )

        if (
            graph_node_ids is not None
            and traffic_node_ids is not None
            and tuple(graph_node_ids) != tuple(traffic_node_ids)
        ):
            raise ValueError(
                "GraphState and MultiScaleTrafficState node ordering differ"
            )

    # ------------------------------------------------------------------
    # Debug / inspection
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return deterministic metadata for logging and tests."""

        return {
            "history_length": self.history_length,
            "current_size": len(self._snapshots),
            "padding_count": self.history_length - len(self._snapshots),
            "episode_id": self._episode_id,
            "scenario_id": self._scenario_id,
            "graph_version": self._graph_version,
            "last_time_s": self._last_time_s,
            "last_state_version": self._last_state_version,
        }


# ============================================================================
# Immutability helpers
# ============================================================================


def _immutable_copy(value: Any) -> Any:
    """
    Deep-copy a state object and make every numpy array read-only.

    This is intentionally generic so the temporal layer does not become
    coupled to the implementation details of GraphState/EventState/etc.
    """

    copied = deepcopy(value)
    _freeze_numpy_arrays(copied)
    return copied


def _freeze_numpy_arrays(value: Any, *, _seen: set[int] | None = None) -> None:
    """
    Recursively make numpy arrays read-only.

    Existing Step 16 state classes already freeze their arrays, but temporal
    memory must defend against future state implementations as well.
    """

    if _seen is None:
        _seen = set()

    object_id = id(value)
    if object_id in _seen:
        return

    _seen.add(object_id)

    if isinstance(value, np.ndarray):
        value.setflags(write=False)
        return

    if isinstance(value, Mapping):
        for item in value.values():
            _freeze_numpy_arrays(item, _seen=_seen)
        return

    if isinstance(value, (tuple, list)):
        for item in value:
            _freeze_numpy_arrays(item, _seen=_seen)
        return

    if hasattr(value, "__dict__"):
        for item in vars(value).values():
            _freeze_numpy_arrays(item, _seen=_seen)


# ============================================================================
# Decision-context adapter
# ============================================================================


def decision_context_from_scope(
    scope_decision: Any,
    *,
    total_vehicle_count: int,
    total_request_count: int,
) -> TemporalDecisionContext:
    """
    Convert the existing ScopeDecision contract into temporal memory context.

    Kept as an adapter rather than coupling TemporalMemory itself to the
    routing package.
    """

    if scope_decision is None:
        return TemporalDecisionContext.empty()

    return TemporalDecisionContext(
        requested_action=_optional_string(
            getattr(scope_decision, "requested_action", None)
        ),
        executed_action=_optional_string(
            getattr(scope_decision, "executed_action", None)
        ),
        affected_vehicle_count=len(
            tuple(
                getattr(
                    scope_decision,
                    "affected_vehicle_ids",
                    (),
                )
            )
        ),
        total_vehicle_count=int(total_vehicle_count),
        mutable_request_count=len(
            tuple(
                getattr(
                    scope_decision,
                    "mutable_request_ids",
                    (),
                )
            )
        ),
        total_request_count=int(total_request_count),
        budget_seconds=float(
            getattr(
                scope_decision,
                "budget_seconds",
                0.0,
            )
        ),
        overridden=bool(
            getattr(
                scope_decision,
                "overridden",
                False,
            )
        ),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)