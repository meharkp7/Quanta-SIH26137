"""
Step 17.1 — Structured causal policy-state adapter.

This module connects the Step-16 state contracts to the PPO representation
boundary.

Important architectural rules:

    Observation
        -> GraphState
        -> MultiScaleTrafficState
        -> EventState
        -> ServiceRiskState
        -> TemporalMemory
        -> DecisionMemory
        -> FusedState
        -> PolicyState
        -> RepresentationProvider
        -> PPO

This module does NOT:
    - train the forecaster
    - run PPO
    - call QPSO
    - execute SUMO
    - consume future simulator truth

Forecast and uncertainty are supplied explicitly because they are separate
causal inputs to the PPO representation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from src.contracts.observation import Observation

from .representation import (
    RepresentationOutput,
    RepresentationSpec,
)
from .state.decision_memory import (
    DecisionMemory,
    DecisionMemoryBatch,
)
from .state.event_state import EventStateBuilder
from .state.graph_state import GraphStateBuilder
from .state.multiscale_traffic import MultiScaleTrafficBuilder
from .state.service_risk import ServiceRiskBuilder
from .state.state_fusion import (
    FusedState,
    StateFusionBuilder,
)
from .state.temporal_memory import (
    TemporalDecisionContext,
    TemporalMemory,
)


@dataclass(frozen=True)
class PPOPolicyState:
    """
    Complete causal state presented to the PPO representation adapter.

    `fused_state` contains the structured Step-16 environment state.

    `forecast_features` and `uncertainty_features` are explicit because
    forecasting is an independent pipeline and must never be silently
    reconstructed from future information.
    """

    fused_state: FusedState

    forecast_features: np.ndarray
    uncertainty_features: np.ndarray

    forecast_version: str
    uncertainty_version: str

    def __post_init__(self) -> None:
        if self.forecast_features.shape != (12,):
            raise ValueError(
                "forecast_features must have shape (12,)"
            )

        if self.uncertainty_features.shape != (6,):
            raise ValueError(
                "uncertainty_features must have shape (6,)"
            )

        if not np.all(np.isfinite(self.forecast_features)):
            raise ValueError(
                "forecast_features contains NaN or infinite values"
            )

        if not np.all(np.isfinite(self.uncertainty_features)):
            raise ValueError(
                "uncertainty_features contains NaN or infinite values"
            )

        if not self.forecast_version.strip():
            raise ValueError(
                "forecast_version must not be empty"
            )

        if not self.uncertainty_version.strip():
            raise ValueError(
                "uncertainty_version must not be empty"
            )

        if not np.isfinite(
            float(self.fused_state.observation_time_s)
        ):
            raise ValueError(
                "fused_state observation time must be finite"
            )

        self.forecast_features.setflags(write=False)
        self.uncertainty_features.setflags(write=False)

    @property
    def observation_time_s(self) -> float:
        return float(
            self.fused_state.observation_time_s
        )

    @property
    def graph_version(self) -> str:
        return str(
            self.fused_state.graph_state.graph_version
        )


class PPOPolicyStateBuilder:
    """
    Build a complete causal PPO policy state.

    The builder owns the Step-16 state-memory lifecycle:

        current Observation
            -> current structured state
            -> append to temporal memory
            -> fuse current state

    Decision memory is updated only when an already-made routing decision is
    supplied. Therefore no future decision can leak into the current state.
    """

    def __init__(
        self,
        scenario,
        *,
        temporal_memory: TemporalMemory | None = None,
        decision_memory: DecisionMemory | None = None,
        state_fusion_builder: StateFusionBuilder | None = None,
        history_length: int = 4,
        forecast_dim: int = 12,
        uncertainty_dim: int = 6,
    ) -> None:

        if forecast_dim != 12:
            raise ValueError(
                "Step-17 PPO forecast contract requires 12 dimensions"
            )

        if uncertainty_dim != 6:
            raise ValueError(
                "Step-17 PPO uncertainty contract requires 6 dimensions"
            )

        self.scenario = scenario

        self.graph_builder = GraphStateBuilder(
            scenario
        )

        self.traffic_builder = MultiScaleTrafficBuilder(
            scenario
        )

        self.event_builder = EventStateBuilder(
            scenario
        )

        self.service_builder = ServiceRiskBuilder(
            scenario
        )

        self.temporal_memory = (
            temporal_memory
            if temporal_memory is not None
            else TemporalMemory(
                history_length=history_length
            )
        )

        self.decision_memory = (
            decision_memory
            if decision_memory is not None
            else DecisionMemory(
                history_length=history_length
            )
        )

        self.fusion_builder = (
            state_fusion_builder
            if state_fusion_builder is not None
            else StateFusionBuilder()
        )

        self._initialized_episode_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, observation: Observation) -> None:
        """
        Reset causal memory for a new episode.

        The first observation itself is not inserted here. Call `build()`
        afterwards so that exactly one current observation is appended.
        """

        episode_id = str(observation.episode_id)
        scenario_id = str(observation.scenario_id)
        graph_version = str(observation.graph_version)

        if scenario_id != str(self.scenario.scenario_id):
            raise ValueError(
                "observation scenario_id does not match Scenario"
            )

        self.temporal_memory.reset(
            episode_id=episode_id,
            scenario_id=scenario_id,
            graph_version=graph_version,
        )

        self.decision_memory.reset(
            episode_id=episode_id,
            scenario_id=scenario_id,
        )

        self._initialized_episode_id = episode_id

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        observation: Observation,
        *,
        forecast_features: Sequence[float],
        uncertainty_features: Sequence[float],
        forecast_version: str,
        uncertainty_version: str,
        decision_context: TemporalDecisionContext | None = None,
    ) -> PPOPolicyState:
        """
        Build one causal policy state.

        All supplied forecast/uncertainty values must correspond to the
        current observation time. This function does not infer future values.
        """

        if not isinstance(
            observation,
            Observation,
        ):
            raise TypeError(
                "observation must be an Observation"
            )

        episode_id = str(observation.episode_id)

        if (
            self._initialized_episode_id is None
            or episode_id != self._initialized_episode_id
        ):
            self.reset(observation)

        forecast = self._vector(
            forecast_features,
            expected=12,
            name="forecast_features",
        )

        uncertainty = self._vector(
            uncertainty_features,
            expected=6,
            name="uncertainty_features",
        )

        # --------------------------------------------------------------
        # Step 16 structured representations
        # --------------------------------------------------------------

        graph_state = self.graph_builder.build(
            observation
        )

        traffic_state = self.traffic_builder.build(
            graph_state
        )

        event_state = self.event_builder.build(
            observation,
            graph_state,
        )

        service_state = self.service_builder.build(
            observation
        )

        # --------------------------------------------------------------
        # Causal temporal memory
        # --------------------------------------------------------------

        self.temporal_memory.append(
            graph_state=graph_state,
            multiscale_traffic_state=traffic_state,
            event_state=event_state,
            service_risk_state=service_state,
            observation_time_s=float(
                observation.observation_time_s
            ),
            state_version=self._state_version_to_int(
                observation.state_version
            ),
            decision_context=decision_context,
        )

        # --------------------------------------------------------------
        # Unified state
        # --------------------------------------------------------------

        fused_state = self.fusion_builder.build(
            graph_state=graph_state,
            multiscale_traffic_state=traffic_state,
            event_state=event_state,
            service_risk_state=service_state,
            temporal_batch=self.temporal_memory.as_batch(),
            decision_memory_batch=self.decision_memory.as_batch(),
        )

        return PPOPolicyState(
            fused_state=fused_state,
            forecast_features=forecast,
            uncertainty_features=uncertainty,
            forecast_version=str(
                forecast_version
            ),
            uncertainty_version=str(
                uncertainty_version
            ),
        )

    # ------------------------------------------------------------------
    # Decision memory
    # ------------------------------------------------------------------

    def record_decision(self, decision) -> None:
        """
        Add a completed controller decision to causal decision memory.

        This must be called AFTER the decision has occurred. The decision
        therefore becomes available only to the next policy observation.
        """

        self.decision_memory.append(
            decision
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vector(
        values: Sequence[float],
        *,
        expected: int,
        name: str,
    ) -> np.ndarray:

        array = np.asarray(
            tuple(values),
            dtype=np.float32,
        )

        if array.shape != (expected,):
            raise ValueError(
                f"{name} must have shape {(expected,)}, "
                f"got {array.shape}"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} contains NaN or infinite values"
            )

        array.setflags(write=False)
        return array

    @staticmethod
    def _state_version_to_int(state_version: str | int) -> int:
        """
        Convert the externally represented state version into the monotonic
        integer required by TemporalMemory.

        Supported forms:
        0
        "0"
        "state-v0"
        "state-v1"
        "state-1"
        """
        if isinstance(state_version, bool):
            raise TypeError("state_version must be str or int")

        if isinstance(state_version, int):
            if state_version < 0:
                raise ValueError("state_version must be non-negative")
            return state_version

        value = str(state_version).strip()

        if not value:
            raise ValueError("state_version cannot be empty")

        if value.isdigit():
            return int(value)

        if value.startswith("state-v"):
            suffix = value[len("state-v"):]
            if suffix.isdigit():
                return int(suffix)

        if value.startswith("state-"):
            suffix = value[len("state-"):]
            if suffix.isdigit():
                return int(suffix)

        raise ValueError(
            f"Unsupported state_version format: {state_version!r}"
        )


class StructuredPPORepresentationProvider:
    """
    Deterministic adapter from PPOPolicyState to the fixed 332-D contract.

    This is a DEVELOPMENT representation adapter.

    It intentionally does not claim to be the final GNN-Transformer
    representation. The later learned encoder can replace this adapter while
    keeping the PPO-facing RepresentationSpec unchanged.
    """

    def __init__(
        self,
        *,
        spec: RepresentationSpec | None = None,
    ) -> None:

        self._spec = (
            spec
            if spec is not None
            else RepresentationSpec()
        )

    @property
    def spec(self) -> RepresentationSpec:
        return self._spec

    def encode(
        self,
        state: PPOPolicyState,
    ) -> RepresentationOutput:

        if not isinstance(
            state,
            PPOPolicyState,
        ):
            raise TypeError(
                "state must be PPOPolicyState"
            )

        fused = state.fused_state

        spatial_source = self._spatial_source(
            fused
        )

        temporal_source = self._temporal_source(
            fused
        )

        context_source = self._context_source(
            fused
        )

        spatial = self._fit_vector(
            spatial_source,
            self.spec.spatial_dim,
        )

        temporal = self._fit_vector(
            temporal_source,
            self.spec.temporal_dim,
        )

        context = self._fit_vector(
            context_source,
            self.spec.context_dim,
        )

        return RepresentationOutput(
            spec=self.spec,
            spatial=spatial,
            temporal=temporal,
            forecast=state.forecast_features.copy(),
            uncertainty=state.uncertainty_features.copy(),
            context=context,
            observation_time_s=state.observation_time_s,
            graph_version=state.graph_version,
            representation_version=self.spec.version,
            metadata={
                "adapter": "structured-development-v1",
                "forecast_version": state.forecast_version,
                "uncertainty_version": (
                    state.uncertainty_version
                ),
            },
        )

    # ------------------------------------------------------------------
    # Source construction
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_source(
        fused: FusedState,
    ) -> np.ndarray:

        graph = fused.graph_state
        traffic = fused.multiscale_traffic_state

        parts: list[np.ndarray] = []

        for array in (
            np.asarray(
                graph.node_features,
                dtype=np.float32,
            ),
            np.asarray(
                graph.edge_features,
                dtype=np.float32,
            ),
            np.asarray(
                traffic.parent_road_features,
                dtype=np.float32,
            ),
            np.asarray(
                traffic.node_neighborhood_features,
                dtype=np.float32,
            ),
            np.asarray(
                traffic.zone_features,
                dtype=np.float32,
            ),
        ):
            if array.size:
                parts.append(
                    np.asarray(
                        [
                            float(np.mean(array)),
                            float(np.std(array)),
                            float(np.min(array)),
                            float(np.max(array)),
                        ],
                        dtype=np.float32,
                    )
                )

        if not parts:
            return np.zeros(
                1,
                dtype=np.float32,
            )

        return np.concatenate(
            parts
        ).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _temporal_source(
        fused: FusedState,
    ) -> np.ndarray:

        temporal = fused.temporal_batch

        parts: list[np.ndarray] = [
            np.asarray(
                temporal.valid_mask,
                dtype=np.float32,
            ),
            np.asarray(
                temporal.delta_time_s,
                dtype=np.float32,
            ),
            np.asarray(
                temporal.decision_features,
                dtype=np.float32,
            ).reshape(-1),
        ]

        return np.concatenate(
            parts
        ).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _context_source(
        fused: FusedState,
    ) -> np.ndarray:

        parts: list[np.ndarray] = [
            np.asarray(
                fused.global_features,
                dtype=np.float32,
            ),
            np.asarray(
                fused.service_risk_state.job_features,
                dtype=np.float32,
            ).reshape(-1),
            np.asarray(
                fused.service_risk_state.vehicle_features,
                dtype=np.float32,
            ).reshape(-1),
            np.asarray(
                fused.event_state.event_features,
                dtype=np.float32,
            ).reshape(-1),
            np.asarray(
                fused.event_state.event_mask,
                dtype=np.float32,
            ).reshape(-1),
            np.asarray(
                fused.decision_memory_batch.features,
                dtype=np.float32,
            ).reshape(-1),
        ]

        return np.concatenate(
            parts
        ).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _fit_vector(
        source: np.ndarray,
        dimension: int,
    ) -> np.ndarray:
        """
        Deterministically resize a finite feature sequence.

        This is deliberately simple and transparent. It is not presented as
        a learned projection.
        """

        source = np.asarray(
            source,
            dtype=np.float32,
        ).reshape(-1)

        if source.size == 0:
            result = np.zeros(
                dimension,
                dtype=np.float32,
            )
        elif source.size >= dimension:
            result = source[:dimension].copy()
        else:
            result = np.zeros(
                dimension,
                dtype=np.float32,
            )
            result[:source.size] = source

        if not np.all(np.isfinite(result)):
            raise ValueError(
                "representation source produced non-finite values"
            )

        result.setflags(write=False)
        return result