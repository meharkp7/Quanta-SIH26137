"""
Step 16.9 — Causal state fusion.

Combines the model-independent state representations produced by Steps
16.2–16.8 into one deterministic observation contract.

This module is deliberately a DATA CONTRACT layer. It does not perform:
    - GNN message passing
    - Transformer encoding
    - PPO inference/training
    - QPSO optimization
    - SUMO execution
    - future-label access
    - simulator-truth access

Identity/time ownership:
    GraphState is the authoritative current-observation identity because it
    contains scenario_id, episode_id, graph_version and observation_time_s.

    MultiScaleTrafficState, EventState and ServiceRiskState are aligned
    representations and intentionally do not duplicate that metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .decision_memory import DecisionMemoryBatch
from .event_state import EVENT_FEATURE_NAMES, EventState
from .graph_state import EDGE_FEATURE_NAMES, GraphState
from .multiscale_traffic import (
    SCALE_FEATURE_NAMES,
    MultiScaleTrafficState,
)
from .service_risk import (
    JOB_FEATURE_NAMES,
    VEHICLE_RISK_FEATURE_NAMES,
    ServiceRiskState,
)
from .temporal_memory import TemporalBatch


FUSED_GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "node_count_normalized",
    "edge_count_normalized",
    "observed_edge_fraction",
    "closed_edge_fraction",
    "traffic_coverage",
    "mean_speed_ratio",
    "mean_travel_time_ratio",
    "mean_occupancy",
    "mean_halting_count_normalized",
    "event_count_normalized",
    "active_event_fraction",
    "affected_edge_fraction",
    "pending_job_fraction",
    "in_progress_job_fraction",
    "served_job_fraction",
    "cancelled_job_fraction",
    "mean_deadline_pressure",
    "mean_workload_pressure",
    "fleet_load_pressure",
    "temporal_valid_fraction",
    "decision_valid_fraction",
)


@dataclass(frozen=True)
class FusedState:
    """Immutable unified causal observation contract."""

    scenario_id: str
    episode_id: str
    observation_time_s: float

    graph_state: GraphState
    multiscale_traffic_state: MultiScaleTrafficState
    event_state: EventState
    service_risk_state: ServiceRiskState

    temporal_batch: TemporalBatch
    decision_memory_batch: DecisionMemoryBatch

    global_features: np.ndarray
    global_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.global_feature_names != FUSED_GLOBAL_FEATURE_NAMES:
            raise ValueError(
                "global_feature_names does not match FUSED_GLOBAL_FEATURE_NAMES"
            )

        expected = (len(FUSED_GLOBAL_FEATURE_NAMES),)
        if self.global_features.shape != expected:
            raise ValueError(
                f"global_features must have shape {expected}; "
                f"got {self.global_features.shape}"
            )

        if not np.all(np.isfinite(self.global_features)):
            raise ValueError("global_features contains NaN or infinite values")

        if not np.isfinite(float(self.observation_time_s)):
            raise ValueError("observation_time_s must be finite")
        if self.observation_time_s < 0:
            raise ValueError("observation_time_s must be non-negative")

        self.global_features.setflags(write=False)


class StateFusionBuilder:
    """
    Validate and fuse the current Step-16 representations.

    The builder does not reconstruct metadata that the component contracts
    intentionally do not expose. GraphState owns current observation identity.
    """

    def __init__(
        self,
        *,
        max_nodes: float = 1000.0,
        max_edges: float = 5000.0,
        max_events: float = 100.0,
        max_jobs: float = 100.0,
    ) -> None:
        for name, value in (
            ("max_nodes", max_nodes),
            ("max_edges", max_edges),
            ("max_events", max_events),
            ("max_jobs", max_jobs),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        self.max_nodes = float(max_nodes)
        self.max_edges = float(max_edges)
        self.max_events = float(max_events)
        self.max_jobs = float(max_jobs)

    def build(
        self,
        *,
        graph_state: GraphState,
        multiscale_traffic_state: MultiScaleTrafficState,
        event_state: EventState,
        service_risk_state: ServiceRiskState,
        temporal_batch: TemporalBatch,
        decision_memory_batch: DecisionMemoryBatch,
    ) -> FusedState:
        """Build one causally aligned fused state."""

        self._validate_types(
            graph_state,
            multiscale_traffic_state,
            event_state,
            service_risk_state,
            temporal_batch,
            decision_memory_batch,
        )
        self._validate_alignment(
            graph_state=graph_state,
            multiscale_traffic_state=multiscale_traffic_state,
            event_state=event_state,
            service_risk_state=service_risk_state,
            temporal_batch=temporal_batch,
            decision_memory_batch=decision_memory_batch,
        )

        global_features = self._build_global_features(
            graph_state=graph_state,
            multiscale_traffic_state=multiscale_traffic_state,
            event_state=event_state,
            service_risk_state=service_risk_state,
            temporal_batch=temporal_batch,
            decision_memory_batch=decision_memory_batch,
        )

        return FusedState(
            scenario_id=str(graph_state.scenario_id),
            episode_id=str(graph_state.episode_id),
            observation_time_s=float(graph_state.observation_time_s),
            graph_state=graph_state,
            multiscale_traffic_state=multiscale_traffic_state,
            event_state=event_state,
            service_risk_state=service_risk_state,
            temporal_batch=temporal_batch,
            decision_memory_batch=decision_memory_batch,
            global_features=global_features,
            global_feature_names=FUSED_GLOBAL_FEATURE_NAMES,
        )

    # ------------------------------------------------------------------
    # Contract validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_types(
        graph_state: Any,
        multiscale_traffic_state: Any,
        event_state: Any,
        service_risk_state: Any,
        temporal_batch: Any,
        decision_memory_batch: Any,
    ) -> None:
        expected = (
            ("graph_state", graph_state, GraphState),
            ("multiscale_traffic_state", multiscale_traffic_state, MultiScaleTrafficState),
            ("event_state", event_state, EventState),
            ("service_risk_state", service_risk_state, ServiceRiskState),
            ("temporal_batch", temporal_batch, TemporalBatch),
            ("decision_memory_batch", decision_memory_batch, DecisionMemoryBatch),
        )

        for name, value, cls in expected:
            if not isinstance(value, cls):
                raise TypeError(f"{name} must be {cls.__name__}")

    @staticmethod
    def _validate_alignment(
        *,
        graph_state: GraphState,
        multiscale_traffic_state: MultiScaleTrafficState,
        event_state: EventState,
        service_risk_state: ServiceRiskState,
        temporal_batch: TemporalBatch,
        decision_memory_batch: DecisionMemoryBatch,
    ) -> None:
        scenario_id = str(graph_state.scenario_id)
        episode_id = str(graph_state.episode_id)
        graph_version = str(graph_state.graph_version)
        now = float(graph_state.observation_time_s)

        # Multi-scale representation must use exactly the same node universe.
        if tuple(multiscale_traffic_state.node_ids) != tuple(graph_state.node_ids):
            raise ValueError(
                "MultiScaleTrafficState node ordering does not match GraphState"
            )

        # Event masks are indexed in the physical GraphState edge universe.
        if event_state.affected_edge_mask.shape[1] != graph_state.edge_count:
            raise ValueError(
                "EventState affected_edge_mask edge dimension does not match GraphState"
            )

        if event_state.event_edge_index.shape[1] != event_state.event_count:
            raise ValueError("EventState event_edge_index is inconsistent")

        if event_state.event_count:
            if np.any(event_state.event_edge_index < 0):
                raise ValueError("EventState event_edge_index contains negative indices")
            if np.any(event_state.event_edge_index >= graph_state.edge_count):
                raise ValueError(
                    "EventState event_edge_index references an unknown graph edge"
                )

        # ServiceRiskState has no identity metadata by design. Its IDs are
        # therefore validated structurally, while current episode identity is
        # owned by GraphState/Observation.
        if len(service_risk_state.job_ids) != len(
            service_risk_state.job_id_to_index
        ):
            raise ValueError("ServiceRiskState job lookup is inconsistent")
        if len(service_risk_state.vehicle_ids) != len(
            service_risk_state.vehicle_id_to_index
        ):
            raise ValueError("ServiceRiskState vehicle lookup is inconsistent")

        # Temporal memory must terminate at this exact current observation.
        latest = temporal_batch.latest
        if latest is None:
            raise ValueError(
                "temporal_batch must contain at least one real observation"
            )

        if not np.isclose(float(latest.observation_time_s), now):
            raise ValueError(
                "temporal history latest timestamp does not match GraphState"
            )

        latest_graph = latest.graph_state
        if latest_graph is None:
            raise ValueError("temporal latest snapshot has no GraphState")

        if str(latest_graph.scenario_id) != scenario_id:
            raise ValueError(
                "temporal latest GraphState scenario_id does not match current state"
            )
        if str(latest_graph.episode_id) != episode_id:
            raise ValueError(
                "temporal latest GraphState episode_id does not match current state"
            )
        if str(latest_graph.graph_version) != graph_version:
            raise ValueError(
                "temporal latest GraphState graph_version does not match current state"
            )
        if tuple(latest_graph.node_ids) != tuple(graph_state.node_ids):
            raise ValueError(
                "temporal latest GraphState node ordering does not match current state"
            )
        if tuple(latest_graph.edge_ids) != tuple(graph_state.edge_ids):
            raise ValueError(
                "temporal latest GraphState edge ordering does not match current state"
            )

        # Decision history may be empty at episode start. If present, the
        # newest decision must belong to this scenario and cannot be future.
        latest_decision = decision_memory_batch.latest
        if latest_decision is not None:
            if str(latest_decision.scenario_id) != scenario_id:
                raise ValueError(
                    "decision memory scenario_id does not match GraphState"
                )
            if float(latest_decision.decision_time_s) > now:
                raise ValueError(
                    "decision memory contains a future decision"
                )

    # ------------------------------------------------------------------
    # Global summaries
    # ------------------------------------------------------------------

    def _build_global_features(
        self,
        *,
        graph_state: GraphState,
        multiscale_traffic_state: MultiScaleTrafficState,
        event_state: EventState,
        service_risk_state: ServiceRiskState,
        temporal_batch: TemporalBatch,
        decision_memory_batch: DecisionMemoryBatch,
    ) -> np.ndarray:
        edge = np.asarray(graph_state.edge_features, dtype=np.float32)
        mask = np.asarray(
            graph_state.edge_observation_mask,
            dtype=np.float32,
        )

        observed_edge_fraction = (
            float(np.mean(np.any(mask > 0.0, axis=1)))
            if graph_state.edge_count
            else 0.0
        )

        closed_fraction = self._mean(
            edge,
            EDGE_FEATURE_NAMES.index("known_closed"),
        )

        # Use the multi-scale parent-road representation for traffic
        # aggregates. Its canonical feature names are module-level constants.
        traffic = np.asarray(
            multiscale_traffic_state.parent_road_features,
            dtype=np.float32,
        )

        traffic_coverage = self._mean(
            traffic,
            SCALE_FEATURE_NAMES.index("traffic_coverage"),
        )
        mean_speed = self._mean(
            traffic,
            SCALE_FEATURE_NAMES.index("mean_speed_ratio"),
            clip_high=10.0,
        )
        mean_travel = self._mean(
            traffic,
            SCALE_FEATURE_NAMES.index("mean_travel_time_ratio"),
            clip_high=10.0,
        )
        mean_occupancy = self._mean(
            traffic,
            SCALE_FEATURE_NAMES.index("mean_occupancy"),
        )
        mean_halting = self._mean(
            traffic,
            SCALE_FEATURE_NAMES.index("mean_halting_count_normalized"),
        )

        event_count = event_state.event_count
        active_event_fraction = self._event_mean(
            event_state,
            "effect_started",
        )
        affected_edge_fraction = self._event_mean(
            event_state,
            "affected_edge_fraction",
        )

        job = np.asarray(
            service_risk_state.job_features,
            dtype=np.float32,
        )
        vehicle = np.asarray(
            service_risk_state.vehicle_features,
            dtype=np.float32,
        )

        pending = self._service_mean(job, JOB_FEATURE_NAMES, "status_pending")
        in_progress = self._service_mean(
            job, JOB_FEATURE_NAMES, "status_in_progress"
        )
        served = self._service_mean(job, JOB_FEATURE_NAMES, "status_served")
        cancelled = self._service_mean(
            job, JOB_FEATURE_NAMES, "status_cancelled"
        )

        # These two features are VEHICLE features, not job features.
        deadline_pressure = self._service_mean(
            vehicle,
            VEHICLE_RISK_FEATURE_NAMES,
            "deadline_pressure",
        )
        workload_pressure = self._service_mean(
            vehicle,
            VEHICLE_RISK_FEATURE_NAMES,
            "workload_pressure",
        )
        fleet_load_pressure = self._service_mean(
            vehicle,
            VEHICLE_RISK_FEATURE_NAMES,
            "remaining_load_normalized",
        )

        result = np.asarray(
            [
                np.clip(graph_state.node_count / self.max_nodes, 0.0, 1.0),
                np.clip(graph_state.edge_count / self.max_edges, 0.0, 1.0),
                observed_edge_fraction,
                closed_fraction,
                traffic_coverage,
                mean_speed,
                mean_travel,
                mean_occupancy,
                mean_halting,
                np.clip(event_count / self.max_events, 0.0, 1.0),
                active_event_fraction,
                affected_edge_fraction,
                pending,
                in_progress,
                served,
                cancelled,
                deadline_pressure,
                workload_pressure,
                fleet_load_pressure,
                float(
                    temporal_batch.valid_count
                    / max(temporal_batch.length, 1)
                ),
                float(
                    decision_memory_batch.valid_count
                    / max(decision_memory_batch.length, 1)
                ),
            ],
            dtype=np.float32,
        )

        result.setflags(write=False)
        return result

    @staticmethod
    def _mean(
        values: np.ndarray,
        index: int,
        *,
        clip_high: float = 1.0,
    ) -> float:
        if values.ndim != 2 or values.shape[0] == 0:
            return 0.0
        if index < 0 or index >= values.shape[1]:
            raise ValueError(f"feature index {index} is out of bounds")
        return float(np.clip(np.mean(values[:, index]), 0.0, clip_high))

    @staticmethod
    def _event_mean(state: EventState, feature_name: str) -> float:
        if state.event_count == 0:
            return 0.0
        index = EVENT_FEATURE_NAMES.index(feature_name)
        return float(
            np.clip(
                np.mean(state.event_features[:, index]),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _service_mean(
        values: np.ndarray,
        names: tuple[str, ...],
        feature_name: str,
    ) -> float:
        if values.ndim != 2 or values.shape[0] == 0:
            return 0.0
        index = names.index(feature_name)
        return float(np.clip(np.mean(values[:, index]), 0.0, 1.0))
