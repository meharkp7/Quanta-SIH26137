from __future__ import annotations

import numpy as np
import pytest

from src.contracts.observation import Observation
from src.learning.state.decision_memory import (
    DecisionMemory,
    DecisionMemoryBatch,
    DecisionMemoryRecord,
)
from src.learning.state.event_state import EventStateBuilder
from src.learning.state.graph_state import GraphStateBuilder
from src.learning.state.multiscale_traffic import MultiScaleTrafficBuilder
from src.learning.state.service_risk import ServiceRiskBuilder
from src.learning.state.state_fusion import (
    FUSED_GLOBAL_FEATURE_NAMES,
    StateFusionBuilder,
)
from src.learning.state.temporal_memory import TemporalMemory


def _make_observation(
    scenario,
    *,
    time_s: float = 120.0,
    state_version: str = "1",
):
    return Observation(
        scenario_id=scenario.scenario_id,
        episode_id="episode-test",
        observation_time_s=time_s,
        graph_version=scenario.graph_version,
        edge_observations=(),
        visible_jobs=(),
        fleet=(),
        visible_events=(),
        pending_request_ids=(),
        state_version=state_version,
    )


def _build_components(scenario, *, time_s: float = 120.0):
    observation = _make_observation(
        scenario,
        time_s=time_s,
    )

    graph_state = GraphStateBuilder(
        scenario
    ).build(
        observation
    )

    # IMPORTANT:
    # MultiScaleTrafficBuilder consumes GraphState only.
    traffic_state = MultiScaleTrafficBuilder(
        scenario
    ).build(
        graph_state
    )

    event_state = EventStateBuilder(
        scenario
    ).build(
        observation,
        graph_state,
    )

    service_state = ServiceRiskBuilder(
        scenario
    ).build(
        observation
    )

    temporal = TemporalMemory(
        history_length=4
    )

    temporal.append(
        graph_state=graph_state,
        multiscale_traffic_state=traffic_state,
        event_state=event_state,
        service_risk_state=service_state,
        observation_time_s=time_s,
        state_version=int(observation.state_version),
    )

    decision_memory = DecisionMemory(
        history_length=4
    )

    return (
        graph_state,
        traffic_state,
        event_state,
        service_state,
        temporal.as_batch(),
        decision_memory.as_batch(),
    )


def _fuse(
    scenario,
    *,
    time_s: float = 120.0,
):
    components = _build_components(
        scenario,
        time_s=time_s,
    )

    return StateFusionBuilder().build(
        graph_state=components[0],
        multiscale_traffic_state=components[1],
        event_state=components[2],
        service_risk_state=components[3],
        temporal_batch=components[4],
        decision_memory_batch=components[5],
    )


def test_fusion_preserves_structured_components(
    scenario,
):
    components = _build_components(
        scenario
    )

    fused = StateFusionBuilder().build(
        graph_state=components[0],
        multiscale_traffic_state=components[1],
        event_state=components[2],
        service_risk_state=components[3],
        temporal_batch=components[4],
        decision_memory_batch=components[5],
    )

    assert fused.graph_state is components[0]
    assert fused.multiscale_traffic_state is components[1]
    assert fused.event_state is components[2]
    assert fused.service_risk_state is components[3]
    assert fused.temporal_batch is components[4]
    assert fused.decision_memory_batch is components[5]


def test_fusion_has_stable_global_feature_contract(
    scenario,
):
    fused = _fuse(
        scenario
    )

    assert (
        fused.global_feature_names
        == FUSED_GLOBAL_FEATURE_NAMES
    )

    assert fused.global_features.shape == (
        len(FUSED_GLOBAL_FEATURE_NAMES),
    )

    assert np.all(
        np.isfinite(
            fused.global_features
        )
    )


def test_fusion_uses_graph_state_as_identity_owner(
    scenario,
):
    fused = _fuse(
        scenario
    )

    assert (
        fused.scenario_id
        == fused.graph_state.scenario_id
    )

    assert (
        fused.episode_id
        == fused.graph_state.episode_id
    )

    assert (
        fused.observation_time_s
        == fused.graph_state.observation_time_s
    )


def test_fusion_is_causal_at_current_observation(
    scenario,
):
    fused = _fuse(
        scenario,
        time_s=120.0,
    )

    assert (
        fused.observation_time_s
        == pytest.approx(120.0)
    )

    assert (
        fused.temporal_batch.latest
        is not None
    )

    assert (
        fused.temporal_batch.latest.observation_time_s
        == pytest.approx(120.0)
    )


def test_empty_decision_memory_is_valid(
    scenario,
):
    fused = _fuse(
        scenario
    )

    assert (
        fused.decision_memory_batch.valid_count
        == 0
    )

    index = (
        FUSED_GLOBAL_FEATURE_NAMES.index(
            "decision_valid_fraction"
        )
    )

    assert (
        fused.global_features[index]
        == pytest.approx(0.0)
    )


def test_fusion_rejects_future_decision(
    scenario,
):
    components = list(
        _build_components(
            scenario
        )
    )

    record = DecisionMemoryRecord(
        scenario_id=str(
            scenario.scenario_id
        ),
        state_version="1",
        decision_time_s=121.0,
        requested_action="KEEP",
        executed_action="KEEP",
        affected_vehicle_ids=(),
        mutable_request_ids=(),
        budget_seconds=1.0,
        overridden=False,
        override_reason=None,
        selection_reason="test",
        selected_by="test",
        decision_version="1",
        decision_id="decision-test",
    )

    # Direct batch construction isolates StateFusion's causal check.
    decision_batch = DecisionMemoryBatch(
        records=(
            None,
            None,
            None,
            record,
        ),
        valid_mask=np.array(
            [0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        ),
        decision_times_s=np.array(
            [0.0, 0.0, 0.0, 121.0],
            dtype=np.float64,
        ),
        features=np.zeros(
            (4, 22),
            dtype=np.float32,
        ),
    )

    components[5] = decision_batch

    with pytest.raises(
        ValueError,
        match="future decision",
    ):
        StateFusionBuilder().build(
            graph_state=components[0],
            multiscale_traffic_state=components[1],
            event_state=components[2],
            service_risk_state=components[3],
            temporal_batch=components[4],
            decision_memory_batch=components[5],
        )


def test_fusion_rejects_temporal_timestamp_mismatch(
    scenario,
):
    components = list(
        _build_components(
            scenario
        )
    )

    observation = _make_observation(
        scenario,
        time_s=180.0,
        state_version="2",
    )

    graph_state = GraphStateBuilder(
        scenario
    ).build(
        observation
    )

    traffic_state = MultiScaleTrafficBuilder(
        scenario
    ).build(
        graph_state
    )

    event_state = EventStateBuilder(
        scenario
    ).build(
        observation,
        graph_state,
    )

    service_state = ServiceRiskBuilder(
        scenario
    ).build(
        observation
    )

    temporal = TemporalMemory(
        history_length=4
    )

    temporal.append(
        graph_state=graph_state,
        multiscale_traffic_state=traffic_state,
        event_state=event_state,
        service_risk_state=service_state,
        observation_time_s=180.0,
        state_version=2,
    )

    components[4] = (
        temporal.as_batch()
    )

    with pytest.raises(
        ValueError,
        match="latest timestamp",
    ):
        StateFusionBuilder().build(
            graph_state=components[0],
            multiscale_traffic_state=components[1],
            event_state=components[2],
            service_risk_state=components[3],
            temporal_batch=components[4],
            decision_memory_batch=components[5],
        )


def test_fusion_rejects_multiscale_node_universe_mismatch(
    scenario,
):
    components = _build_components(
        scenario
    )

    traffic = components[1]

    # Bypass the strict constructor only to create a deliberately malformed
    # object and test StateFusion's own alignment check.
    object.__setattr__(
        traffic,
        "node_ids",
        tuple(
            reversed(
                traffic.node_ids
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="node ordering",
    ):
        StateFusionBuilder().build(
            graph_state=components[0],
            multiscale_traffic_state=traffic,
            event_state=components[2],
            service_risk_state=components[3],
            temporal_batch=components[4],
            decision_memory_batch=components[5],
        )


def test_fusion_rejects_event_edge_dimension_mismatch(
    scenario,
):
    components = _build_components(
        scenario
    )

    event = components[2]

    object.__setattr__(
        event,
        "affected_edge_mask",
        np.zeros(
            (
                event.event_count,
                components[0].edge_count + 1,
            ),
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        ValueError,
        match="edge dimension",
    ):
        StateFusionBuilder().build(
            graph_state=components[0],
            multiscale_traffic_state=components[1],
            event_state=event,
            service_risk_state=components[3],
            temporal_batch=components[4],
            decision_memory_batch=components[5],
        )


def test_fusion_does_not_flatten_graph_topology(
    scenario,
):
    fused = _fuse(
        scenario
    )

    assert fused.graph_state.node_ids
    assert fused.graph_state.edge_ids

    assert (
        fused.graph_state.edge_index.shape[0]
        == 2
    )

    assert (
        fused.graph_state.edge_features.ndim
        == 2
    )


def test_global_features_are_read_only(
    scenario,
):
    fused = _fuse(
        scenario
    )

    assert (
        fused.global_features.flags.writeable
        is False
    )

    with pytest.raises(
        ValueError
    ):
        fused.global_features[0] = 99.0


def test_service_pressure_features_use_vehicle_state(
    scenario,
):
    fused = _fuse(
        scenario
    )

    deadline_index = (
        FUSED_GLOBAL_FEATURE_NAMES.index(
            "mean_deadline_pressure"
        )
    )

    workload_index = (
        FUSED_GLOBAL_FEATURE_NAMES.index(
            "mean_workload_pressure"
        )
    )

    load_index = (
        FUSED_GLOBAL_FEATURE_NAMES.index(
            "fleet_load_pressure"
        )
    )

    # The helper observation has no visible fleet, so these summaries must
    # remain zero. This also guards against accidentally indexing job columns
    # for vehicle-only features.
    assert (
        fused.global_features[deadline_index]
        == pytest.approx(0.0)
    )

    assert (
        fused.global_features[workload_index]
        == pytest.approx(0.0)
    )

    assert (
        fused.global_features[load_index]
        == pytest.approx(0.0)
    )
