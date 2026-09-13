from __future__ import annotations

import numpy as np
import pytest

from src.contracts.observation import Observation
from src.learning.state.graph_state import GraphStateBuilder
from src.learning.state.multiscale_traffic import MultiScaleTrafficBuilder
from src.learning.state.event_state import EventStateBuilder
from src.learning.state.service_risk import ServiceRiskBuilder
from src.learning.state.temporal_memory import (
    DECISION_FEATURE_NAMES,
    TemporalDecisionContext,
    TemporalMemory,
    TemporalSnapshot,
)


# ============================================================================
# Test-state construction
# ============================================================================


def _make_observation(
    scenario,
    *,
    time_s: float = 120.0,
    state_version: str = "state-test",
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


def _build_states(
    scenario,
    *,
    time_s: float = 120.0,
):
    observation = _make_observation(
        scenario,
        time_s=time_s,
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

    return (
        graph_state,
        traffic_state,
        event_state,
        service_state,
    )


def _append(
    memory,
    *,
    graph_state,
    traffic_state,
    event_state,
    service_state,
    time_s,
    version,
    decision=None,
):
    return memory.append(
        graph_state=graph_state,
        multiscale_traffic_state=traffic_state,
        event_state=event_state,
        service_risk_state=service_state,
        observation_time_s=time_s,
        state_version=version,
        decision_context=decision,
    )


# ============================================================================
# Basic lifecycle
# ============================================================================


def test_empty_memory_is_empty_and_padded(
    scenario,
):
    memory = TemporalMemory(
        history_length=4
    )

    assert memory.is_empty
    assert memory.size == 0

    batch = memory.as_batch()

    assert batch.length == 4
    assert batch.valid_count == 0
    assert batch.padding_count == 4

    assert np.all(
        batch.valid_mask == 0
    )

    assert np.all(
        batch.delta_time_s == 0
    )


def test_append_produces_causal_snapshot(
    scenario,
):
    (
        graph_state,
        traffic_state,
        event_state,
        service_state,
    ) = _build_states(
        scenario,
        time_s=60.0,
    )

    memory = TemporalMemory(
        history_length=4
    )

    snapshot = _append(
        memory,
        graph_state=graph_state,
        traffic_state=traffic_state,
        event_state=event_state,
        service_state=service_state,
        time_s=60.0,
        version=1,
    )

    assert snapshot.observation_time_s == pytest.approx(
        60.0
    )

    assert snapshot.state_version == 1

    assert snapshot.delta_time_s == pytest.approx(
        0.0
    )

    assert not snapshot.is_padding


def test_delta_time_is_computed_from_previous_real_observation(
    scenario,
):
    states = _build_states(
        scenario,
        time_s=60.0,
    )

    memory = TemporalMemory(
        history_length=4
    )

    _append(
        memory,
        graph_state=states[0],
        traffic_state=states[1],
        event_state=states[2],
        service_state=states[3],
        time_s=60.0,
        version=1,
    )

    states = _build_states(
        scenario,
        time_s=120.0,
    )

    second = _append(
        memory,
        graph_state=states[0],
        traffic_state=states[1],
        event_state=states[2],
        service_state=states[3],
        time_s=120.0,
        version=2,
    )

    assert second.delta_time_s == pytest.approx(
        60.0
    )


# ============================================================================
# Causal ordering
# ============================================================================


def test_non_increasing_time_is_rejected(
    scenario,
):
    states = _build_states(
        scenario,
        time_s=60.0,
    )

    memory = TemporalMemory(
        history_length=4
    )

    _append(
        memory,
        graph_state=states[0],
        traffic_state=states[1],
        event_state=states[2],
        service_state=states[3],
        time_s=60.0,
        version=1,
    )

    states = _build_states(
        scenario,
        time_s=120.0,
    )

    with pytest.raises(
        ValueError,
        match="observation_time_s",
    ):
        _append(
            memory,
            graph_state=states[0],
            traffic_state=states[1],
            event_state=states[2],
            service_state=states[3],
            time_s=60.0,
            version=2,
        )

    with pytest.raises(
        ValueError,
        match="observation_time_s",
    ):
        _append(
            memory,
            graph_state=states[0],
            traffic_state=states[1],
            event_state=states[2],
            service_state=states[3],
            time_s=30.0,
            version=3,
        )


def test_non_increasing_state_version_is_rejected(
    scenario,
):
    states = _build_states(
        scenario,
        time_s=60.0,
    )

    memory = TemporalMemory(
        history_length=4
    )

    _append(
        memory,
        graph_state=states[0],
        traffic_state=states[1],
        event_state=states[2],
        service_state=states[3],
        time_s=60.0,
        version=2,
    )

    states = _build_states(
        scenario,
        time_s=120.0,
    )

    with pytest.raises(
        ValueError,
        match="state_version",
    ):
        _append(
            memory,
            graph_state=states[0],
            traffic_state=states[1],
            event_state=states[2],
            service_state=states[3],
            time_s=120.0,
            version=2,
        )

    with pytest.raises(
        ValueError,
        match="state_version",
    ):
        _append(
            memory,
            graph_state=states[0],
            traffic_state=states[1],
            event_state=states[2],
            service_state=states[3],
            time_s=180.0,
            version=1,
        )


# ============================================================================
# Bounded history
# ============================================================================


def test_history_is_bounded_and_oldest_state_is_evicted(
    scenario,
):
    memory = TemporalMemory(
        history_length=3
    )

    for index in range(4):
        time_s = float(
            (index + 1) * 60
        )

        states = _build_states(
            scenario,
            time_s=time_s,
        )

        _append(
            memory,
            graph_state=states[0],
            traffic_state=states[1],
            event_state=states[2],
            service_state=states[3],
            time_s=time_s,
            version=index + 1,
        )

    snapshots = memory.snapshots()

    assert len(snapshots) == 3

    assert [
        item.observation_time_s
        for item in snapshots
    ] == [
        120.0,
        180.0,
        240.0,
    ]

    assert [
        item.state_version
        for item in snapshots
    ] == [
        2,
        3,
        4,
    ]


def test_batch_left_pads_and_preserves_real_order(
    scenario,
):
    memory = TemporalMemory(
        history_length=4
    )

    for index, time_s in enumerate(
        (60.0, 120.0),
        start=1,
    ):
        states = _build_states(
            scenario,
            time_s=time_s,
        )

        _append(
            memory,
            graph_state=states[0],
            traffic_state=states[1],
            event_state=states[2],
            service_state=states[3],
            time_s=time_s,
            version=index,
        )

    batch = memory.as_batch()

    assert batch.length == 4

    assert batch.valid_mask.tolist() == [
        0.0,
        0.0,
        1.0,
        1.0,
    ]

    assert batch.observation_times_s.tolist() == [
        0.0,
        0.0,
        60.0,
        120.0,
    ]

    assert batch.state_versions.tolist() == [
        0,
        0,
        1,
        2,
    ]

    assert batch.delta_time_s.tolist() == [
        0.0,
        0.0,
        0.0,
        60.0,
    ]


# ============================================================================
# Decision memory
# ============================================================================


def test_decision_context_is_encoded(
    scenario,
):
    decision = TemporalDecisionContext(
        requested_action="GLOBAL",
        executed_action="REGIONAL",
        affected_vehicle_count=2,
        total_vehicle_count=4,
        mutable_request_count=3,
        total_request_count=6,
        budget_seconds=30.0,
        overridden=True,
    )

    states = _build_states(
        scenario,
        time_s=60.0,
    )

    memory = TemporalMemory(
        history_length=2,
        max_budget_seconds=60.0,
    )

    _append(
        memory,
        graph_state=states[0],
        traffic_state=states[1],
        event_state=states[2],
        service_state=states[3],
        time_s=60.0,
        version=1,
        decision=decision,
    )

    batch = memory.as_batch()

    assert batch.decision_features.shape == (
        2,
        len(DECISION_FEATURE_NAMES),
    )

    features = batch.decision_features[-1]

    # Index map:
    # 0  has_decision
    # 1-5 requested action
    # 6-10 executed action
    # 11 was_overridden
    # 12 affected_vehicle_fraction
    # 13 mutable_request_fraction
    # 14 budget_normalized

    assert features[0] == pytest.approx(
        1.0
    )

    assert features[5] == pytest.approx(
        1.0
    )  # requested GLOBAL

    assert features[9] == pytest.approx(
        1.0
    )  # executed REGIONAL

    assert features[11] == pytest.approx(
        1.0
    )  # overridden

    assert features[12] == pytest.approx(
        0.5
    )

    assert features[13] == pytest.approx(
        0.5
    )

    assert features[14] == pytest.approx(
        0.5
    )


# ============================================================================
# Reset
# ============================================================================


def test_memory_reset_removes_old_history(
    scenario,
):
    states = _build_states(
        scenario,
        time_s=60.0,
    )

    memory = TemporalMemory(
        history_length=4
    )

    _append(
        memory,
        graph_state=states[0],
        traffic_state=states[1],
        event_state=states[2],
        service_state=states[3],
        time_s=60.0,
        version=1,
    )

    memory.reset(
        episode_id="episode-2",
        scenario_id=str(
            scenario.scenario_id
        ),
        graph_version=str(
            scenario.graph_version
        ),
    )

    assert memory.is_empty
    assert memory.size == 0
    assert memory.episode_id == "episode-2"

    batch = memory.as_batch()

    assert batch.valid_count == 0
    assert np.all(
        batch.valid_mask == 0
    )


# ============================================================================
# Identity consistency
# ============================================================================


def test_scenario_identity_cannot_change(
    scenario,
):
    states = _build_states(
        scenario,
        time_s=60.0,
    )

    memory = TemporalMemory(
        history_length=4
    )

    _append(
        memory,
        graph_state=states[0],
        traffic_state=states[1],
        event_state=states[2],
        service_state=states[3],
        time_s=60.0,
        version=1,
    )

    altered_graph = states[0].__class__(
        **{
            **vars(states[0]),
            "scenario_id": "different-scenario",
        }
    )

    with pytest.raises(
        ValueError,
        match="scenario_id",
    ):
        _append(
            memory,
            graph_state=altered_graph,
            traffic_state=states[1],
            event_state=states[2],
            service_state=states[3],
            time_s=120.0,
            version=2,
        )


def test_graph_version_cannot_change(
    scenario,
):
    states = _build_states(
        scenario,
        time_s=60.0,
    )

    memory = TemporalMemory(
        history_length=4
    )

    _append(
        memory,
        graph_state=states[0],
        traffic_state=states[1],
        event_state=states[2],
        service_state=states[3],
        time_s=60.0,
        version=1,
    )

    altered_graph = states[0].__class__(
        **{
            **vars(states[0]),
            "graph_version": "different-version",
        }
    )

    with pytest.raises(
        ValueError,
        match="graph_version",
    ):
        _append(
            memory,
            graph_state=altered_graph,
            traffic_state=states[1],
            event_state=states[2],
            service_state=states[3],
            time_s=120.0,
            version=2,
        )


def test_node_ordering_mismatch_is_rejected(
    scenario,
):
    states = _build_states(
        scenario,
        time_s=60.0,
    )

    class AlteredTrafficState:
        pass

    altered_traffic = AlteredTrafficState()

    # Preserve every attribute TemporalMemory needs, but deliberately
    # provide a different node ordering.
    for name in (
        "parent_road_ids",
        "zone_ids",
        "parent_road_features",
        "node_neighborhood_features",
        "zone_features",
        "parent_road_edge_index",
        "zone_edge_index",
        "node_id_to_index",
        "parent_road_id_to_index",
        "zone_id_to_index",
    ):
        setattr(
            altered_traffic,
            name,
            getattr(states[1], name),
        )

    altered_traffic.node_ids = tuple(
        reversed(states[1].node_ids)
    )

    memory = TemporalMemory(
        history_length=4
    )

    with pytest.raises(
        ValueError,
        match="node ordering",
    ):
        _append(
            memory,
            graph_state=states[0],
            traffic_state=altered_traffic,
            event_state=states[2],
            service_state=states[3],
            time_s=60.0,
            version=1,
        )

# ============================================================================
# Immutability
# ============================================================================


def test_stored_numpy_arrays_are_not_writable(
    scenario,
):
    states = _build_states(
        scenario,
        time_s=60.0,
    )

    memory = TemporalMemory(
        history_length=2
    )

    _append(
        memory,
        graph_state=states[0],
        traffic_state=states[1],
        event_state=states[2],
        service_state=states[3],
        time_s=60.0,
        version=1,
    )

    stored = memory.snapshots()[0]

    assert (
        stored.graph_state.node_features.flags.writeable
        is False
    )

    assert (
        stored.graph_state.edge_features.flags.writeable
        is False
    )

    with pytest.raises(
        ValueError
    ):
        stored.graph_state.node_features[
            0,
            0,
        ] = 999.0


def test_temporal_memory_defensively_copies_state(
    scenario,
):
    states = _build_states(
        scenario,
        time_s=60.0,
    )

    # Create a mutable graph state specifically for this test.
    mutable_graph = states[0].__class__(
        **{
            **vars(states[0]),
            "node_features": states[0].node_features.copy(),
            "edge_features": states[0].edge_features.copy(),
            "edge_index": states[0].edge_index.copy(),
            "edge_observation_mask": states[0].edge_observation_mask.copy(),
        }
    )

    # The constructor freezes the arrays, so create a mutable copy only
    # after construction for the purpose of testing the boundary.
    mutable_graph.node_features.setflags(
        write=True
    )

    original = float(
        mutable_graph.node_features[
            0,
            0,
        ]
    )

    memory = TemporalMemory(
        history_length=2
    )

    _append(
        memory,
        graph_state=mutable_graph,
        traffic_state=states[1],
        event_state=states[2],
        service_state=states[3],
        time_s=60.0,
        version=1,
    )

    mutable_graph.node_features[
        0,
        0,
    ] = original + 100.0

    stored_value = float(
        memory.snapshots()[
            0
        ].graph_state.node_features[
            0,
            0,
        ]
    )

    assert stored_value == pytest.approx(
        original
    )


# ============================================================================
# Padding contract
# ============================================================================


def test_padding_snapshot_has_no_state():
    snapshot = TemporalSnapshot.padding()

    assert snapshot.is_padding

    assert snapshot.graph_state is None
    assert snapshot.multiscale_traffic_state is None
    assert snapshot.event_state is None
    assert snapshot.service_risk_state is None

    assert snapshot.delta_time_s == 0.0


def test_component_node_universe_mismatch_is_rejected(
    scenario,
):
    states = _build_states(
        scenario,
        time_s=60.0,
    )

    class AlteredTrafficState:
        pass

    altered_traffic = AlteredTrafficState()

    for name in (
        "parent_road_ids",
        "zone_ids",
        "parent_road_features",
        "node_neighborhood_features",
        "zone_features",
        "node_id_to_index",
        "parent_road_id_to_index",
        "zone_id_to_index",
        "parent_road_edge_index",
        "zone_edge_index",
    ):
        setattr(
            altered_traffic,
            name,
            getattr(states[1], name),
        )

    altered_traffic.node_ids = tuple(
        list(states[1].node_ids)
        + ["fake-node"]
    )

    memory = TemporalMemory(
        history_length=4
    )

    with pytest.raises(
        ValueError,
        match="node ordering",
    ):
        _append(
            memory,
            graph_state=states[0],
            traffic_state=altered_traffic,
            event_state=states[2],
            service_state=states[3],
            time_s=60.0,
            version=1,
        )