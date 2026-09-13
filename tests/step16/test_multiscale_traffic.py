from __future__ import annotations

import numpy as np
import pytest

from src.contracts.observation import EdgeObservation, Observation
from src.learning.state.graph_state import GraphStateBuilder, EDGE_FEATURE_NAMES
from src.learning.state.multiscale_traffic import (
    MultiScaleTrafficBuilder,
    SCALE_FEATURE_NAMES,
)


def _obs(scenario, edge_observations=()):
    return Observation(
        scenario_id=scenario.scenario_id,
        episode_id="episode-test",
        observation_time_s=120.0,
        graph_version=scenario.graph_version,
        edge_observations=tuple(edge_observations),
        visible_jobs=(),
        fleet=(),
        visible_events=(),
        pending_request_ids=(),
        state_version="state-test",
    )


def _state(scenario, edge_observations=()):
    graph = GraphStateBuilder(scenario).build(_obs(scenario, edge_observations))
    return graph, MultiScaleTrafficBuilder(scenario).build(graph)


def test_multiscale_has_stable_scales_and_shapes(scenario):
    graph, state = _state(scenario)

    assert state.parent_road_ids == tuple(
        sorted({str(e.parent_road_id) for e in scenario.edges})
    )
    assert state.zone_ids == tuple(
        sorted({str(n.zone_id) for n in scenario.nodes})
    )
    assert state.parent_road_features.shape == (
        state.parent_road_count,
        len(SCALE_FEATURE_NAMES),
    )
    assert state.node_neighborhood_features.shape == (
        len(scenario.nodes),
        len(SCALE_FEATURE_NAMES),
    )
    assert state.zone_features.shape == (
        state.zone_count,
        len(SCALE_FEATURE_NAMES),
    )
    assert state.parent_road_edge_index.shape == (2, state.parent_road_count)
    assert state.zone_edge_index.shape == (2, state.zone_count)


def test_missing_edges_are_excluded_from_numeric_traffic_aggregates(scenario):
    edges = list(scenario.edges[:2])
    if len(edges) < 2 or str(edges[0].parent_road_id) != str(edges[1].parent_road_id):
        pytest.skip("fixture does not contain two edges on one parent road")

    observations = (
        EdgeObservation(
            edge_id=edges[0].edge_id,
            observed_speed_mps=float(edges[0].speed_limit_mps) * 0.5,
            observed_travel_time_s=float(edges[0].free_flow_time_s) * 2,
            observation_age_s=10,
            missing=False,
            known_closed=False,
            occupancy=0.2,
            halting_count=2,
        ),
        EdgeObservation(
            edge_id=edges[1].edge_id,
            observed_speed_mps=None,
            observed_travel_time_s=None,
            observation_age_s=20,
            missing=True,
            known_closed=False,
            occupancy=None,
            halting_count=None,
        ),
    )
    _, state = _state(scenario, observations)
    road = state.parent_road_features[state.parent_road_id_to_index[str(edges[0].parent_road_id)]]

    assert road[SCALE_FEATURE_NAMES.index("mean_speed_ratio")] == pytest.approx(0.5)
    assert road[SCALE_FEATURE_NAMES.index("traffic_coverage")] == pytest.approx(0.5)
    assert road[SCALE_FEATURE_NAMES.index("missing_fraction")] == pytest.approx(0.5)


def test_closure_propagates_without_becoming_speed_zero(scenario):
    edge = scenario.edges[0]
    observations = (
        EdgeObservation(
            edge_id=edge.edge_id,
            observed_speed_mps=0.0,
            observed_travel_time_s=None,
            observation_age_s=0,
            missing=False,
            known_closed=True,
            occupancy=None,
            halting_count=None,
        ),
    )
    _, state = _state(scenario, observations)
    road = state.parent_road_features[state.parent_road_id_to_index[str(edge.parent_road_id)]]
    assert road[SCALE_FEATURE_NAMES.index("closed_fraction")] > 0
    assert road[SCALE_FEATURE_NAMES.index("traffic_coverage")] > 0


def test_deterministic_build(scenario):
    graph, a = _state(scenario)
    b = MultiScaleTrafficBuilder(scenario).build(graph)
    np.testing.assert_array_equal(a.parent_road_features, b.parent_road_features)
    np.testing.assert_array_equal(a.node_neighborhood_features, b.node_neighborhood_features)
    np.testing.assert_array_equal(a.zone_features, b.zone_features)


def test_scale_state_is_read_only(scenario):
    _, state = _state(scenario)
    with pytest.raises(ValueError):
        state.zone_features[0, 0] = 1.0
