from __future__ import annotations

import numpy as np
import pytest

from src.contracts.observation import (
    EdgeObservation,
    Observation,
)
from src.learning.state.graph_state import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    GraphStateBuilder,
)


def _make_observation(
    scenario,
    *,
    time_s: float = 120.0,
    edge_observations=(),
):
    return Observation(
        scenario_id=scenario.scenario_id,
        episode_id="episode-test",
        observation_time_s=time_s,
        graph_version=scenario.graph_version,
        edge_observations=tuple(
            edge_observations
        ),
        visible_jobs=(),
        fleet=(),
        visible_events=(),
        pending_request_ids=(),
        state_version="state-test",
    )


def test_graph_state_has_stable_topology_and_shapes(
    scenario,
):
    observation = _make_observation(
        scenario
    )

    builder = GraphStateBuilder(
        scenario
    )

    state = builder.build(
        observation
    )

    assert state.node_ids == tuple(
        sorted(
            str(node.node_id)
            for node in scenario.nodes
        )
    )

    assert state.edge_ids == tuple(
        sorted(
            str(edge.edge_id)
            for edge in scenario.edges
        )
    )

    assert state.node_features.shape == (
        len(scenario.nodes),
        len(NODE_FEATURE_NAMES),
    )

    assert state.edge_features.shape == (
        len(scenario.edges),
        len(EDGE_FEATURE_NAMES),
    )

    assert state.edge_index.shape == (
        2,
        len(scenario.edges),
    )

    assert state.edge_observation_mask.shape == (
        len(scenario.edges),
        4,
    )


def test_graph_state_edge_index_preserves_directed_topology(
    scenario,
):
    observation = _make_observation(
        scenario
    )

    state = GraphStateBuilder(
        scenario
    ).build(
        observation
    )

    for index, edge in enumerate(
        sorted(
            scenario.edges,
            key=lambda item: str(
                item.edge_id
            ),
        )
    ):
        from_index = state.node_id_to_index[
            str(edge.from_node)
        ]

        to_index = state.node_id_to_index[
            str(edge.to_node)
        ]

        assert state.edge_index[
            0,
            index
        ] == from_index

        assert state.edge_index[
            1,
            index
        ] == to_index


def test_missing_edge_is_explicitly_masked(
    scenario,
):
    observation = _make_observation(
        scenario
    )

    state = GraphStateBuilder(
        scenario
    ).build(
        observation
    )

    for index in range(
        len(scenario.edges)
    ):
        assert state.edge_features[
            index,
            EDGE_FEATURE_NAMES.index(
                "missing_observation"
            )
        ] == pytest.approx(1.0)

        assert np.all(
            state.edge_observation_mask[
                index
            ] == 0.0
        )


def test_observed_speed_and_travel_time_are_encoded(
    scenario,
):
    edge = scenario.edges[0]

    observation = _make_observation(
        scenario,
        edge_observations=(
            EdgeObservation(
                edge_id=edge.edge_id,
                observed_speed_mps=(
                    float(
                        edge.speed_limit_mps
                    )
                    * 0.5
                ),
                observed_travel_time_s=(
                    float(
                        edge.free_flow_time_s
                    )
                    * 2.0
                ),
                observation_age_s=10.0,
                missing=False,
                known_closed=False,
                occupancy=0.25,
                halting_count=5,
            ),
        ),
    )

    state = GraphStateBuilder(
        scenario
    ).build(
        observation
    )

    edge_index = state.edge_id_to_index[
        str(edge.edge_id)
    ]

    speed_column = (
        EDGE_FEATURE_NAMES.index(
            "speed_ratio"
        )
    )

    travel_column = (
        EDGE_FEATURE_NAMES.index(
            "travel_time_ratio"
        )
    )

    occupancy_column = (
        EDGE_FEATURE_NAMES.index(
            "occupancy"
        )
    )

    assert state.edge_features[
        edge_index,
        speed_column,
    ] == pytest.approx(
        0.5
    )

    assert state.edge_features[
        edge_index,
        travel_column,
    ] == pytest.approx(
        2.0
    )

    assert state.edge_features[
        edge_index,
        occupancy_column,
    ] == pytest.approx(
        0.25
    )

    assert np.all(
        state.edge_observation_mask[
            edge_index
        ]
        == np.array(
            [1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
    )


def test_known_closure_is_encoded(
    scenario,
):
    edge = scenario.edges[0]

    observation = _make_observation(
        scenario,
        edge_observations=(
            EdgeObservation(
                edge_id=edge.edge_id,
                observed_speed_mps=0.0,
                observed_travel_time_s=None,
                observation_age_s=0.0,
                missing=False,
                known_closed=True,
                occupancy=None,
                halting_count=None,
            ),
        ),
    )

    state = GraphStateBuilder(
        scenario
    ).build(
        observation
    )

    index = state.edge_id_to_index[
        str(edge.edge_id)
    ]

    assert state.edge_features[
        index,
        EDGE_FEATURE_NAMES.index(
            "known_closed"
        ),
    ] == pytest.approx(1.0)


def test_missing_observation_never_becomes_observed_zero(
    scenario,
):
    edge = scenario.edges[0]

    observation = _make_observation(
        scenario,
        edge_observations=(
            EdgeObservation(
                edge_id=edge.edge_id,
                observed_speed_mps=None,
                observed_travel_time_s=None,
                observation_age_s=30.0,
                missing=True,
                known_closed=False,
                occupancy=None,
                halting_count=None,
            ),
        ),
    )

    state = GraphStateBuilder(
        scenario
    ).build(
        observation
    )

    index = state.edge_id_to_index[
        str(edge.edge_id)
    ]

    speed_index = EDGE_FEATURE_NAMES.index(
        "speed_ratio"
    )

    assert state.edge_features[
        index,
        speed_index,
    ] == pytest.approx(0.0)

    assert state.edge_features[
        index,
        EDGE_FEATURE_NAMES.index(
            "missing_observation"
        ),
    ] == pytest.approx(1.0)

    assert state.edge_observation_mask[
        index,
        0,
    ] == pytest.approx(0.0)


def test_graph_state_is_deterministic(
    scenario,
):
    observation = _make_observation(
        scenario
    )

    builder = GraphStateBuilder(
        scenario
    )

    state_a = builder.build(
        observation
    )

    state_b = builder.build(
        observation
    )

    assert state_a.node_ids == state_b.node_ids
    assert state_a.edge_ids == state_b.edge_ids

    np.testing.assert_array_equal(
        state_a.node_features,
        state_b.node_features,
    )

    np.testing.assert_array_equal(
        state_a.edge_features,
        state_b.edge_features,
    )

    np.testing.assert_array_equal(
        state_a.edge_index,
        state_b.edge_index,
    )

    np.testing.assert_array_equal(
        state_a.edge_observation_mask,
        state_b.edge_observation_mask,
    )


def test_graph_state_rejects_wrong_graph_version(
    scenario,
):
    observation = _make_observation(
        scenario
    )

    observation = observation.model_copy(
        update={
            "graph_version": "wrong-graph-version"
        }
    )

    builder = GraphStateBuilder(
        scenario
    )

    with pytest.raises(
        ValueError,
        match="graph_version",
    ):
        builder.build(
            observation
        )


def test_graph_state_rejects_unknown_edge_observation(
    scenario,
):
    observation = _make_observation(
        scenario,
        edge_observations=(
            EdgeObservation(
                edge_id="definitely-not-an-edge",
                observed_speed_mps=None,
                observed_travel_time_s=None,
                observation_age_s=0.0,
                missing=True,
                known_closed=False,
                occupancy=None,
                halting_count=None,
            ),
        ),
    )

    builder = GraphStateBuilder(
        scenario
    )

    with pytest.raises(
        ValueError,
        match="unknown.*edge_id",
    ):
        builder.build(
            observation
        )


def test_graph_state_lookup_helpers(
    scenario,
):
    observation = _make_observation(
        scenario
    )

    state = GraphStateBuilder(
        scenario
    ).build(
        observation
    )

    first_node = state.node_ids[0]
    first_edge = state.edge_ids[0]

    assert state.node_features_for(
        first_node
    ).shape == (
        len(NODE_FEATURE_NAMES),
    )

    assert state.edge_features_for(
        first_edge
    ).shape == (
        len(EDGE_FEATURE_NAMES),
    )

    assert state.edge_index_for(
        first_edge
    ) == 0

    with pytest.raises(
        KeyError
    ):
        state.node_features_for(
            "unknown-node"
        )

    with pytest.raises(
        KeyError
    ):
        state.edge_features_for(
            "unknown-edge"
        )