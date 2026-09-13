from __future__ import annotations

import numpy as np
import pytest

from src.contracts.observation import Observation, VisibleEvent
from src.learning.state.graph_state import GraphStateBuilder
from src.learning.state.event_state import EventStateBuilder, EVENT_FEATURE_NAMES


def _obs(scenario, events=(), time_s=120.0):
    return Observation(
        scenario_id=scenario.scenario_id,
        episode_id="episode-test",
        observation_time_s=time_s,
        graph_version=scenario.graph_version,
        edge_observations=(),
        visible_jobs=(),
        fleet=(),
        visible_events=tuple(events),
        pending_request_ids=(),
        state_version="state-test",
    )


def _state(scenario, events=(), time_s=120.0):
    obs = _obs(scenario, events, time_s)
    graph = GraphStateBuilder(scenario).build(obs)
    event_state = EventStateBuilder(scenario).build(obs, graph)
    return obs, event_state


def _event(event_id, event_type, roads, reveal=60.0, effect=100.0):
    return VisibleEvent(
        event_id=event_id,
        revealed_at_s=reveal,
        event_type=event_type,
        affected_parent_road_ids=tuple(roads),
        effect_start_s=effect,
    )


def test_no_visible_events_has_valid_empty_state(scenario):
    _, state = _state(scenario)
    assert state.event_count == 0
    assert state.event_features.shape == (0, len(EVENT_FEATURE_NAMES))
    assert state.affected_edge_mask.shape[0] == 0


def test_visible_event_is_encoded_and_causally_visible(scenario):
    road = str(scenario.edges[0].parent_road_id)
    event = _event("incident-1", "incident", [road], reveal=60.0, effect=180.0)
    _, state = _state(scenario, [event], time_s=120.0)

    assert state.event_ids == ("incident-1",)
    row = state.event_features[0]
    assert row[EVENT_FEATURE_NAMES.index("type_incident")] == pytest.approx(1.0)
    assert row[EVENT_FEATURE_NAMES.index("effect_started")] == pytest.approx(0.0)
    assert row[EVENT_FEATURE_NAMES.index("effect_delay_normalized")] > 0
    assert state.event_mask[0, 0] == 1.0
    assert state.event_mask[0, 1] == 1.0
    assert np.sum(state.affected_edge_mask[0]) > 0


def test_future_effect_is_not_marked_active(scenario):
    road = str(scenario.edges[0].parent_road_id)
    event = _event("closure-1", "closure", [road], reveal=50.0, effect=500.0)
    _, state = _state(scenario, [event], time_s=100.0)

    assert state.event_features[0, EVENT_FEATURE_NAMES.index("effect_started")] == 0.0
    assert state.event_features[0, EVENT_FEATURE_NAMES.index("effect_delay_normalized")] > 0


def test_started_effect_is_encoded_without_inventing_end_time(scenario):
    road = str(scenario.edges[0].parent_road_id)
    event = _event("incident-2", "incident", [road], reveal=50.0, effect=80.0)
    _, state = _state(scenario, [event], time_s=100.0)

    assert state.event_features[0, EVENT_FEATURE_NAMES.index("effect_started")] == 1.0
    assert state.event_mask[0, 1] == 0.0


def test_deterministic_event_order(scenario):
    roads = [str(scenario.edges[0].parent_road_id)]
    events = (
        _event("z-event", "incident", roads, reveal=80.0),
        _event("a-event", "closure", roads, reveal=60.0),
    )
    _, state = _state(scenario, events)
    assert state.event_ids == ("a-event", "z-event")


def test_unknown_event_type_is_safe_other_class(scenario):
    road = str(scenario.edges[0].parent_road_id)
    event = _event("unknown", "road_hazard", [road])
    _, state = _state(scenario, [event])
    assert state.event_features[
        0, EVENT_FEATURE_NAMES.index("type_other")
    ] == pytest.approx(1.0)


def test_duplicate_event_ids_rejected(scenario):
    road = str(scenario.edges[0].parent_road_id)
    events = (
        _event("same", "incident", [road]),
        _event("same", "closure", [road]),
    )
    obs = _obs(scenario, events)
    graph = GraphStateBuilder(scenario).build(obs)
    with pytest.raises(ValueError, match="duplicate"):
        EventStateBuilder(scenario).build(obs, graph)


def test_state_is_read_only(scenario):
    road = str(scenario.edges[0].parent_road_id)
    _, state = _state(scenario, [_event("e", "incident", [road])])
    with pytest.raises(ValueError):
        state.event_features[0, 0] = 1.0
