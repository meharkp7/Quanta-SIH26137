from __future__ import annotations

from src.learning.ppo_observation_adapter import (
    PPOObservationAdapter,
)


def test_adapter_builds_causal_observation(scenario):
    adapter = PPOObservationAdapter(
        scenario
    )

    observation = adapter.build(
        episode_id="episode-step17",
        observation_time_s=60.0,
        state_version="1",
        latest_state={},
    )

    assert observation.scenario_id == (
        scenario.scenario_id
    )

    assert observation.episode_id == (
        "episode-step17"
    )

    assert observation.observation_time_s == 60.0

    assert all(
        edge.missing
        for edge in observation.edge_observations
    )


def test_future_jobs_are_not_exposed(scenario):
    adapter = PPOObservationAdapter(
        scenario
    )

    observation = adapter.build(
        episode_id="episode-step17",
        observation_time_s=0.0,
        state_version="1",
    )

    for job in observation.visible_jobs:
        assert job.release_s <= 0.0


def test_visible_jobs_are_pending_only_when_released(
    scenario,
):
    adapter = PPOObservationAdapter(
        scenario
    )

    observation = adapter.build(
        episode_id="episode-step17",
        observation_time_s=0.0,
        state_version="1",
    )

    visible_ids = {
        job.request_id
        for job in observation.visible_jobs
    }

    assert set(
        observation.pending_request_ids
    ).issubset(
        visible_ids
    )


def test_fixture_state_can_update_vehicle_loads(
    scenario,
):
    adapter = PPOObservationAdapter(
        scenario
    )

    observation = adapter.build(
        episode_id="episode-step17",
        observation_time_s=60.0,
        state_version="2",
        latest_state={
            "vehicle_loads": [
                2.0,
                3.0,
            ]
        },
    )

    loads = [
        vehicle.remaining_load
        for vehicle in observation.fleet
    ]

    assert loads[:2] == [
        2.0,
        3.0,
    ]


def test_no_future_event_information_is_created(
    scenario,
):
    adapter = PPOObservationAdapter(
        scenario
    )

    observation = adapter.build(
        episode_id="episode-step17",
        observation_time_s=60.0,
        state_version="1",
    )

    assert observation.visible_events == ()