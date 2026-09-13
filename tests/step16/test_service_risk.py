from __future__ import annotations

import numpy as np
import pytest

from src.contracts.observation import Observation, VisibleJob, VehicleObservation
from src.learning.state.graph_state import GraphStateBuilder
from src.learning.state.service_risk import (
    JOB_FEATURE_NAMES,
    ServiceRiskBuilder,
)


def _obs(scenario, jobs=(), fleet=(), time_s=120.0):
    return Observation(
        scenario_id=scenario.scenario_id,
        episode_id="episode-test",
        observation_time_s=time_s,
        graph_version=scenario.graph_version,
        edge_observations=(),
        visible_jobs=tuple(jobs),
        fleet=tuple(fleet),
        visible_events=(),
        pending_request_ids=tuple(str(j.request_id) for j in jobs),
        state_version="state-test",
    )


def _job(request, status="pending", latest=None):
    return VisibleJob(
        request_id=request.request_id,
        demand=request.demand,
        release_s=request.release_s,
        earliest_service_start_s=request.earliest_service_start_s,
        latest_service_start_s=(
            request.latest_service_start_s if latest is None else latest
        ),
        service_duration_s=request.service_duration_s,
        status=status,
    )


def test_empty_observation_has_valid_empty_risk_state(scenario):
    state = ServiceRiskBuilder(scenario).build(_obs(scenario))
    assert state.job_count == 0
    assert state.vehicle_count == 0


def test_deadline_risk_is_causal_and_increases_near_deadline(scenario):
    request = scenario.requests[0]
    early = _job(request, latest=1000.0)
    near = _job(request, latest=130.0)

    early_state = ServiceRiskBuilder(scenario).build(
        _obs(scenario, [early], time_s=120.0)
    )
    near_state = ServiceRiskBuilder(scenario).build(
        _obs(scenario, [near], time_s=120.0)
    )

    idx = JOB_FEATURE_NAMES.index("urgency")
    assert near_state.job_features[0, idx] > early_state.job_features[0, idx]


def test_future_release_is_not_marked_released(scenario):
    request = scenario.requests[0]

    future_release = max(
        float(request.release_s) + 60.0,
        60.0,
    )

    future = _job(
        request.model_copy(
            update={"release_s": future_release}
        )
    )

    state = ServiceRiskBuilder(scenario).build(
        _obs(
            scenario,
            [future],
            time_s=future_release - 1.0,
        )
    )

    assert state.job_features[
        0,
        JOB_FEATURE_NAMES.index("released"),
    ] == pytest.approx(0.0)


def test_past_release_is_marked_released(scenario):
    request = scenario.requests[0]
    state = ServiceRiskBuilder(scenario).build(
        _obs(scenario, [_job(request)], time_s=request.release_s + 1.0)
    )
    assert state.job_features[0, JOB_FEATURE_NAMES.index("released")] == 1.0


def test_status_is_explicitly_encoded(scenario):
    request = scenario.requests[0]
    state = ServiceRiskBuilder(scenario).build(
        _obs(scenario, [_job(request, status="in_progress")])
    )
    assert state.job_features[
        0, JOB_FEATURE_NAMES.index("status_in_progress")
    ] == pytest.approx(1.0)


def test_unknown_visible_job_is_rejected(scenario):
    job = VisibleJob(
        request_id="not-in-scenario",
        demand=1.0,
        release_s=0.0,
        earliest_service_start_s=0.0,
        latest_service_start_s=100.0,
        service_duration_s=10.0,
        status="pending",
    )
    with pytest.raises(ValueError, match="does not exist"):
        ServiceRiskBuilder(scenario).build(_obs(scenario, [job]))


def test_deterministic_order_and_read_only(scenario):
    jobs = [_job(r) for r in reversed(scenario.requests)]
    state = ServiceRiskBuilder(scenario).build(_obs(scenario, jobs))
    assert state.job_ids == tuple(sorted(str(r.request_id) for r in scenario.requests))
    with pytest.raises(ValueError):
        state.job_features[0, 0] = 1.0


def test_vehicle_deadline_pressure_uses_committed_jobs_only(scenario):
    request = scenario.requests[0]
    vehicle = scenario.fleet[0]
    obs_vehicle = VehicleObservation(
        vehicle_id=vehicle.vehicle_id,
        current_edge_id=None,
        current_node_id=vehicle.start_node_id,
        distance_remaining_m=0.0,
        onboard_request_ids=(),
        remaining_load=0.0,
        executed_prefix_edge_ids=(),
        committed_request_ids=(request.request_id,),
    )
    state = ServiceRiskBuilder(scenario).build(
        _obs(scenario, [_job(request)], [obs_vehicle])
    )
    assert state.vehicle_features[0, 4] >= 0.0
    assert np.all(np.isfinite(state.vehicle_features))
