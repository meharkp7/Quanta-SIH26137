import numpy as np
import pytest

from src.learning.ppo_policy_state_runtime import (
    PPOPolicyStateRuntime,
)


def test_runtime_reset_builds_structured_state(scenario):
    runtime = PPOPolicyStateRuntime(
        scenario=scenario,
    )

    result = runtime.reset(
        episode_id="episode-1",
    )

    assert result.observation.scenario_id == scenario.scenario_id
    assert result.observation.observation_time_s == 0.0
    assert result.policy_state.fused_state.observation_time_s == 0.0
    assert result.policy_state.forecast_features.shape == (12,)
    assert result.policy_state.uncertainty_features.shape == (6,)


def test_runtime_advance_moves_causal_time(scenario):
    runtime = PPOPolicyStateRuntime(
        scenario=scenario,
    )

    first = runtime.reset(
        episode_id="episode-1",
    )

    second = runtime.advance(
        observation_time_s=60.0,
        state_version="state-v1",
    )

    assert second.observation.observation_time_s == 60.0
    assert second.policy_state.fused_state.observation_time_s == 60.0
    assert (
        second.policy_state.fused_state.temporal_batch.latest
        is not None
    )
    assert (
        second.policy_state.fused_state.temporal_batch.latest
        .observation_time_s
        == pytest.approx(60.0)
    )


def test_runtime_never_uses_legacy_32d_observation(scenario):
    runtime = PPOPolicyStateRuntime(
        scenario=scenario,
    )

    result = runtime.reset(
        episode_id="episode-1",
    )

    representation = result.policy_state

    assert representation.forecast_features.shape != (32,)
    assert representation.uncertainty_features.shape != (32,)
    assert representation.fused_state.global_features.shape == (21,)


def test_runtime_accepts_explicit_forecast_and_uncertainty(scenario):
    runtime = PPOPolicyStateRuntime(
        scenario=scenario,
    )

    forecast = np.arange(12, dtype=np.float32)
    uncertainty = np.arange(6, dtype=np.float32)

    result = runtime.reset(
        episode_id="episode-1",
    )

    result = runtime.advance(
        observation_time_s=60.0,
        state_version="state-v1",
        forecast_features=forecast,
        uncertainty_features=uncertainty,
        forecast_version="forecast-v1",
        uncertainty_version="uncertainty-v1",
    )

    np.testing.assert_array_equal(
        result.policy_state.forecast_features,
        forecast,
    )

    np.testing.assert_array_equal(
        result.policy_state.uncertainty_features,
        uncertainty,
    )

    assert result.policy_state.forecast_version == "forecast-v1"
    assert result.policy_state.uncertainty_version == "uncertainty-v1"