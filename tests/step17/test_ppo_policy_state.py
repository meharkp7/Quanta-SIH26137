from __future__ import annotations

import numpy as np
import pytest

from src.contracts.observation import Observation
from src.learning.ppo_policy_state import (
    PPOPolicyState,
    PPOPolicyStateBuilder,
    StructuredPPORepresentationProvider,
)


def _observation(
    scenario,
    *,
    time_s: float = 120.0,
    state_version: str = "1",
):
    return Observation(
        scenario_id=scenario.scenario_id,
        episode_id="episode-step17",
        observation_time_s=time_s,
        graph_version=scenario.graph_version,
        edge_observations=(),
        visible_jobs=(),
        fleet=(),
        visible_events=(),
        pending_request_ids=(),
        state_version=state_version,
    )


def _build(
    scenario,
    *,
    time_s: float = 120.0,
    state_version: str = "1",
):
    builder = PPOPolicyStateBuilder(
        scenario,
        history_length=4,
    )

    observation = _observation(
        scenario,
        time_s=time_s,
        state_version=state_version,
    )

    state = builder.build(
        observation,
        forecast_features=np.arange(
            12,
            dtype=np.float32,
        ),
        uncertainty_features=np.arange(
            6,
            dtype=np.float32,
        ),
        forecast_version="persistence-v1",
        uncertainty_version="uncertainty-v1",
    )

    return builder, state


def test_policy_state_is_structured_and_causal(
    scenario,
):
    _, state = _build(
        scenario
    )

    assert isinstance(
        state,
        PPOPolicyState,
    )

    assert state.observation_time_s == pytest.approx(
        120.0
    )

    assert state.fused_state.temporal_batch.latest is not None

    assert (
        state.fused_state.temporal_batch.latest.observation_time_s
        == pytest.approx(120.0)
    )


def test_policy_state_does_not_use_legacy_32d_observation(
    scenario,
):
    _, state = _build(
        scenario
    )

    assert state.fused_state.global_features.shape == (
        21,
    )

    provider = StructuredPPORepresentationProvider()

    representation = provider.encode(
        state
    )

    assert representation.flat.shape == (
        332,
    )


def test_forecast_and_uncertainty_are_explicit(
    scenario,
):
    _, state = _build(
        scenario
    )

    np.testing.assert_array_equal(
        state.forecast_features,
        np.arange(
            12,
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        state.uncertainty_features,
        np.arange(
            6,
            dtype=np.float32,
        ),
    )


def test_representation_metadata_preserves_model_versions(
    scenario,
):
    _, state = _build(
        scenario
    )

    provider = StructuredPPORepresentationProvider()

    representation = provider.encode(
        state
    )

    assert representation.metadata[
        "forecast_version"
    ] == "persistence-v1"

    assert representation.metadata[
        "uncertainty_version"
    ] == "uncertainty-v1"


def test_second_observation_extends_temporal_history(
    scenario,
):
    builder = PPOPolicyStateBuilder(
        scenario,
        history_length=4,
    )

    first = _observation(
        scenario,
        time_s=120.0,
        state_version="1",
    )

    second = _observation(
        scenario,
        time_s=180.0,
        state_version="2",
    )

    builder.build(
        first,
        forecast_features=np.zeros(
            12,
            dtype=np.float32,
        ),
        uncertainty_features=np.zeros(
            6,
            dtype=np.float32,
        ),
        forecast_version="persistence-v1",
        uncertainty_version="uncertainty-v1",
    )

    state = builder.build(
        second,
        forecast_features=np.zeros(
            12,
            dtype=np.float32,
        ),
        uncertainty_features=np.zeros(
            6,
            dtype=np.float32,
        ),
        forecast_version="persistence-v1",
        uncertainty_version="uncertainty-v1",
    )

    temporal = state.fused_state.temporal_batch

    assert temporal.valid_count == 2

    assert temporal.latest is not None

    assert temporal.latest.observation_time_s == pytest.approx(
        180.0
    )

    assert temporal.latest.delta_time_s == pytest.approx(
        60.0
    )


def test_invalid_forecast_dimension_is_rejected(
    scenario,
):
    builder = PPOPolicyStateBuilder(
        scenario
    )

    observation = _observation(
        scenario
    )

    with pytest.raises(ValueError):
        builder.build(
            observation,
            forecast_features=np.zeros(
                11,
                dtype=np.float32,
            ),
            uncertainty_features=np.zeros(
                6,
                dtype=np.float32,
            ),
            forecast_version="persistence-v1",
            uncertainty_version="uncertainty-v1",
        )


def test_invalid_uncertainty_dimension_is_rejected(
    scenario,
):
    builder = PPOPolicyStateBuilder(
        scenario
    )

    observation = _observation(
        scenario
    )

    with pytest.raises(ValueError):
        builder.build(
            observation,
            forecast_features=np.zeros(
                12,
                dtype=np.float32,
            ),
            uncertainty_features=np.zeros(
                5,
                dtype=np.float32,
            ),
            forecast_version="uncertainty-v1",
            uncertainty_version="uncertainty-v1",
        )