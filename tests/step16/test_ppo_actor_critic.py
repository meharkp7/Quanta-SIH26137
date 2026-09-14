from __future__ import annotations

import numpy as np
import pytest
import torch

from src.learning.ppo_actor_critic import (
    PPOActorCritic,
    PPOActorCriticConfig,
)
from src.learning.representation import (
    RepresentationOutput,
    RepresentationSpec,
)


def make_representation() -> RepresentationOutput:
    spec = RepresentationSpec()

    return RepresentationOutput(
        spec=spec,
        spatial=np.zeros(spec.spatial_dim, dtype=np.float32),
        temporal=np.ones(spec.temporal_dim, dtype=np.float32),
        forecast=np.full(
            spec.forecast_dim,
            0.5,
            dtype=np.float32,
        ),
        uncertainty=np.full(
            spec.uncertainty_dim,
            0.1,
            dtype=np.float32,
        ),
        context=np.full(
            spec.context_dim,
            0.2,
            dtype=np.float32,
        ),
        observation_time_s=60.0,
        graph_version="graph-v1",
        representation_version=spec.version,
    )


def test_actor_critic_output_shapes() -> None:
    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    representation = make_representation()

    logits, values = model(representation)

    assert logits.shape == (1, 5)
    assert values.shape == (1, 1)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()


def test_batched_tensor_input() -> None:
    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    x = torch.randn(8, spec.total_dim)

    logits, values = model(x)

    assert logits.shape == (8, 5)
    assert values.shape == (8, 1)


def test_numpy_input() -> None:
    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    x = np.random.randn(
        4,
        spec.total_dim,
    ).astype(np.float32)

    logits, values = model(x)

    assert logits.shape == (4, 5)
    assert values.shape == (4, 1)


def test_action_mask_blocks_invalid_actions() -> None:
    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    representation = make_representation()

    mask = np.asarray(
        [True, False, True, False, False],
        dtype=bool,
    )

    distribution = model.distribution(
        representation,
        action_mask=mask,
    )

    probabilities = distribution.probs[0]

    assert probabilities[0] > 0
    assert probabilities[2] > 0

    assert probabilities[1] == 0
    assert probabilities[3] == 0
    assert probabilities[4] == 0


def test_at_least_one_action_must_be_feasible() -> None:
    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    representation = make_representation()

    mask = np.zeros(5, dtype=bool)

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        model(
            representation,
            action_mask=mask,
        )


def test_action_mask_shape_is_validated() -> None:
    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    representation = make_representation()

    bad_mask = np.ones(4, dtype=bool)

    with pytest.raises(ValueError, match="action_mask"):
        model(
            representation,
            action_mask=bad_mask,
        )


def test_sample_action_respects_mask() -> None:
    torch.manual_seed(42)

    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    representation = make_representation()

    mask = torch.tensor(
        [True, False, False, True, False],
        dtype=torch.bool,
    )

    for _ in range(100):
        action, log_prob, value = model.sample_action(
            representation,
            action_mask=mask,
        )

        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)

        assert int(action.item()) in {0, 3}


def test_evaluate_actions_returns_ppo_quantities() -> None:
    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    x = torch.randn(6, spec.total_dim)

    actions = torch.tensor(
        [0, 1, 2, 3, 4, 0],
        dtype=torch.long,
    )

    log_prob, entropy, values = model.evaluate_actions(
        x,
        actions,
    )

    assert log_prob.shape == (6,)
    assert entropy.shape == (6,)
    assert values.shape == (6,)

    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(values).all()


def test_representation_dimension_mismatch_is_rejected() -> None:
    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    bad = torch.zeros(1, spec.total_dim + 1)

    with pytest.raises(ValueError, match="dimension mismatch"):
        model(bad)


def test_non_finite_input_is_rejected() -> None:
    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    x = torch.zeros(1, spec.total_dim)
    x[0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        model(x)


def test_config_rejects_wrong_action_count() -> None:
    with pytest.raises(
        ValueError,
        match="exactly 5",
    ):
        PPOActorCriticConfig(action_dim=4)


def test_actor_and_critic_are_distinct_heads() -> None:
    spec = RepresentationSpec()
    model = PPOActorCritic(spec)

    assert model.actor.out_features == 5
    assert model.critic.out_features == 1