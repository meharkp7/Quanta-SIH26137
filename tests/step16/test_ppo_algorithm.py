from __future__ import annotations

import pytest
import torch

from src.learning.ppo_algorithm import (
    PPOAlgorithmConfig,
    compute_gae,
    compute_ppo_loss,
    normalize_advantages,
)


def test_gae_single_step() -> None:
    rewards = torch.tensor([2.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([1.0])
    terminated = torch.tensor([False])

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        gamma=0.9,
        gae_lambda=0.95,
    )

    expected_delta = 2.0 + 0.9 * 1.0 - 0.5

    assert advantages.shape == (1,)
    assert returns.shape == (1,)

    assert advantages.item() == pytest.approx(
        expected_delta
    )

    assert returns.item() == pytest.approx(
        expected_delta + 0.5
    )


def test_terminal_state_does_not_bootstrap() -> None:
    rewards = torch.tensor([2.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([100.0])
    terminated = torch.tensor([True])

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages.item() == pytest.approx(1.5)
    assert returns.item() == pytest.approx(2.0)


def test_truncated_state_can_bootstrap() -> None:
    rewards = torch.tensor([2.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([1.0])

    # Truncation is intentionally NOT represented as terminated.
    terminated = torch.tensor([False])

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert advantages.item() == pytest.approx(2.0 + 0.9 - 0.5)
    assert returns.item() == pytest.approx(
        2.0 + 0.9
    )


def test_gae_resets_across_terminal_boundary() -> None:
    rewards = torch.tensor(
        [1.0, 10.0]
    )

    values = torch.tensor(
        [0.0, 0.0]
    )

    next_values = torch.tensor(
        [10.0, 0.0]
    )

    terminated = torch.tensor(
        [True, False]
    )

    advantages, _ = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert advantages[0].item() == pytest.approx(1.0)
    assert advantages[1].item() == pytest.approx(10.0)


def test_advantage_normalization() -> None:
    advantages = torch.tensor(
        [1.0, 2.0, 3.0, 4.0]
    )

    normalized = normalize_advantages(
        advantages
    )

    assert normalized.mean().item() == pytest.approx(
        0.0,
        abs=1e-6,
    )

    assert normalized.std(
        unbiased=False
    ).item() == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_ppo_equal_policies_have_ratio_one() -> None:
    old_log_probs = torch.tensor(
        [-1.0, -0.5, -2.0]
    )

    new_log_probs = old_log_probs.clone()

    advantages = torch.tensor(
        [1.0, 2.0, -1.0]
    )

    returns = torch.tensor(
        [1.0, 2.0, 0.0]
    )

    values = torch.tensor(
        [0.5, 1.5, 0.5]
    )

    entropy = torch.tensor(
        [0.5, 0.5, 0.5]
    )

    result = compute_ppo_loss(
        new_log_probabilities=new_log_probs,
        old_log_probabilities=old_log_probs,
        advantages=advantages,
        returns=returns,
        new_values=values,
        entropy=entropy,
        config=PPOAlgorithmConfig(),
    )

    assert result.ratio_mean.item() == pytest.approx(
        1.0
    )

    assert result.clip_fraction.item() == pytest.approx(
        0.0
    )

    assert torch.isfinite(
        result.total_loss
    )


def test_large_policy_change_is_clipped() -> None:
    old_log_probs = torch.tensor(
        [-1.0, -1.0]
    )

    new_log_probs = torch.tensor(
        [0.0, 0.0]
    )

    advantages = torch.tensor(
        [1.0, 1.0]
    )

    returns = torch.tensor(
        [1.0, 1.0]
    )

    values = torch.tensor(
        [1.0, 1.0]
    )

    entropy = torch.tensor(
        [0.0, 0.0]
    )

    result = compute_ppo_loss(
        new_log_probabilities=new_log_probs,
        old_log_probabilities=old_log_probs,
        advantages=advantages,
        returns=returns,
        new_values=values,
        entropy=entropy,
        config=PPOAlgorithmConfig(
            clip_epsilon=0.2,
        ),
    )

    assert result.clip_fraction.item() == pytest.approx(
        1.0
    )


def test_negative_advantage_uses_clipped_surrogate_correctly() -> None:
    old_log_probs = torch.tensor(
        [-1.0]
    )

    new_log_probs = torch.tensor(
        [-0.2]
    )

    advantages = torch.tensor(
        [-2.0]
    )

    returns = torch.tensor(
        [0.0]
    )

    values = torch.tensor(
        [0.0]
    )

    entropy = torch.tensor(
        [0.0]
    )

    result = compute_ppo_loss(
        new_log_probabilities=new_log_probs,
        old_log_probabilities=old_log_probs,
        advantages=advantages,
        returns=returns,
        new_values=values,
        entropy=entropy,
        config=PPOAlgorithmConfig(
            clip_epsilon=0.2
        ),
    )

    assert torch.isfinite(
        result.policy_loss
    )


def test_non_finite_ppo_input_is_rejected() -> None:
    old_log_probs = torch.tensor(
        [float("nan")]
    )

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        compute_ppo_loss(
            new_log_probabilities=torch.tensor(
                [-1.0]
            ),
            old_log_probabilities=old_log_probs,
            advantages=torch.tensor([1.0]),
            returns=torch.tensor([1.0]),
            new_values=torch.tensor([0.0]),
            entropy=torch.tensor([0.0]),
            config=PPOAlgorithmConfig(),
        )


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        PPOAlgorithmConfig(gamma=0.0)

    with pytest.raises(ValueError):
        PPOAlgorithmConfig(
            clip_epsilon=1.0
        )