"""
Step 16.15 — PPO learning mathematics.

Contains:
    - GAE(lambda)
    - discounted returns
    - PPO clipped policy loss
    - value-function loss
    - entropy bonus
    - combined PPO loss

This module is deliberately independent of:
    - SUMO
    - QPSO
    - route planning
    - GNN internals
    - Transformer internals

It operates on rollout-level tensors only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class PPOAlgorithmConfig:
    """Numerically explicit PPO hyperparameters."""

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_epsilon: float = 0.20

    value_loss_coefficient: float = 0.50
    entropy_coefficient: float = 0.01

    max_grad_norm: float = 0.50

    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must satisfy 0 < gamma <= 1")

        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(
                "gae_lambda must satisfy 0 <= gae_lambda <= 1"
            )

        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError(
                "clip_epsilon must satisfy 0 < clip_epsilon < 1"
            )

        if self.value_loss_coefficient < 0.0:
            raise ValueError(
                "value_loss_coefficient must be non-negative"
            )

        if self.entropy_coefficient < 0.0:
            raise ValueError(
                "entropy_coefficient must be non-negative"
            )

        if self.max_grad_norm <= 0.0:
            raise ValueError(
                "max_grad_norm must be positive"
            )


# ============================================================================
# GAE
# ============================================================================


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor | None = None,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """
    Compute Generalized Advantage Estimation.

    Parameters
    ----------
    rewards:
        [T]

    values:
        V(s_t), shape [T]

    next_values:
        V(s_{t+1}), shape [T]

    terminated:
        True only for genuine terminal states. This controls whether
        V(s_{t+1}) is used in the TD target.

    truncated:
        Optional time-limit/episode-boundary flag. Truncation still permits
        bootstrapping, but it breaks the recursive GAE chain so advantages
        cannot leak across an environment reset.

    Returns
    -------
    advantages:
        [T]

    returns:
        [T]

    Notes
    -----
    `terminated` controls value bootstrapping, while
    `terminated | truncated` controls recursive GAE propagation.
    """

    if rewards.ndim != 1:
        raise ValueError("rewards must have shape [T]")

    if values.shape != rewards.shape:
        raise ValueError(
            "values must have the same shape as rewards"
        )

    if next_values.shape != rewards.shape:
        raise ValueError(
            "next_values must have the same shape as rewards"
        )

    if terminated.shape != rewards.shape:
        raise ValueError(
            "terminated must have the same shape as rewards"
        )

    if truncated is None:
        truncated = torch.zeros_like(terminated, dtype=torch.bool)
    elif truncated.shape != rewards.shape:
        raise ValueError(
            "truncated must have the same shape as rewards"
        )

    if not (
        torch.isfinite(rewards).all()
        and torch.isfinite(values).all()
        and torch.isfinite(next_values).all()
    ):
        raise ValueError(
            "rewards, values and next_values must be finite"
        )

    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must satisfy 0 < gamma <= 1")

    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(
            "gae_lambda must satisfy 0 <= gae_lambda <= 1"
        )

    terminated = terminated.to(
        device=rewards.device,
        dtype=torch.bool,
    )
    truncated = truncated.to(
        device=rewards.device,
        dtype=torch.bool,
    )

    # A transition can never be both a genuine terminal and a truncation.
    if torch.any(terminated & truncated):
        raise ValueError(
            "a transition cannot be both terminated and truncated"
        )

    advantages = torch.zeros_like(rewards)

    gae = torch.zeros(
        (),
        dtype=rewards.dtype,
        device=rewards.device,
    )

    for t in range(rewards.shape[0] - 1, -1, -1):
        # Bootstrap through truncation, but do not propagate GAE across
        # an episode reset. This separates the TD bootstrap boundary from
        # the recursive episode boundary.
        not_terminal = (~terminated[t]).to(rewards.dtype)
        not_boundary = (~(terminated[t] | truncated[t])).to(
            rewards.dtype
        )

        delta = (
            rewards[t]
            + gamma * next_values[t] * not_terminal
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * not_boundary
            * gae
        )

        advantages[t] = gae

    returns = advantages + values

    return advantages, returns


# ============================================================================
# Advantage normalization
# ============================================================================


def normalize_advantages(
    advantages: Tensor,
    *,
    epsilon: float = 1e-8,
) -> Tensor:
    """Normalize advantages for stable PPO optimization."""

    if advantages.ndim != 1:
        raise ValueError(
            "advantages must have shape [T]"
        )

    if not torch.isfinite(advantages).all():
        raise ValueError(
            "advantages must be finite"
        )

    if epsilon <= 0.0:
        raise ValueError(
            "epsilon must be positive"
        )

    mean = advantages.mean()
    std = advantages.std(unbiased=False)

    return (advantages - mean) / (
        std + epsilon
    )


# ============================================================================
# PPO losses
# ============================================================================


@dataclass(frozen=True)
class PPOLossResult:
    """Individual PPO loss components and useful diagnostics."""

    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy_bonus: Tensor

    approximate_kl: Tensor
    clip_fraction: Tensor

    ratio_mean: Tensor


def compute_ppo_loss(
    *,
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    returns: Tensor,
    new_values: Tensor,
    entropy: Tensor,
    config: PPOAlgorithmConfig,
) -> PPOLossResult:
    """
    Compute the clipped PPO objective.

    All tensors must have shape [B].
    """

    tensors = {
        "new_log_probabilities": new_log_probabilities,
        "old_log_probabilities": old_log_probabilities,
        "advantages": advantages,
        "returns": returns,
        "new_values": new_values,
        "entropy": entropy,
    }

    for name, tensor in tensors.items():
        if tensor.ndim != 1:
            raise ValueError(
                f"{name} must have shape [B]"
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains NaN or infinite values"
            )

    batch_size = new_log_probabilities.shape[0]

    if any(
        tensor.shape[0] != batch_size
        for tensor in tensors.values()
    ):
        raise ValueError(
            "all PPO loss tensors must have the same batch size"
        )

    log_ratio = (
        new_log_probabilities
        - old_log_probabilities
    )

    # Clamp only to protect exp() from numerical overflow. The actual PPO
    # clipping is performed below on the probability ratio.
    ratio = torch.exp(
        torch.clamp(
            log_ratio,
            min=-20.0,
            max=20.0,
        )
    )

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - config.clip_epsilon,
        1.0 + config.clip_epsilon,
    )

    surrogate_unclipped = (
        ratio * advantages
    )

    surrogate_clipped = (
        clipped_ratio * advantages
    )

    policy_loss = -torch.minimum(
        surrogate_unclipped,
        surrogate_clipped,
    ).mean()

    value_loss = 0.5 * (
        new_values - returns
    ).pow(2).mean()

    entropy_bonus = entropy.mean()

    total_loss = (
        policy_loss
        + config.value_loss_coefficient * value_loss
        - config.entropy_coefficient * entropy_bonus
    )

    approximate_kl = (
        old_log_probabilities
        - new_log_probabilities
    ).mean()

    clip_fraction = (
        (torch.abs(ratio - 1.0) > config.clip_epsilon)
        .float()
        .mean()
    )

    ratio_mean = ratio.mean()

    return PPOLossResult(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_bonus=entropy_bonus,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        ratio_mean=ratio_mean,
    )