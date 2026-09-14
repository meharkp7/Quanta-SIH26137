"""
Step 16.16 — PPO optimizer/trainer.

Consumes:
    - PPOActorCritic
    - PPORolloutBuffer
    - PPOAlgorithmConfig

Produces:
    - policy/value updates
    - training diagnostics

The trainer knows nothing about:
    - SUMO
    - QPSO
    - routing algorithms
    - GNN architecture
    - Transformer architecture
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import Tensor, nn

from src.learning.ppo_actor_critic import PPOActorCritic
from src.learning.ppo_algorithm import (
    PPOAlgorithmConfig,
    compute_gae,
    compute_ppo_loss,
    normalize_advantages,
)
from src.learning.ppo_rollout import PPORolloutBuffer


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class PPOTrainerConfig:
    """Optimizer/training-loop configuration."""

    learning_rate: float = 3e-4

    epochs_per_rollout: int = 4
    minibatch_size: int = 64

    seed: int = 42

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.epochs_per_rollout <= 0:
            raise ValueError(
                "epochs_per_rollout must be positive"
            )

        if self.minibatch_size <= 0:
            raise ValueError(
                "minibatch_size must be positive"
            )


# ============================================================================
# Training metrics
# ============================================================================


@dataclass(frozen=True)
class PPOTrainingMetrics:
    """Aggregate diagnostics from one PPO update."""

    updates: int

    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float

    approximate_kl: float
    clip_fraction: float
    ratio_mean: float

    mean_reward: float
    mean_advantage: float
    mean_return: float

    gradient_norm: float


# ============================================================================
# Trainer
# ============================================================================


class PPOTrainer:
    """
    Performs PPO updates using an existing actor-critic network.

    One call to update() consumes one on-policy rollout and then the rollout
    must be collected again using the updated policy.
    """

    def __init__(
        self,
        model: PPOActorCritic,
        *,
        algorithm_config: PPOAlgorithmConfig | None = None,
        trainer_config: PPOTrainerConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.algorithm_config = (
            algorithm_config
            or PPOAlgorithmConfig()
        )
        self.trainer_config = (
            trainer_config
            or PPOTrainerConfig()
        )

        self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.trainer_config.learning_rate,
            eps=1e-5,
        )

        self._rng = np.random.default_rng(
            self.trainer_config.seed
        )

    # ------------------------------------------------------------------
    # Batch iteration
    # ------------------------------------------------------------------

    def _minibatches(
        self,
        size: int,
    ) -> Iterator[np.ndarray]:
        indices = np.arange(size)

        for _ in range(
            self.trainer_config.epochs_per_rollout
        ):
            self._rng.shuffle(indices)

            for start in range(
                0,
                size,
                self.trainer_config.minibatch_size,
            ):
                yield indices[
                    start : start
                    + self.trainer_config.minibatch_size
                ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_rollout(
        arrays: dict[str, np.ndarray],
    ) -> None:
        required = {
            "observations",
            "actions",
            "log_probabilities",
            "values",
            "rewards",
            "terminated",
            "truncated",
            "action_masks",
        }

        missing = required - arrays.keys()

        if missing:
            raise ValueError(
                f"rollout is missing fields: {sorted(missing)}"
            )

        size = arrays["actions"].shape[0]

        if size == 0:
            raise ValueError(
                "rollout must contain at least one transition"
            )

        for name in (
            "log_probabilities",
            "values",
            "rewards",
        ):
            if arrays[name].shape != (size,):
                raise ValueError(
                    f"{name} must have shape ({size},)"
                )

            if not np.all(
                np.isfinite(arrays[name])
            ):
                raise ValueError(
                    f"{name} contains non-finite values"
                )

        if arrays["observations"].shape[0] != size:
            raise ValueError(
                "observations and actions have different lengths"
            )

        if arrays["action_masks"].shape != (
            size,
            5,
        ):
            raise ValueError(
                "action_masks must have shape [T, 5]"
            )

        if not arrays["action_masks"].any(axis=1).all():
            raise ValueError(
                "every transition must have at least "
                "one feasible action"
            )

        actions = arrays["actions"]

        if not np.all(
            (actions >= 0) & (actions < 5)
        ):
            raise ValueError(
                "actions must be in [0, 4]"
            )

        if not np.all(
            arrays["action_masks"][
                np.arange(size),
                actions,
            ]
        ):
            raise ValueError(
                "rollout contains an action that was "
                "not feasible under its stored mask"
            )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        rollout: PPORolloutBuffer,
        *,
        bootstrap_value: float = 0.0,
    ) -> PPOTrainingMetrics:
        """
        Perform PPO optimization on one collected rollout.

        `bootstrap_value` is V(s_T) for the state following the final stored
        transition when that state is not genuinely terminal.

        The environment integration layer is responsible for supplying this
        value from the final next observation.
        """

        arrays = rollout.as_arrays()

        self._validate_rollout(arrays)

        observations = torch.as_tensor(
            arrays["observations"],
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            arrays["actions"],
            dtype=torch.long,
            device=self.device,
        )

        old_log_probabilities = torch.as_tensor(
            arrays["log_probabilities"],
            dtype=torch.float32,
            device=self.device,
        )

        values = torch.as_tensor(
            arrays["values"],
            dtype=torch.float32,
            device=self.device,
        )

        rewards = torch.as_tensor(
            arrays["rewards"],
            dtype=torch.float32,
            device=self.device,
        )

        terminated = torch.as_tensor(
            arrays["terminated"],
            dtype=torch.bool,
            device=self.device,
        )

        # Build V(s_{t+1}).
        #
        # For transitions before the final one, the next value is the value
        # stored by the following transition.
        #
        # For the final transition, use the explicitly supplied bootstrap
        # value. The environment integration layer will set this to zero for
        # a genuine terminal state.
        next_values = torch.empty_like(values)

        if values.shape[0] > 1:
            next_values[:-1] = values[1:]

        if terminated[-1]:
            next_values[-1] = 0.0
        else:
            if not np.isfinite(bootstrap_value):
                raise ValueError(
                    "bootstrap_value must be finite"
                )

            next_values[-1] = float(
                bootstrap_value
            )

        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            gamma=self.algorithm_config.gamma,
            gae_lambda=self.algorithm_config.gae_lambda,
        )

        if (
            self.algorithm_config.normalize_advantages
        ):
            advantages = normalize_advantages(
                advantages
            )

        action_masks = torch.as_tensor(
            arrays["action_masks"],
            dtype=torch.bool,
            device=self.device,
        )

        self.model.train()

        total_losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clip_fractions: list[float] = []
        ratios: list[float] = []
        gradient_norms: list[float] = []

        for batch_indices in self._minibatches(
            observations.shape[0]
        ):
            idx = torch.as_tensor(
                batch_indices,
                dtype=torch.long,
                device=self.device,
            )

            batch_observations = observations[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = (
                old_log_probabilities[idx]
            )
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]
            batch_masks = action_masks[idx]

            new_log_probs, entropy, new_values = (
                self.model.evaluate_actions(
                    batch_observations,
                    batch_actions,
                    action_mask=batch_masks,
                )
            )

            result = compute_ppo_loss(
                new_log_probabilities=new_log_probs,
                old_log_probabilities=batch_old_log_probs,
                advantages=batch_advantages,
                returns=batch_returns,
                new_values=new_values,
                entropy=entropy,
                config=self.algorithm_config,
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            result.total_loss.backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.algorithm_config.max_grad_norm,
            )

            if not torch.isfinite(
                gradient_norm
            ):
                self.optimizer.zero_grad(
                    set_to_none=True
                )
                raise FloatingPointError(
                    "non-finite gradient norm during PPO update"
                )

            self.optimizer.step()

            total_losses.append(
                result.total_loss.detach().item()
            )

            policy_losses.append(
                result.policy_loss.detach().item()
            )

            value_losses.append(
                result.value_loss.detach().item()
            )

            entropies.append(
                result.entropy_bonus.detach().item()
            )

            kls.append(
                result.approximate_kl.detach().item()
            )

            clip_fractions.append(
                result.clip_fraction.detach().item()
            )

            ratios.append(
                result.ratio_mean.detach().item()
            )

            gradient_norms.append(
                float(gradient_norm.detach().item())
            )

        # Rollout is on-policy and must not be reused after an update.
        rollout.clear()

        return PPOTrainingMetrics(
            updates=len(total_losses),
            total_loss=float(
                np.mean(total_losses)
            ),
            policy_loss=float(
                np.mean(policy_losses)
            ),
            value_loss=float(
                np.mean(value_losses)
            ),
            entropy=float(
                np.mean(entropies)
            ),
            approximate_kl=float(
                np.mean(kls)
            ),
            clip_fraction=float(
                np.mean(clip_fractions)
            ),
            ratio_mean=float(
                np.mean(ratios)
            ),
            mean_reward=float(
                rewards.mean().detach().item()
            ),
            mean_advantage=float(
                advantages.mean().detach().item()
            ),
            mean_return=float(
                returns.mean().detach().item()
            ),
            gradient_norm=float(
                np.mean(gradient_norms)
            ),
        )