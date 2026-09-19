"""
Step 16.19 — PPO training-loop integration.

Coordinates:

    causal policy state
        ↓
    rollout collection
        ↓
    final policy state
        ↓
    PPO critic V(s_T)
        ↓
    GAE / PPO update

The rollout transition is the authoritative source for terminal semantics.

The final policy state is obtained from the collector only when an automatic
critic bootstrap is required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.learning.ppo_rollout import (
    PPORolloutBuffer,
)
from src.learning.ppo_rollout_collector import (
    PPORolloutCollector,
)
from src.learning.ppo_trainer import (
    PPOTrainer,
    PPOTrainingMetrics,
)


@dataclass(frozen=True)
class PPOTrainingIteration:
    """Result of one rollout-collection/update cycle."""

    transitions_collected: int
    metrics: PPOTrainingMetrics


class PPOTrainingLoop:
    """
    Coordinates fresh on-policy rollout collection and PPO updates.

    Responsibilities:
        - collect fresh transitions;
        - determine terminal/truncation semantics;
        - obtain V(s_T) when bootstrapping is required;
        - invoke PPOTrainer;
        - consume the rollout after a successful update.

    The loop does not derive PPO representations from the environment's
    legacy observation vector.
    """

    def __init__(
        self,
        *,
        collector: PPORolloutCollector,
        trainer: PPOTrainer,
        rollout_buffer: PPORolloutBuffer,
    ) -> None:
        self.collector = collector
        self.trainer = trainer
        self.rollout_buffer = rollout_buffer

        if (
            self.collector.rollout_buffer
            is not self.rollout_buffer
        ):
            raise ValueError(
                "collector and training loop must use "
                "the same rollout buffer"
            )

        if (
            self.trainer.model
            is not self.collector.controller.runtime.model
        ):
            raise ValueError(
                "trainer and collector must use the same PPO model"
            )

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(
        self,
        *,
        policy_state,
        action_mask=None,
        next_policy_state=None,
    ):
        """
        Collect one fresh environment transition.

        `next_policy_state` is the causal state after environment execution.
        It is kept separate from the environment's legacy observation vector.
        """

        return self.collector.collect_step(
            policy_state=policy_state,
            action_mask=action_mask,
            next_policy_state=next_policy_state,
        )

    # ------------------------------------------------------------------
    # Critic bootstrap
    # ------------------------------------------------------------------

    def _critic_value(
        self,
        policy_state,
    ) -> float:
        """
        Evaluate V(s) using the current PPO critic.

        The policy state crosses the model-independent representation
        boundary before entering the actor-critic.
        """

        if policy_state is None:
            raise ValueError(
                "final policy state is required to "
                "compute critic bootstrap value"
            )

        representation = (
            self.collector.representation_provider.encode(
                policy_state
            )
        )

        model = (
            self.collector.controller.runtime.model
        )

        model.eval()

        with torch.no_grad():
            _, value = model(
                representation
            )

        if not torch.isfinite(value).all():
            raise ValueError(
                "critic produced a non-finite bootstrap value"
            )

        return float(
            value.squeeze(-1).item()
        )

    # ------------------------------------------------------------------
    # Bootstrap resolution
    # ------------------------------------------------------------------

    def _resolve_bootstrap_value(
        self,
        last_transition,
        explicit_bootstrap_value: float | None,
    ) -> float:
        """
        Resolve V(s_T) according to the final transition.

        Genuine terminal:
            bootstrap = 0

        Truncated/non-terminal:
            use explicit bootstrap if supplied;
            otherwise evaluate the final causal policy state with the
            current critic.
        """

        terminated = bool(
            last_transition.terminated
        )

        truncated = bool(
            last_transition.truncated
        )

        if terminated and truncated:
            raise RuntimeError(
                "rollout cannot end with both terminated "
                "and truncated"
            )

        # Genuine terminal state.
        if terminated:
            if (
                explicit_bootstrap_value is not None
                and abs(
                    float(explicit_bootstrap_value)
                ) > 1e-12
            ):
                raise ValueError(
                    "terminated rollout must use "
                    "bootstrap_value=0"
                )

            return 0.0

        # Explicit value supplied by the integration layer.
        if explicit_bootstrap_value is not None:
            resolved = float(
                explicit_bootstrap_value
            )

            if not torch.isfinite(
                torch.tensor(resolved)
            ):
                raise ValueError(
                    "bootstrap_value must be finite"
                )

            return resolved

        # No explicit value: evaluate V(s_T).
        return self._critic_value(
            self.collector.last_policy_state
        )

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        bootstrap_value: float | None = None,
    ) -> PPOTrainingIteration:
        """
        Perform one PPO update.

        Terminal semantics come directly from the final rollout transition.

        For a genuine terminal state:
            V(s_T) = 0

        For a truncated/non-terminal state:
            V(s_T) is evaluated by the current critic unless an explicit
            bootstrap value is supplied.

        The rollout is cleared only after a successful trainer update.
        """

        transitions_collected = (
            self.rollout_buffer.size
        )

        if transitions_collected == 0:
            raise RuntimeError(
                "cannot update PPO with an empty rollout"
            )

        last_transition = (
            self.rollout_buffer.transitions()[-1]
        )

        resolved_bootstrap = (
            self._resolve_bootstrap_value(
                last_transition,
                bootstrap_value,
            )
        )

        metrics = self.trainer.update(
            self.rollout_buffer,
            bootstrap_value=resolved_bootstrap,
        )

        # PPOTrainer owns rollout-buffer consumption after a successful
        # optimization step. The loop only clears collector-local state.
        self.collector.clear()

        return PPOTrainingIteration(
            transitions_collected=transitions_collected,
            metrics=metrics,
        )