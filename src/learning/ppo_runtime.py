"""
Step 16.17 — PPO runtime integration.

This module connects the model-independent PPO policy to the routing
environment without exposing GNN/Transformer internals to PPO.

Responsibilities:
    1. Encode the current environment state.
    2. Build/apply the feasibility mask.
    3. Query actor + critic.
    4. Sample an action.
    5. Return the information required by the environment and rollout buffer.

Responsibilities deliberately NOT here:
    - SUMO stepping
    - QPSO execution
    - route planning
    - safety override execution
    - reward calculation

Those remain environment/runtime responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.learning.action_mask import ActionMask
from src.learning.ppo_actor_critic import PPOActorCritic
from src.learning.representation import (
    RepresentationOutput,
    RepresentationProvider,
    validate_representation_provider,
)


# ============================================================================
# Runtime decision
# ============================================================================


@dataclass(frozen=True)
class PPOPolicyDecision:
    """One policy decision before environment execution."""

    action: int
    log_probability: float
    value: float

    representation: RepresentationOutput
    action_mask: np.ndarray


# ============================================================================
# Runtime
# ============================================================================


class PPORuntime:
    """
    Production-facing PPO policy runtime.

    The runtime owns inference only. Training remains in PPOTrainer.

    This separation is important:
        inference/runtime != optimizer/training
    """

    def __init__(
        self,
        model: PPOActorCritic,
        representation_provider: RepresentationProvider,
        *,
        device: torch.device | str = "cpu",
        deterministic: bool = False,
    ) -> None:
        self.model = model
        self.representation_provider = (
            representation_provider
        )

        self.device = torch.device(device)
        self.deterministic = deterministic

        spec = validate_representation_provider(
            representation_provider
        )

        if spec != model.representation_spec:
            raise ValueError(
                "representation provider spec does not match "
                "actor-critic representation spec"
            )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mask_to_numpy(
        action_mask: ActionMask | np.ndarray,
    ) -> np.ndarray:
        """
        Normalize the project's ActionMask representation to [5] bool.

        Supports the existing ActionMask contract as well as a raw mask for
        lightweight integration/testing.
        """

        if isinstance(action_mask, ActionMask):
            values = np.asarray(
                action_mask.values,
                dtype=bool,
            )
        else:
            values = np.asarray(
                action_mask,
                dtype=bool,
            )

        if values.shape != (5,):
            raise ValueError(
                "action mask must have shape (5,)"
            )

        if not values.any():
            raise ValueError(
                "at least one PPO action must be feasible"
            )

        return values

    def decide(
        self,
        state: object,
        action_mask: ActionMask | np.ndarray,
    ) -> PPOPolicyDecision:
        """
        Produce one policy decision.

        `state` must represent only the current causal environment state.
        """

        representation = (
            self.representation_provider.encode(
                state
            )
        )

        mask = self._mask_to_numpy(
            action_mask
        )

        with torch.no_grad():
            logits, value = self.model(
                representation,
                action_mask=mask,
            )

            distribution = torch.distributions.Categorical(
                logits=logits
            )

            if self.deterministic:
                action = torch.argmax(
                    logits,
                    dim=-1,
                )
            else:
                action = distribution.sample()

            log_probability = distribution.log_prob(
                action
            )

        return PPOPolicyDecision(
            action=int(action.item()),
            log_probability=float(
                log_probability.item()
            ),
            value=float(
                value.squeeze(-1).item()
            ),
            representation=representation,
            action_mask=mask.copy(),
        )