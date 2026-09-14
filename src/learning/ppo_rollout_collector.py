"""
Step 16.19 — PPO rollout collection with final policy-state tracking.

Bridges PPO policy inference with the existing environment and rollout
buffer without mixing training logic into the environment.

The collector additionally retains the latest causal policy state so the
training loop can evaluate the final state with the PPO critic when a
rollout is truncated or otherwise ends without a genuine terminal state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.learning.ppo_controller import (
    PPOController,
    PPOControllerDecision,
)
from src.learning.ppo_env import (
    TrafficRoutingPPOEnv,
)
from src.learning.ppo_rollout import (
    PPOTransition,
    PPORolloutBuffer,
)
from src.learning.representation import (
    RepresentationProvider,
)


@dataclass(frozen=True)
class PPORolloutStep:
    """Result of collecting one PPO environment transition."""

    transition: PPOTransition
    decision: PPOControllerDecision
    observation: np.ndarray
    next_observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class PPORolloutCollector:
    """
    Collect on-policy PPO transitions from TrafficRoutingPPOEnv.

    Responsibilities:
        1. encode the current policy state;
        2. obtain PPO action/log-probability/value;
        3. execute that action through the environment;
        4. create PPOTransition;
        5. append it to the rollout buffer;
        6. retain the latest causal policy state for bootstrapping.

    Training, GAE, and optimizer updates remain outside this class.
    """

    def __init__(
        self,
        *,
        env: TrafficRoutingPPOEnv,
        controller: PPOController,
        representation_provider: RepresentationProvider,
        rollout_buffer: PPORolloutBuffer,
    ) -> None:
        self.env = env
        self.controller = controller
        self.representation_provider = representation_provider
        self.rollout_buffer = rollout_buffer

        self._last_policy_state: Any | None = None
        self._last_transition_terminated = False
        self._last_transition_truncated = False

        self._validate_contracts()

    def _validate_contracts(self) -> None:
        """Validate that controller and representation agree."""

        controller_spec = (
            self.controller.runtime.model.representation_spec
        )

        provider_spec = self.representation_provider.spec

        if controller_spec != provider_spec:
            raise ValueError(
                "representation provider spec does not match "
                "PPO controller model spec"
            )

    @staticmethod
    def _normalize_mask(
        action_mask: Any,
    ) -> np.ndarray:
        """Normalize an action mask to a five-element boolean array."""

        if hasattr(action_mask, "values"):
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

        if not bool(values.any()):
            raise ValueError(
                "action mask must allow at least one action"
            )

        return values.copy()

    def collect_step(
        self,
        *,
        policy_state: Any,
        action_mask: Any | None = None,
        next_policy_state: Any | None = None,
    ) -> PPORolloutStep:
        """
        Collect and store exactly one PPO transition.

        Parameters
        ----------
        policy_state:
            Current causal state supplied to the representation provider.

        action_mask:
            Optional five-action feasibility mask.

        next_policy_state:
            Causal state corresponding to the environment state after this
            transition. Required when the caller wants automatic critic
            bootstrapping after a non-terminal transition.

            It is intentionally supplied by the orchestration layer rather
            than inferred from the environment's legacy 32-D observation.
        """

        self._last_policy_state = (
            policy_state
        )

        representation = (
            self.representation_provider.encode(
                policy_state
            )
        )

        resolved_action_mask = (
            self.env._action_mask()
            if action_mask is None
            else action_mask
        )

        normalized_mask = self._normalize_mask(
            resolved_action_mask
        )

        decision = self.controller.decide_from_representation(
            representation,
            action_mask=normalized_mask,
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(
            decision.action_index
        )

        if terminated and truncated:
            raise RuntimeError(
                "environment cannot mark a transition as both "
                "terminated and truncated"
            )

        self._last_transition_terminated = bool(
            terminated
        )
        self._last_transition_truncated = bool(
            truncated
        )

        if next_policy_state is not None:
            self._last_policy_state = (
                next_policy_state
            )

        observation = representation.flat.copy()

        next_observation_array = np.asarray(
            next_observation,
            dtype=np.float32,
        ).copy()

        transition = PPOTransition(
            observation=observation,
            action=int(
                decision.action_index
            ),
            log_probability=float(
                decision.log_probability
            ),
            value=float(
                decision.value
            ),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            action_mask=normalized_mask,
        )

        self.rollout_buffer.append(
            transition
        )

        enriched_info = dict(info)

        enriched_info.update(
            {
                "ppo_collection_action": int(
                    decision.action_index
                ),
                "ppo_collection_log_probability": float(
                    decision.log_probability
                ),
                "ppo_collection_value": float(
                    decision.value
                ),
            }
        )

        return PPORolloutStep(
            transition=transition,
            decision=decision,
            observation=observation,
            next_observation=next_observation_array,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=enriched_info,
        )

    @property
    def last_policy_state(self) -> Any | None:
        """Return the latest causal policy state."""

        return self._last_policy_state

    @property
    def last_transition_terminated(self) -> bool:
        """Whether the latest transition genuinely terminated."""

        return self._last_transition_terminated

    @property
    def last_transition_truncated(self) -> bool:
        """Whether the latest transition was truncated."""

        return self._last_transition_truncated

    def buffer_size(self) -> int:
        """Return the number of transitions currently collected."""

        return len(
            self.rollout_buffer.transitions()
        )

    def clear(self) -> None:
        """Clear rollout data and final-state metadata."""

        self.rollout_buffer.clear()

        self._last_policy_state = None
        self._last_transition_terminated = False
        self._last_transition_truncated = False