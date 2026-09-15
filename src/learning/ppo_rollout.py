"""
Step 16.14 — PPO rollout storage.

Stores causal on-policy transitions collected from the routing environment.

A rollout entry contains the information required later for:
    - PPO policy-ratio calculation
    - value-function targets
    - GAE advantage estimation
    - action-mask reconstruction
    - rollout auditing

This module deliberately contains no SUMO, QPSO, GNN, or Transformer logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================================
# Single transition
# ============================================================================


@dataclass(frozen=True)
class PPOTransition:
    """
    One environment transition.

    Shapes:
        observation: [D]
        action_mask: [5]

    Scalars:
        action
        log_probability
        value
        reward
        terminated
        truncated
    """

    observation: np.ndarray
    action: int
    log_probability: float
    value: float
    reward: float
    terminated: bool
    truncated: bool
    action_mask: np.ndarray

    def __post_init__(self) -> None:
        observation = np.asarray(
            self.observation,
            dtype=np.float32,
        )

        action_mask = np.asarray(
            self.action_mask,
            dtype=bool,
        )

        if observation.ndim != 1:
            raise ValueError(
                "observation must be one-dimensional"
            )

        if observation.size == 0:
            raise ValueError(
                "observation must not be empty"
            )

        if not np.all(np.isfinite(observation)):
            raise ValueError(
                "observation contains NaN or infinite values"
            )

        if action_mask.shape != (5,):
            raise ValueError(
                "action_mask must have shape (5,)"
            )

        if not action_mask.any():
            raise ValueError(
                "at least one action must be feasible"
            )

        if not 0 <= int(self.action) < 5:
            raise ValueError(
                "action must be one of 0, 1, 2, 3, 4"
            )

        if not action_mask[int(self.action)]:
            raise ValueError(
                "recorded action is not feasible under action_mask"
            )

        if not np.isfinite(self.log_probability):
            raise ValueError(
                "log_probability must be finite"
            )

        if not np.isfinite(self.value):
            raise ValueError(
                "value must be finite"
            )

        if not np.isfinite(self.reward):
            raise ValueError(
                "reward must be finite"
            )

        observation.setflags(write=False)
        action_mask.setflags(write=False)

        object.__setattr__(
            self,
            "observation",
            observation,
        )

        object.__setattr__(
            self,
            "action_mask",
            action_mask,
        )


# ============================================================================
# Rollout buffer
# ============================================================================


class PPORolloutBuffer:
    """
    Fixed-capacity on-policy rollout buffer.

    The buffer is intentionally simple:
        append transitions
        inspect collected transitions
        convert them to arrays
        clear after an update

    No advantage calculation is performed here. That belongs to the PPO
    learning algorithm so that rollout storage remains independent of the
    optimization implementation.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(
                "capacity must be positive"
            )

        self.capacity = int(capacity)
        self._transitions: list[PPOTransition] = []

    @property
    def size(self) -> int:
        return len(self._transitions)

    @property
    def full(self) -> bool:
        return self.size >= self.capacity

    @property
    def empty(self) -> bool:
        return self.size == 0

    def append(
        self,
        transition: PPOTransition,
    ) -> None:
        if self.full:
            raise RuntimeError(
                "rollout buffer is full"
            )

        self._transitions.append(transition)

    def extend(
        self,
        transitions: list[PPOTransition],
    ) -> None:
        if self.size + len(transitions) > self.capacity:
            raise RuntimeError(
                "adding transitions would exceed buffer capacity"
            )

        for transition in transitions:
            self.append(transition)

    def transitions(self) -> tuple[PPOTransition, ...]:
        """
        Return an immutable view of the collected transitions.
        """

        return tuple(self._transitions)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """
        Convert the rollout into dense arrays.

        Returns:
            observations:      [T, D]
            actions:           [T]
            log_probabilities: [T]
            values:            [T]
            rewards:           [T]
            terminated:        [T]
            truncated:         [T]
            action_masks:      [T, 5]
        """

        if self.empty:
            raise RuntimeError(
                "cannot convert an empty rollout to arrays"
            )

        observations = np.stack(
            [t.observation for t in self._transitions],
            axis=0,
        )

        actions = np.asarray(
            [t.action for t in self._transitions],
            dtype=np.int64,
        )

        log_probabilities = np.asarray(
            [
                t.log_probability
                for t in self._transitions
            ],
            dtype=np.float32,
        )

        values = np.asarray(
            [t.value for t in self._transitions],
            dtype=np.float32,
        )

        rewards = np.asarray(
            [t.reward for t in self._transitions],
            dtype=np.float32,
        )

        terminated = np.asarray(
            [t.terminated for t in self._transitions],
            dtype=bool,
        )

        truncated = np.asarray(
            [t.truncated for t in self._transitions],
            dtype=bool,
        )

        action_masks = np.stack(
            [t.action_mask for t in self._transitions],
            axis=0,
        )

        return {
            "observations": observations,
            "actions": actions,
            "log_probabilities": log_probabilities,
            "values": values,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "action_masks": action_masks,
        }

    def clear(self) -> None:
        self._transitions.clear()