"""
Step 16.17 — PPO controller integration boundary.

Connects:

    causal state
        ↓
    RepresentationProvider
        ↓
    PPORuntime
        ↓
    PPOPolicyDecision
        ↓
    requested ScopeAction

The controller does not execute routing decisions.

Safety overrides, exact scope selection, QPSO, SUMO, and reward calculation
remain downstream responsibilities of the environment/runtime stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.contracts.decision import ScopeAction

from src.learning.action_mask import ActionMask
from src.learning.ppo_env import (
    ACTION_TO_INDEX,
    INDEX_TO_ACTION,
)
from src.learning.ppo_runtime import (
    PPOPolicyDecision,
    PPORuntime,
)
from src.learning.representation import (
    RepresentationProvider,
    validate_representation_provider,
)


@dataclass(frozen=True)
class PPOControllerDecision:
    """
    Immutable policy proposal.

    `requested_action` is what PPO selected.

    There is deliberately no `executed_action` here. The executed action
    belongs to the downstream environment/safety layer.
    """

    requested_action: ScopeAction
    action_index: int
    log_probability: float
    value: float
    policy_decision: PPOPolicyDecision

    def __post_init__(self) -> None:
        if self.action_index not in INDEX_TO_ACTION:
            raise ValueError(
                f"invalid action_index={self.action_index}"
            )

        if (
            INDEX_TO_ACTION[self.action_index]
            != self.requested_action
        ):
            raise ValueError(
                "action_index and requested_action do not agree"
            )

        if not np.isfinite(
            float(self.log_probability)
        ):
            raise ValueError(
                "log_probability must be finite"
            )

        if not np.isfinite(
            float(self.value)
        ):
            raise ValueError(
                "value must be finite"
            )


class PPOController:
    """
    Thin adapter between the representation boundary and PPO runtime.

    The controller owns no environment state and performs no route planning.
    """

    def __init__(
        self,
        runtime: PPORuntime,
        representation_provider: RepresentationProvider,
    ) -> None:
        if runtime is None:
            raise ValueError(
                "runtime is required"
            )

        if representation_provider is None:
            raise ValueError(
                "representation_provider is required"
            )

        provider_spec = (
            validate_representation_provider(
                representation_provider
            )
        )

        # PPOActorCritic's actual public contract is
        # `representation_spec`, not `spec`.
        model_spec = (
            runtime.model.representation_spec
        )

        if model_spec != provider_spec:
            raise ValueError(
                "PPO runtime and representation provider "
                "specs must match"
            )

        # PPORuntime already validates this relationship during construction.
        # We intentionally check it here too because the controller itself
        # owns the representation/runtime integration contract.
        runtime_provider_spec = (
            validate_representation_provider(
                runtime.representation_provider
            )
        )

        if runtime_provider_spec != provider_spec:
            raise ValueError(
                "controller representation provider does not match "
                "the runtime representation provider"
            )

        self.runtime = runtime

        self.representation_provider = (
            representation_provider
        )

    @property
    def representation_spec(self):
        """Return the representation contract consumed by PPO."""

        return self.representation_provider.spec

    @property
    def action_count(self) -> int:
        """Return the number of discrete scope actions."""

        return len(ACTION_TO_INDEX)

    def decide(
        self,
        state: Any,
        *,
        action_mask: ActionMask
        | np.ndarray
        | None = None,
    ) -> PPOControllerDecision:
        """
        Encode causal state and obtain one PPO action proposal.

        No safety override is applied here.
        """

        representation = (
            self.representation_provider.encode(
                state
            )
        )

        decision = self.runtime.decide(
            representation,
            action_mask=action_mask,
        )

        return self._to_controller_decision(
            decision
        )

    def decide_from_representation(
        self,
        representation: Any,
        *,
        action_mask: ActionMask
        | np.ndarray
        | None = None,
    ) -> PPOControllerDecision:
        """
        Decide directly from an already-created representation.

        This avoids encoding the same causal state twice when an upstream
        orchestrator already owns the representation.
        """

        decision = self.runtime.decide(
            representation,
            action_mask=action_mask,
        )

        return self._to_controller_decision(
            decision
        )

    @staticmethod
    def _to_controller_decision(
        decision: PPOPolicyDecision,
    ) -> PPOControllerDecision:
        action_index = int(
            decision.action
        )

        if action_index not in INDEX_TO_ACTION:
            raise RuntimeError(
                "PPORuntime returned an unknown action index: "
                f"{action_index}"
            )

        return PPOControllerDecision(
            requested_action=(
                INDEX_TO_ACTION[action_index]
            ),
            action_index=action_index,
            log_probability=float(
                decision.log_probability
            ),
            value=float(
                decision.value
            ),
            policy_decision=decision,
        )