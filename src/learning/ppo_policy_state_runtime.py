from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.learning.ppo_observation_adapter import PPOObservationAdapter
from src.learning.ppo_policy_state import PPOPolicyState, PPOPolicyStateBuilder
from src.learning.state.temporal_memory import TemporalDecisionContext


@dataclass(frozen=True)
class PPOStructuredState:
    """Causal structured PPO state paired with its source observation."""

    observation: Any
    policy_state: PPOPolicyState


class PPOPolicyStateRuntime:
    """
    Owns the causal Observation -> structured PPO state transition.

    This layer deliberately sits outside TrafficRoutingPPOEnv so the
    environment's legacy Gym observation contract remains backward compatible.
    """

    def __init__(
        self,
        *,
        scenario,
        observation_adapter: PPOObservationAdapter | None = None,
        policy_state_builder: PPOPolicyStateBuilder | None = None,
    ) -> None:
        self.scenario = scenario

        self.observation_adapter = (
            observation_adapter
            or PPOObservationAdapter(scenario)
        )

        self.policy_state_builder = (
            policy_state_builder
            or PPOPolicyStateBuilder(scenario)
        )

        self._current: PPOStructuredState | None = None

    @property
    def current(self) -> PPOStructuredState | None:
        return self._current

    @property
    def current_policy_state(self) -> PPOPolicyState | None:
        return (
            None
            if self._current is None
            else self._current.policy_state
        )

    def reset(
        self,
        *,
        episode_id: str,
        observation_time_s: float = 0.0,
        state_version: str = "state-v0",
        latest_state: dict[str, Any] | None = None,
    ) -> PPOStructuredState:
        observation = self.observation_adapter.build(
            episode_id=episode_id,
            observation_time_s=observation_time_s,
            state_version=state_version,
            latest_state=latest_state,
        )

        self.policy_state_builder.reset(observation)

        policy_state = self.policy_state_builder.build(
            observation,
            forecast_features=np.zeros(12, dtype=np.float32),
            uncertainty_features=np.zeros(6, dtype=np.float32),
            forecast_version="persistence-v0",
            uncertainty_version="none-v0",
        )

        self._current = PPOStructuredState(
            observation=observation,
            policy_state=policy_state,
        )

        return self._current

    def advance(
        self,
        *,
        observation_time_s: float,
        state_version: str,
        latest_state: dict[str, Any] | None = None,
        forecast_features=None,
        uncertainty_features=None,
        forecast_version: str = "persistence-v0",
        uncertainty_version: str = "none-v0",
        decision_context: TemporalDecisionContext | None = None,
    ) -> PPOStructuredState:
        if self._current is None:
            raise RuntimeError(
                "PPOPolicyStateRuntime.advance() called before reset()"
            )

        observation = self.observation_adapter.build(
            episode_id=self._current.observation.episode_id,
            observation_time_s=observation_time_s,
            state_version=state_version,
            latest_state=latest_state,
        )

        if forecast_features is None:
            forecast_features = np.zeros(12, dtype=np.float32)

        if uncertainty_features is None:
            uncertainty_features = np.zeros(6, dtype=np.float32)

        policy_state = self.policy_state_builder.build(
            observation,
            forecast_features=np.asarray(
                forecast_features,
                dtype=np.float32,
            ),
            uncertainty_features=np.asarray(
                uncertainty_features,
                dtype=np.float32,
            ),
            forecast_version=forecast_version,
            uncertainty_version=uncertainty_version,
            decision_context=decision_context,
        )

        self._current = PPOStructuredState(
            observation=observation,
            policy_state=policy_state,
        )

        return self._current

    def record_decision(self, decision) -> None:
        self.policy_state_builder.record_decision(decision)
