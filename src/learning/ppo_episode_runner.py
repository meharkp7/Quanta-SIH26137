"""
Step 17.2C — PPO episode runner.

Owns the orchestration boundary between:

    TrafficRoutingPPOEnv
        ↓
    causal Observation
        ↓
    PPOPolicyStateRuntime
        ↓
    PPOController / RepresentationProvider
        ↓
    PPORolloutCollector
        ↓
    fresh PPO transitions

The runner never converts the legacy 32-D Gym observation into PPO input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.learning.ppo_env import TrafficRoutingPPOEnv
from src.learning.ppo_policy_state_runtime import (
    PPOPolicyStateRuntime,
)
from src.learning.ppo_rollout_collector import (
    PPORolloutCollector,
    PPORolloutStep,
)
from src.learning.state.temporal_memory import (
    decision_context_from_scope,
)


@dataclass(frozen=True)
class PPOEpisodeResult:
    """Summary of one fresh on-policy episode."""

    episode_id: str
    transitions_collected: int
    total_reward: float
    terminated: bool
    truncated: bool
    final_policy_state: Any
    steps: tuple[PPORolloutStep, ...]


class PPOEpisodeRunner:
    """
    Collect one fresh on-policy PPO episode.

    Important:
        - environment owns execution;
        - PPOController owns action selection;
        - PPOPolicyStateRuntime owns structured state;
        - PPORolloutCollector owns transition storage;
        - PPOTrainingLoop owns GAE/update.

    Decision-memory rule:
        - ScopeDecision is the causal routing decision;
        - DecisionRecord is the post-execution audit record;
        - only ScopeDecision enters DecisionMemory.
    """

    def __init__(
        self,
        *,
        env: TrafficRoutingPPOEnv,
        state_runtime: PPOPolicyStateRuntime,
        collector: PPORolloutCollector,
    ) -> None:
        self.env = env
        self.state_runtime = state_runtime
        self.collector = collector

        if collector.env is not env:
            raise ValueError(
                "collector must use the same environment as runner"
            )

    # ------------------------------------------------------------------
    # Forecast / uncertainty
    # ------------------------------------------------------------------

    def _forecast_features(self) -> np.ndarray:
        """
        Obtain the current causal forecast.

        If no neural forecaster is configured, use the explicitly supported
        persistence-pilot representation: a zero feature vector.

        The final neural forecaster can be injected later without changing
        the rollout orchestration.
        """

        provider = self.env.forecast_provider

        if provider is None:
            return np.zeros(
                12,
                dtype=np.float32,
            )

        values = np.asarray(
            tuple(
                float(value)
                for value in provider(
                    float(self.env.simulator.sim_time_s)
                )
            ),
            dtype=np.float32,
        )

        if values.shape != (12,):
            raise ValueError(
                "forecast_provider must return exactly 12 features "
                f"for PPO, got shape={values.shape}"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "forecast_provider returned non-finite values"
            )

        return values

    @staticmethod
    def _uncertainty_features() -> np.ndarray:
        """
        Development uncertainty representation.

        The current environment exposes no uncertainty provider, so this
        remains explicitly zero-valued rather than fabricating uncertainty.
        """

        return np.zeros(
            6,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Episode
    # ------------------------------------------------------------------

    def run_episode(
        self,
        *,
        episode_id: str,
        max_steps: int,
    ) -> PPOEpisodeResult:
        """
        Collect one fresh on-policy episode.

        The rollout buffer must be empty before starting. Reusing old PPO
        trajectories would violate the on-policy training contract.
        """

        if not episode_id:
            raise ValueError(
                "episode_id must be non-empty"
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be positive"
            )

        if not self.collector.rollout_buffer.empty:
            raise RuntimeError(
                "rollout buffer must be empty before starting "
                "a fresh PPO episode"
            )

        # --------------------------------------------------------------
        # Environment reset
        # --------------------------------------------------------------

        _, _ = self.env.reset()

        # --------------------------------------------------------------
        # Structured causal state at t=0
        # --------------------------------------------------------------

        structured = self.state_runtime.reset(
            episode_id=episode_id,
            observation_time_s=float(
                self.env.simulator.sim_time_s
            ),
            state_version="state-v0",
            latest_state=getattr(
                self.env.simulator,
                "latest_state",
                None,
            ),
        )

        current_policy_state = structured.policy_state

        collected: list[PPORolloutStep] = []
        total_reward = 0.0

        terminated = False
        truncated = False

        # State versions are monotonic integers at the TemporalMemory
        # boundary, represented externally as state-vN.
        state_version = 0

        # --------------------------------------------------------------
        # Fresh on-policy rollout
        # --------------------------------------------------------------

        for _ in range(max_steps):
            action_mask = self.env._action_mask()

            step_result = self.collector.collect_step(
                policy_state=current_policy_state,
                action_mask=action_mask,
            )

            collected.append(step_result)

            total_reward += float(
                step_result.reward
            )

            terminated = bool(
                step_result.terminated
            )

            truncated = bool(
                step_result.truncated
            )

            # ----------------------------------------------------------
            # Environment has now advanced to t+60.
            #
            # IMPORTANT:
            # Do NOT read env.decision_log[-1] here.
            #
            # decision_log contains DecisionRecord, which is the
            # post-execution audit/reward record. DecisionMemory requires
            # the causal ScopeDecision produced by the environment during
            # action resolution.
            # ----------------------------------------------------------

            state_version += 1

            scope_decision = getattr(
                self.env,
                "last_scope_decision",
                None,
            )

            decision_context = None

            if scope_decision is not None:
                # DecisionMemory receives the authoritative causal
                # routing decision, not the reward-bearing audit record.
                self.state_runtime.record_decision(
                    scope_decision
                )

                decision_context = (
                    decision_context_from_scope(
                        scope_decision,
                        total_vehicle_count=len(
                            self.env.scenario.fleet
                        ),
                        total_request_count=len(
                            self.env.scenario.requests
                        ),
                    )
                )

            # ----------------------------------------------------------
            # Build next causal policy state
            # ----------------------------------------------------------

            next_structured = (
                self.state_runtime.advance(
                    observation_time_s=float(
                        self.env.simulator.sim_time_s
                    ),
                    state_version=(
                        f"state-v{state_version}"
                    ),
                    latest_state=getattr(
                        self.env.simulator,
                        "latest_state",
                        None,
                    ),
                    forecast_features=(
                        self._forecast_features()
                    ),
                    uncertainty_features=(
                        self._uncertainty_features()
                    ),
                    decision_context=decision_context,
                )
            )

            current_policy_state = (
                next_structured.policy_state
            )

            # ----------------------------------------------------------
            # Episode termination
            # ----------------------------------------------------------

            if terminated or truncated:
                break

        return PPOEpisodeResult(
            episode_id=episode_id,
            transitions_collected=len(
                collected
            ),
            total_reward=float(
                total_reward
            ),
            terminated=terminated,
            truncated=truncated,
            final_policy_state=current_policy_state,
            steps=tuple(collected),
        )