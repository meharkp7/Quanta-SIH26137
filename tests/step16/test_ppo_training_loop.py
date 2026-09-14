"""
Step 16.19 — PPO training-loop integration tests.

Tests:

1. Empty rollouts are rejected.
2. Genuine terminal states force zero bootstrap.
3. Explicit bootstrap values are accepted for non-terminal states.
4. Successful updates consume on-policy data.
5. Failed updates preserve on-policy data.
6. Old rollouts cannot be reused.
7. Fresh rollouts can be used after an update.
8. Invalid terminated/truncated combinations are rejected.

The actual critic-bootstrap path is tested separately using a real
PPOActorCritic in the final integration test.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from src.learning.ppo_rollout import (
    PPOTransition,
    PPORolloutBuffer,
)
from src.learning.ppo_training_loop import (
    PPOTrainingLoop,
)
from src.learning.ppo_trainer import (
    PPOTrainingMetrics,
)
from src.learning.ppo_runtime import PPORuntime
from src.learning.ppo_controller import PPOController
from src.learning.ppo_rollout_collector import PPORolloutCollector

# ============================================================================
# Helpers
# ============================================================================


def make_transition(
    *,
    terminated: bool,
    truncated: bool,
    reward: float = 1.0,
) -> PPOTransition:
    """Create one valid PPO transition."""

    return PPOTransition(
        observation=np.zeros(
            332,
            dtype=np.float32,
        ),
        action=0,
        log_probability=-0.5,
        value=0.25,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        action_mask=np.ones(
            5,
            dtype=bool,
        ),
    )


def make_metrics() -> PPOTrainingMetrics:
    """Create deterministic dummy training metrics."""

    return PPOTrainingMetrics(
        updates=1,
        total_loss=0.0,
        policy_loss=0.0,
        value_loss=0.0,
        entropy=0.0,
        approximate_kl=0.0,
        clip_fraction=0.0,
        ratio_mean=1.0,
        mean_reward=1.0,
        mean_advantage=0.0,
        mean_return=1.0,
        gradient_norm=0.0,
    )


def make_loop():
    """
    Build an isolated training-loop test stack.

    The mocked collector exposes the same structural dependencies expected
    by PPOTrainingLoop. Critic evaluation is not exercised in these tests;
    explicit bootstrap values are supplied for non-terminal rollouts.
    """

    buffer = PPORolloutBuffer(
        capacity=32,
    )

    model = object()

    controller = Mock()
    controller.runtime.model = model

    collector = Mock()
    collector.rollout_buffer = buffer
    collector.controller = controller

    trainer = Mock()
    trainer.model = model
    trainer.update.return_value = make_metrics()

    loop = PPOTrainingLoop(
        collector=collector,
        trainer=trainer,
        rollout_buffer=buffer,
    )

    # Mock.clear() does not actually clear the real rollout buffer.
    # Give it the real clear operation while retaining the mock.
    collector.clear.side_effect = buffer.clear

    return (
        loop,
        collector,
        trainer,
        buffer,
    )


# ============================================================================
# Empty rollout
# ============================================================================


def test_update_rejects_empty_rollout():
    loop, _, trainer, buffer = make_loop()

    assert buffer.empty

    with pytest.raises(
        RuntimeError,
        match="empty rollout",
    ):
        loop.update()

    trainer.update.assert_not_called()


# ============================================================================
# Genuine terminal rollout
# ============================================================================


def test_terminated_rollout_forces_zero_bootstrap():
    loop, _, trainer, buffer = make_loop()

    buffer.append(
        make_transition(
            terminated=True,
            truncated=False,
        )
    )

    result = loop.update()

    assert result.transitions_collected == 1

    trainer.update.assert_called_once()

    _, kwargs = trainer.update.call_args

    assert kwargs["bootstrap_value"] == 0.0
    assert buffer.empty


def test_terminated_rollout_rejects_nonzero_bootstrap():
    loop, _, trainer, buffer = make_loop()

    buffer.append(
        make_transition(
            terminated=True,
            truncated=False,
        )
    )

    with pytest.raises(
        ValueError,
        match="terminated rollout must use bootstrap_value=0",
    ):
        loop.update(
            bootstrap_value=1.25,
        )

    trainer.update.assert_not_called()

    # Failed update preserves on-policy data.
    assert buffer.size == 1


# ============================================================================
# Explicit bootstrap for truncated/non-terminal rollout
# ============================================================================


def test_truncated_rollout_accepts_explicit_bootstrap_value():
    loop, _, trainer, buffer = make_loop()

    buffer.append(
        make_transition(
            terminated=False,
            truncated=True,
        )
    )

    result = loop.update(
        bootstrap_value=1.25,
    )

    assert result.transitions_collected == 1

    trainer.update.assert_called_once()

    _, kwargs = trainer.update.call_args

    assert kwargs["bootstrap_value"] == pytest.approx(
        1.25
    )

    assert buffer.empty


def test_non_terminal_rollout_accepts_explicit_bootstrap_value():
    loop, _, trainer, buffer = make_loop()

    buffer.append(
        make_transition(
            terminated=False,
            truncated=False,
        )
    )

    result = loop.update(
        bootstrap_value=0.75,
    )

    assert result.transitions_collected == 1

    trainer.update.assert_called_once()

    _, kwargs = trainer.update.call_args

    assert kwargs["bootstrap_value"] == pytest.approx(
        0.75
    )

    assert buffer.empty


# ============================================================================
# Invalid terminal-state combination
# ============================================================================


def test_terminated_and_truncated_cannot_both_be_true():
    loop, _, trainer, buffer = make_loop()

    buffer.append(
        make_transition(
            terminated=True,
            truncated=True,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="both.*terminated.*truncated",
    ):
        loop.update(
            bootstrap_value=0.0,
        )

    trainer.update.assert_not_called()

    assert buffer.size == 1


# ============================================================================
# On-policy rollout consumption
# ============================================================================


def test_successful_update_consumes_rollout():
    loop, _, _, buffer = make_loop()

    buffer.append(
        make_transition(
            terminated=True,
            truncated=False,
        )
    )

    result = loop.update()

    assert result.transitions_collected == 1
    assert buffer.empty


def test_consumed_rollout_cannot_be_reused():
    loop, _, trainer, buffer = make_loop()

    buffer.append(
        make_transition(
            terminated=True,
            truncated=False,
        )
    )

    first_result = loop.update()

    assert first_result.transitions_collected == 1
    assert buffer.empty

    trainer.update.reset_mock()

    with pytest.raises(
        RuntimeError,
        match="empty rollout",
    ):
        loop.update()

    trainer.update.assert_not_called()


def test_new_rollout_can_be_used_after_previous_update():
    loop, _, trainer, buffer = make_loop()

    # First on-policy rollout.
    buffer.append(
        make_transition(
            terminated=True,
            truncated=False,
            reward=1.0,
        )
    )

    first_result = loop.update()

    assert first_result.transitions_collected == 1
    assert buffer.empty

    trainer.update.reset_mock()

    # Fresh on-policy rollout.
    buffer.append(
        make_transition(
            terminated=True,
            truncated=False,
            reward=2.0,
        )
    )

    second_result = loop.update()

    assert second_result.transitions_collected == 1
    assert buffer.empty

    trainer.update.assert_called_once()


# ============================================================================
# Failed update preserves rollout
# ============================================================================


def test_failed_update_does_not_clear_rollout():
    loop, _, trainer, buffer = make_loop()

    buffer.append(
        make_transition(
            terminated=True,
            truncated=False,
        )
    )

    trainer.update.side_effect = RuntimeError(
        "synthetic PPO update failure"
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic PPO update failure",
    ):
        loop.update()

    # Training failed, so the on-policy data must remain available.
    assert buffer.size == 1
    assert not buffer.empty


def test_invalid_bootstrap_does_not_clear_rollout():
    loop, _, trainer, buffer = make_loop()

    buffer.append(
        make_transition(
            terminated=True,
            truncated=False,
        )
    )

    with pytest.raises(
        ValueError,
        match="terminated rollout must use bootstrap_value=0",
    ):
        loop.update(
            bootstrap_value=5.0,
        )

    trainer.update.assert_not_called()

    assert buffer.size == 1
    assert not buffer.empty

def test_update_uses_actual_critic_value_for_nonterminal_rollout():
    import torch

    from src.learning.ppo_actor_critic import PPOActorCritic, PPOActorCriticConfig
    from src.learning.representation import RepresentationOutput, RepresentationSpec

    spec = RepresentationSpec()

    model = PPOActorCritic(
        representation_spec=spec,
        config=PPOActorCriticConfig(
            hidden_dim=32,
            hidden_dim_2=16,
            action_dim=5,
        ),
    )

    # Make the critic deterministic: V(s) = 2.5
    with torch.no_grad():
        model.critic.weight.zero_()
        model.critic.bias.fill_(2.5)

    class StubProvider:
        def __init__(self, representation_spec):
            self.spec = representation_spec

        def encode(self, state):
            return RepresentationOutput(
                spec=self.spec,
                spatial=np.zeros(self.spec.spatial_dim, dtype=np.float32),
                temporal=np.zeros(self.spec.temporal_dim, dtype=np.float32),
                forecast=np.zeros(self.spec.forecast_dim, dtype=np.float32),
                uncertainty=np.zeros(self.spec.uncertainty_dim, dtype=np.float32),
                context=np.zeros(self.spec.context_dim, dtype=np.float32),
                observation_time_s=0.0,
                graph_version="test-graph-v1",
                representation_version=self.spec.version,
                metadata={"state": state},
            )

    class StubEnv:
        def _action_mask(self):
            return np.ones(5, dtype=bool)

        def step(self, action):
            return (
                np.zeros(32, dtype=np.float32),
                1.0,
                False,
                False,
                {},
            )

    provider = StubProvider(spec)

    runtime = PPORuntime(
        model=model,
        representation_provider=provider,
        deterministic=True,
    )

    controller = PPOController(
        runtime=runtime,
        representation_provider=provider,
    )

    buffer = PPORolloutBuffer(capacity=8)

    env = StubEnv()

    collector = PPORolloutCollector(
        env=env,
        controller=controller,
        representation_provider=provider,
        rollout_buffer=buffer,
    )

    trainer = Mock()
    trainer.model = model
    trainer.update.return_value = make_metrics()

    loop = PPOTrainingLoop(
        collector=collector,
        trainer=trainer,
        rollout_buffer=buffer,
    )

    collector.collect_step(
        policy_state="initial_state",
        next_policy_state="final_state",
    )

    assert collector.last_policy_state == "final_state"

    loop.update()

    trainer.update.assert_called_once()

    _, kwargs = trainer.update.call_args

    assert kwargs["bootstrap_value"] == pytest.approx(2.5)
    assert buffer.empty