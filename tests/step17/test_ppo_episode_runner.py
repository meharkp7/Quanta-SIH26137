from __future__ import annotations

import numpy as np

from src.learning.ppo_actor_critic import (
    PPOActorCritic,
    PPOActorCriticConfig,
)
from src.learning.ppo_controller import PPOController
from src.learning.ppo_env import TrafficRoutingPPOEnv
from src.learning.ppo_policy_state import (
    StructuredPPORepresentationProvider,
)
from src.learning.ppo_policy_state_runtime import (
    PPOPolicyStateRuntime,
)
from src.learning.ppo_rollout import (
    PPORolloutBuffer,
)
from src.learning.ppo_rollout_collector import (
    PPORolloutCollector,
)
from src.learning.ppo_runtime import PPORuntime
from src.learning.ppo_episode_runner import (
    PPOEpisodeRunner,
)

from tests.step16.test_ppo_env import (
    DeterministicSimulator,
    make_plan,
)


def make_runner(scenario):
    env = TrafficRoutingPPOEnv(
        scenario,
        make_plan(),
        simulator=DeterministicSimulator(),
    )

    state_runtime = PPOPolicyStateRuntime(
        scenario=scenario,
    )

    provider = StructuredPPORepresentationProvider()

    model = PPOActorCritic(
        provider.spec,
        PPOActorCriticConfig(
            action_dim=5,
        ),
    )

    runtime = PPORuntime(
        model=model,
        representation_provider=provider,
        device="cpu",
        deterministic=True,
    )

    controller = PPOController(
        runtime=runtime,
        representation_provider=provider,
    )

    buffer = PPORolloutBuffer(
        capacity=16,
    )

    collector = PPORolloutCollector(
        env=env,
        controller=controller,
        representation_provider=provider,
        rollout_buffer=buffer,
    )

    runner = PPOEpisodeRunner(
        env=env,
        state_runtime=state_runtime,
        collector=collector,
    )

    return runner, env, buffer


def test_episode_runner_collects_fresh_structured_rollout(
    scenario,
):
    runner, env, buffer = make_runner(
        scenario
    )

    result = runner.run_episode(
        episode_id="ppo-episode-1",
        max_steps=2,
    )

    assert result.transitions_collected == 2
    assert buffer.size == 2

    assert result.final_policy_state is not None

    assert (
        result.final_policy_state.fused_state
        .observation_time_s
        == 120.0
    )

    transitions = buffer.transitions()

    assert all(
        transition.observation.shape == (332,)
        for transition in transitions
    )

    assert all(
        transition.action_mask.shape == (5,)
        for transition in transitions
    )


def test_episode_runner_advances_temporal_state(
    scenario,
):
    runner, env, buffer = make_runner(
        scenario
    )

    result = runner.run_episode(
        episode_id="ppo-episode-2",
        max_steps=2,
    )

    temporal = (
        result.final_policy_state
        .fused_state
        .temporal_batch
    )

    assert temporal.latest is not None

    assert (
        temporal.latest.observation_time_s
        == 120.0
    )

    assert temporal.valid_mask.sum() == 3.0


def test_episode_runner_does_not_use_legacy_observation(
    scenario,
):
    runner, env, buffer = make_runner(
        scenario
    )

    result = runner.run_episode(
        episode_id="ppo-episode-3",
        max_steps=1,
    )

    transition = buffer.transitions()[0]

    assert transition.observation.shape == (332,)

    # The environment still exposes its backward-compatible Gym vector.
    legacy, _ = env.reset()

    assert legacy.shape == (
        env.config.observation_size,
    )

    assert transition.observation.shape != legacy.shape


def test_episode_runner_collects_real_environment_rewards(
    scenario,
):
    runner, env, buffer = make_runner(
        scenario
    )

    result = runner.run_episode(
        episode_id="ppo-episode-4",
        max_steps=2,
    )

    expected = sum(
        step.reward
        for step in result.steps
    )

    assert np.isclose(
        result.total_reward,
        expected,
    )
