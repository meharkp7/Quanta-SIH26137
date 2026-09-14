from __future__ import annotations

import numpy as np
import torch

from src.learning.ppo_actor_critic import PPOActorCritic
from src.learning.ppo_controller import PPOController
from src.learning.ppo_env import TrafficRoutingPPOEnv
from src.learning.ppo_rollout import PPORolloutBuffer
from src.learning.ppo_rollout_collector import PPORolloutCollector
from src.learning.ppo_runtime import PPORuntime
from src.learning.representation import (
    RepresentationOutput,
    RepresentationSpec,
)

from test_ppo_env import (
    DeterministicSimulator,
    make_plan,
)


class DummyProvider:
    spec = RepresentationSpec()

    def encode(self, state):
        del state

        return RepresentationOutput(
            representation_version=self.spec.version,
            spec=self.spec,
            spatial=np.zeros(
                self.spec.spatial_dim,
                dtype=np.float32,
            ),
            temporal=np.zeros(
                self.spec.temporal_dim,
                dtype=np.float32,
            ),
            forecast=np.zeros(
                self.spec.forecast_dim,
                dtype=np.float32,
            ),
            uncertainty=np.zeros(
                self.spec.uncertainty_dim,
                dtype=np.float32,
            ),
            context=np.zeros(
                self.spec.context_dim,
                dtype=np.float32,
            ),
            observation_time_s=0.0,
            graph_version="graph-v1",
        )


def make_test_objects(scenario):
    provider = DummyProvider()

    model = PPOActorCritic(provider.spec)

    runtime = PPORuntime(
        model,
        provider,
    )

    controller = PPOController(
        runtime,
        provider,
    )

    simulator = DeterministicSimulator()

    env = TrafficRoutingPPOEnv(
        scenario,
        make_plan(),
        simulator=simulator,
    )

    env.reset()

    buffer = PPORolloutBuffer(
        capacity=8,
    )

    collector = PPORolloutCollector(
        env=env,
        controller=controller,
        representation_provider=provider,
        rollout_buffer=buffer,
    )

    return (
        collector,
        controller,
        env,
        buffer,
    )


def test_collect_step_stores_one_transition(scenario):
    collector, controller, env, buffer = make_test_objects(
        scenario
    )

    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()
        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [5.0, 1.0, 2.0, 3.0, 4.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    result = collector.collect_step(
        policy_state=object(),
    )

    transitions = buffer.transitions()

    assert len(transitions) == 1
    assert 0 <= result.transition.action < 5
    assert result.transition.reward == result.reward
    assert result.transition.terminated is False
    assert result.transition.truncated is False

    assert result.transition.observation.shape == (
        RepresentationSpec().total_dim,
    )

    assert result.transition.action_mask.shape == (5,)
    assert result.transition.action_mask.dtype == bool


def test_collect_step_uses_real_environment_reward(scenario):
    collector, controller, env, buffer = make_test_objects(
        scenario
    )

    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()
        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [5.0, 1.0, 2.0, 3.0, 4.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    result = collector.collect_step(
        policy_state=object(),
    )

    transitions = buffer.transitions()

    assert result.reward < 0.0
    assert 0 <= result.info["ppo_collection_action"] < 5
    assert len(env.decision_log) == 1
    assert len(transitions) == 1


def test_collect_step_respects_action_mask(scenario):
    collector, controller, env, buffer = make_test_objects(
        scenario
    )

    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()
        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [5.0, 4.0, 3.0, 2.0, 1.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    mask = np.array(
        [False, False, False, True, False],
        dtype=bool,
    )

    result = collector.collect_step(
        policy_state=object(),
        action_mask=mask,
    )

    transitions = buffer.transitions()

    assert result.transition.action == 3
    assert transitions[0].action == 3
    assert transitions[0].action_mask.tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]


def test_collect_step_does_not_run_policy_twice(scenario):
    collector, controller, env, buffer = make_test_objects(
        scenario
    )

    calls = {"count": 0}

    original_decide = (
        controller.decide_from_representation
    )

    def counted_decide(*args, **kwargs):
        calls["count"] += 1
        return original_decide(*args, **kwargs)

    controller.decide_from_representation = counted_decide

    collector.collect_step(
        policy_state=object(),
    )

    transitions = buffer.transitions()

    assert calls["count"] == 1
    assert len(transitions) == 1
    assert len(env.decision_log) == 1

def test_buffer_size_matches_collected_transitions(scenario):
    collector, controller, env, buffer = make_test_objects(
        scenario
    )

    assert collector.buffer_size() == 0

    collector.collect_step(
        policy_state=object(),
    )

    assert collector.buffer_size() == 1
    assert len(buffer.transitions()) == 1