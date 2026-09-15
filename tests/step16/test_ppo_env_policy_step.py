from __future__ import annotations

import numpy as np
import torch

from src.contracts.decision import ScopeAction
from src.learning.ppo_actor_critic import PPOActorCritic
from src.learning.ppo_controller import PPOController
from src.learning.ppo_env import (
    TrafficRoutingPPOEnv,
    SimulatorBackend,
)
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
    """
    Minimal representation provider for environment/controller integration.

    The environment does not know anything about the internal representation
    construction.
    """

    def __init__(self) -> None:
        self._spec = RepresentationSpec()

    @property
    def spec(self) -> RepresentationSpec:
        return self._spec

    def encode(
        self,
        state: object,
    ) -> RepresentationOutput:
        spec = self._spec

        return RepresentationOutput(
            spec=spec,
            spatial=np.zeros(
                spec.spatial_dim,
                dtype=np.float32,
            ),
            temporal=np.zeros(
                spec.temporal_dim,
                dtype=np.float32,
            ),
            forecast=np.zeros(
                spec.forecast_dim,
                dtype=np.float32,
            ),
            uncertainty=np.zeros(
                spec.uncertainty_dim,
                dtype=np.float32,
            ),
            context=np.zeros(
                spec.context_dim,
                dtype=np.float32,
            ),
            observation_time_s=60.0,
            graph_version="graph-v1",
            representation_version=spec.version,
        )


def make_controller(
    *,
    deterministic: bool = True,
) -> PPOController:
    provider = DummyProvider()

    model = PPOActorCritic(
        provider.spec
    )

    runtime = PPORuntime(
        model,
        provider,
        deterministic=deterministic,
    )

    return PPOController(
        runtime,
        provider,
    )


def test_policy_step_uses_ppo_requested_action(
    scenario,
) -> None:
    simulator = DeterministicSimulator()

    controller = make_controller()

    # Explicitly make KEEP the highest-logit action.
    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()

        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [5.0, 1.0, 2.0, 3.0, 4.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    env = TrafficRoutingPPOEnv(
        scenario,
        make_plan(),
        simulator=simulator,
    )

    env.reset()

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.policy_step(
        policy_state=object(),
        controller=controller,
        action_mask=np.ones(
            5,
            dtype=bool,
        ),
    )

    assert observation.shape == (
        env.config.observation_size,
    )

    assert isinstance(
        reward,
        float,
    )

    assert info["ppo_action_index"] == 0

    assert (
        info["ppo_requested_action"]
        == ScopeAction.KEEP.value
    )

    assert (
        info["requested_action"]
        == ScopeAction.KEEP.value
    )

    assert (
        info["executed_action"]
        == ScopeAction.KEEP.value
    )

    assert info["qpso_called"] is False

    assert simulator.sim_time_s == 60.0

    assert np.isfinite(
        info["ppo_log_probability"]
    )

    assert np.isfinite(
        info["ppo_value"]
    )


def test_policy_step_preserves_controller_mask(
    scenario,
) -> None:
    simulator = DeterministicSimulator()

    controller = make_controller()

    # Unrestricted highest logit = VEHICLE (index 2).
    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()

        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [1.0, 2.0, 5.0, 4.0, 3.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    env = TrafficRoutingPPOEnv(
        scenario,
        make_plan(),
        simulator=simulator,
    )

    env.reset()

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.policy_step(
        policy_state=object(),
        controller=controller,
        action_mask=np.asarray(
            [True, False, False, True, False],
            dtype=bool,
        ),
    )

    # VEHICLE is masked, so REGIONAL becomes the highest feasible action.
    assert info["ppo_action_index"] == 3

    assert (
        info["ppo_requested_action"]
        == ScopeAction.REGIONAL.value
    )

    assert simulator.sim_time_s == 60.0


def test_policy_step_uses_same_execution_path_as_step(
    scenario,
) -> None:
    simulator = DeterministicSimulator()

    controller = make_controller()

    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()

        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [5.0, 1.0, 2.0, 3.0, 4.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    env = TrafficRoutingPPOEnv(
        scenario,
        make_plan(),
        simulator=simulator,
    )

    env.reset()

    (
        _,
        _,
        _,
        _,
        info,
    ) = env.policy_step(
        policy_state=object(),
        controller=controller,
    )

    # policy_step() must eventually use the ordinary env.step() execution
    # path, so the normal execution metadata must still be present.
    assert "requested_action" in info
    assert "executed_action" in info
    assert "reward_components" in info
    assert "decision_elapsed_s" in info
    assert "qpso_called" in info

    assert len(env.decision_log) == 1


def test_policy_step_keeps_requested_and_executed_actions_distinct(
    scenario,
) -> None:
    simulator = DeterministicSimulator()

    controller = make_controller()

    # PPO requests LOCAL.
    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()

        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [1.0, 5.0, 2.0, 3.0, 4.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    env = TrafficRoutingPPOEnv(
        scenario,
        make_plan(),
        simulator=simulator,
    )

    env.reset()

    (
        _,
        _,
        _,
        _,
        info,
    ) = env.policy_step(
        policy_state=object(),
        controller=controller,
        action_mask=np.ones(
            5,
            dtype=bool,
        ),
    )

    assert (
        info["ppo_requested_action"]
        == ScopeAction.LOCAL.value
    )

    # These fields describe the actual environment execution after the
    # normal safety/scope/QPSO path.
    assert "executed_action" in info

    # The controller's original proposal is preserved separately.
    assert "ppo_action_index" in info
    assert "ppo_log_probability" in info
    assert "ppo_value" in info