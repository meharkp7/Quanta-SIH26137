from __future__ import annotations

import numpy as np
import torch
import pytest

from src.contracts.decision import ScopeAction
from src.learning.action_mask import ActionMask
from src.learning.ppo_actor_critic import PPOActorCritic
from src.learning.ppo_controller import (
    PPOController,
    PPOControllerDecision,
)
from src.learning.ppo_runtime import PPORuntime
from src.learning.representation import (
    RepresentationOutput,
    RepresentationSpec,
)


class DummyProvider:
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
    deterministic: bool = False,
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


def test_controller_produces_valid_decision() -> None:
    controller = make_controller()

    decision = controller.decide(
        state=object(),
        action_mask=np.ones(
            5,
            dtype=bool,
        ),
    )

    assert isinstance(
        decision,
        PPOControllerDecision,
    )

    assert 0 <= decision.action_index < 5

    assert decision.requested_action in {
        ScopeAction.KEEP,
        ScopeAction.LOCAL,
        ScopeAction.VEHICLE,
        ScopeAction.REGIONAL,
        ScopeAction.GLOBAL,
    }

    assert np.isfinite(
        decision.log_probability
    )

    assert np.isfinite(
        decision.value
    )

    assert (
        decision.action_index
        == {
            ScopeAction.KEEP: 0,
            ScopeAction.LOCAL: 1,
            ScopeAction.VEHICLE: 2,
            ScopeAction.REGIONAL: 3,
            ScopeAction.GLOBAL: 4,
        }[decision.requested_action]
    )


def test_controller_respects_action_mask() -> None:
    controller = make_controller()

    mask = np.asarray(
        [True, False, False, True, False],
        dtype=bool,
    )

    for _ in range(50):
        decision = controller.decide(
            state=object(),
            action_mask=mask,
        )

        assert decision.action_index in {
            0,
            3,
        }

        assert decision.requested_action in {
            ScopeAction.KEEP,
            ScopeAction.REGIONAL,
        }


def test_controller_deterministic_decision() -> None:
    controller = make_controller(
        deterministic=True
    )

    # Explicitly control the actor output.
    # The actor is a single Linear layer in the current implementation.
    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()

        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [1.0, 2.0, 3.0, 5.0, 4.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    decision = controller.decide(
        state=object(),
        action_mask=np.ones(
            5,
            dtype=bool,
        ),
    )

    assert decision.action_index == 3
    assert (
        decision.requested_action
        == ScopeAction.REGIONAL
    )


def test_controller_mask_can_change_selected_action() -> None:
    controller = make_controller(
        deterministic=True
    )

    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()

        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [1.0, 2.0, 5.0, 4.0, 3.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    unrestricted = controller.decide(
        state=object(),
        action_mask=np.ones(
            5,
            dtype=bool,
        ),
    )

    assert unrestricted.action_index == 2
    assert (
        unrestricted.requested_action
        == ScopeAction.VEHICLE
    )

    restricted = controller.decide(
        state=object(),
        action_mask=np.asarray(
            [True, False, False, True, False],
            dtype=bool,
        ),
    )

    assert restricted.action_index == 3
    assert (
        restricted.requested_action
        == ScopeAction.REGIONAL
    )


def test_controller_representation_path_matches_direct_runtime_path() -> None:
    controller = make_controller(
        deterministic=True
    )

    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()

        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [5.0, 1.0, 2.0, 3.0, 4.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    state = object()

    via_state = controller.decide(
        state,
        action_mask=np.ones(
            5,
            dtype=bool,
        ),
    )

    representation = (
        controller.representation_provider.encode(
            state
        )
    )

    via_representation = (
        controller.decide_from_representation(
            representation,
            action_mask=np.ones(
                5,
                dtype=bool,
            ),
        )
    )

    assert (
        via_state.action_index
        == via_representation.action_index
    )

    assert (
        via_state.requested_action
        == via_representation.requested_action
    )


def test_controller_does_not_apply_safety_override() -> None:
    """
    Controller reports what PPO requested.

    Safety execution belongs downstream to the environment/runtime safety
    layer. This prevents PPO from silently learning from an already-overridden
    action as if it had requested that action itself.
    """

    controller = make_controller(
        deterministic=True
    )

    with torch.no_grad():
        controller.runtime.model.actor.weight.zero_()

        controller.runtime.model.actor.bias.copy_(
            torch.tensor(
                [1.0, 5.0, 2.0, 3.0, 4.0],
                dtype=controller.runtime.model.actor.bias.dtype,
            )
        )

    decision = controller.decide(
        state=object(),
        action_mask=np.ones(
            5,
            dtype=bool,
        ),
    )

    assert (
        decision.requested_action
        == ScopeAction.LOCAL
    )

    # The controller has no executed_action field.
    # That distinction is intentionally owned by the environment.
    assert not hasattr(
        decision,
        "executed_action",
    )


def test_controller_requires_matching_runtime_and_provider_specs() -> None:
    provider = DummyProvider()

    model = PPOActorCritic(
        provider.spec
    )

    runtime = PPORuntime(
        model,
        provider,
    )

    class OtherProvider(DummyProvider):
        @property
        def spec(self) -> RepresentationSpec:
            return RepresentationSpec(
                spatial_dim=64
            )

    mismatched_provider = OtherProvider()

    with pytest.raises(
        ValueError,
        match="spec",
    ):
        PPOController(
            runtime,
            mismatched_provider,
        )


def test_controller_exposes_representation_contract() -> None:
    controller = make_controller()

    assert (
        controller.representation_spec
        == controller.representation_provider.spec
    )

    assert controller.action_count == 5