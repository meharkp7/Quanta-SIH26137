from __future__ import annotations

import numpy as np
import pytest
import torch

from src.learning.action_mask import ActionMask
from src.learning.ppo_actor_critic import PPOActorCritic
from src.learning.ppo_runtime import PPORuntime
from src.learning.representation import (
    RepresentationOutput,
    RepresentationProvider,
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


def make_runtime(
    *,
    deterministic: bool = False,
) -> PPORuntime:
    provider = DummyProvider()

    model = PPOActorCritic(
        provider.spec
    )

    return PPORuntime(
        model,
        provider,
        deterministic=deterministic,
    )


def test_runtime_produces_valid_decision() -> None:
    runtime = make_runtime()

    decision = runtime.decide(
        state=object(),
        action_mask=np.ones(
            5,
            dtype=bool,
        ),
    )

    assert 0 <= decision.action < 5
    assert np.isfinite(
        decision.log_probability
    )
    assert np.isfinite(
        decision.value
    )

    assert decision.representation.flat.shape == (
        332,
    )

    assert decision.action_mask.shape == (5,)


def test_runtime_respects_action_mask() -> None:
    runtime = make_runtime()

    mask = np.asarray(
        [True, False, False, True, False],
        dtype=bool,
    )

    for _ in range(100):
        decision = runtime.decide(
            state=object(),
            action_mask=mask,
        )

        assert decision.action in {0, 3}


def test_deterministic_runtime_selects_highest_logit() -> None:
    runtime = make_runtime(
        deterministic=True
    )

    # Make the expected argmax explicit instead of relying on random
    # model initialization.
    with torch.no_grad():
        final_linear = runtime.model.actor

        final_linear.weight.zero_()
        final_linear.bias.copy_(
            torch.tensor(
                [5.0, 1.0, 2.0, 3.0, 4.0],
                dtype=final_linear.bias.dtype,
            )
        )

    decision = runtime.decide(
        state=object(),
        action_mask=np.ones(
            5,
            dtype=bool,
        ),
    )

    assert decision.action == 0


def test_empty_action_mask_is_rejected() -> None:
    runtime = make_runtime()

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        runtime.decide(
            state=object(),
            action_mask=np.zeros(
                5,
                dtype=bool,
            ),
        )


def test_bad_action_mask_shape_is_rejected() -> None:
    runtime = make_runtime()

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        runtime.decide(
            state=object(),
            action_mask=np.ones(
                4,
                dtype=bool,
            ),
        )


def test_provider_and_model_specs_must_match() -> None:
    class OtherProvider(DummyProvider):
        @property
        def spec(self) -> RepresentationSpec:
            return RepresentationSpec(
                spatial_dim=64
            )

    provider = OtherProvider()

    model = PPOActorCritic(
        RepresentationSpec()
    )

    with pytest.raises(
        ValueError,
        match="spec",
    ):
        PPORuntime(
            model,
            provider,
        )


def test_runtime_is_inference_only() -> None:
    runtime = make_runtime()

    assert runtime.model.training is False

    for parameter in runtime.model.parameters():
        assert parameter.requires_grad is True