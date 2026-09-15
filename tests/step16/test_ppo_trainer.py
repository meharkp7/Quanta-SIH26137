from __future__ import annotations

import numpy as np
import torch

from src.learning.ppo_actor_critic import (
    PPOActorCritic,
)
from src.learning.ppo_rollout import (
    PPOTransition,
    PPORolloutBuffer,
)
from src.learning.ppo_trainer import (
    PPOTrainer,
    PPOTrainerConfig,
)
from src.learning.representation import RepresentationSpec

def make_rollout(
    *,
    size: int = 16,
) -> PPORolloutBuffer:
    buffer = PPORolloutBuffer(
        capacity=size
    )

    rng = np.random.default_rng(123)

    for index in range(size):
        observation = (
            rng.normal(
                size=332
            )
            .astype(np.float32)
        )

        action = index % 5

        buffer.append(
            PPOTransition(
                observation=observation,
                action=action,
                log_probability=-1.5,
                value=0.1,
                reward=float(
                    1.0 if index % 2 == 0 else -0.5
                ),
                terminated=(
                    index == size - 1
                ),
                truncated=False,
                action_mask=np.ones(
                    5,
                    dtype=bool,
                ),
            )
        )

    return buffer


def make_trainer(
    *,
    seed: int = 42,
) -> PPOTrainer:
    model = PPOActorCritic(
        representation_spec=RepresentationSpec()
    )

    return PPOTrainer(
        model,
        trainer_config=PPOTrainerConfig(
            learning_rate=1e-3,
            epochs_per_rollout=2,
            minibatch_size=4,
            seed=seed,
        ),
    )


def test_trainer_updates_model() -> None:
    torch.manual_seed(42)

    trainer = make_trainer()

    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    rollout = make_rollout()

    metrics = trainer.update(
        rollout,
        bootstrap_value=0.0,
    )

    changed = any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter
        in trainer.model.named_parameters()
    )

    assert changed is True

    assert metrics.updates > 0
    assert np.isfinite(metrics.total_loss)
    assert np.isfinite(metrics.policy_loss)
    assert np.isfinite(metrics.value_loss)
    assert np.isfinite(metrics.entropy)
    assert np.isfinite(metrics.approximate_kl)
    assert np.isfinite(metrics.clip_fraction)
    assert np.isfinite(metrics.ratio_mean)
    assert np.isfinite(metrics.gradient_norm)


def test_rollout_is_cleared_after_update() -> None:
    trainer = make_trainer()

    rollout = make_rollout(size=8)

    assert rollout.size == 8

    trainer.update(
        rollout,
        bootstrap_value=0.0,
    )

    assert rollout.empty is True


def test_truncated_rollout_accepts_bootstrap_value() -> None:
    trainer = make_trainer()

    rollout = PPORolloutBuffer(
        capacity=4
    )

    for index in range(4):
        rollout.append(
            PPOTransition(
                observation=np.zeros(
                    332,
                    dtype=np.float32,
                ),
                action=0,
                log_probability=-1.5,
                value=0.0,
                reward=1.0,
                terminated=False,
                truncated=(
                    index == 3
                ),
                action_mask=np.ones(
                    5,
                    dtype=bool,
                ),
            )
        )

    metrics = trainer.update(
        rollout,
        bootstrap_value=2.0,
    )

    assert np.isfinite(
        metrics.total_loss
    )


def test_different_seed_changes_minibatch_order() -> None:
    trainer_a = make_trainer(seed=1)
    trainer_b = make_trainer(seed=2)

    batches_a = list(
        trainer_a._minibatches(10)
    )

    batches_b = list(
        trainer_b._minibatches(10)
    )

    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            batches_a,
            batches_b,
        )
    )


def test_repeated_updates_are_possible() -> None:
    trainer = make_trainer()

    first_rollout = make_rollout(
        size=8
    )

    first_metrics = trainer.update(
        first_rollout,
        bootstrap_value=0.0,
    )

    second_rollout = make_rollout(
        size=8
    )

    second_metrics = trainer.update(
        second_rollout,
        bootstrap_value=0.0,
    )

    assert np.isfinite(
        first_metrics.total_loss
    )

    assert np.isfinite(
        second_metrics.total_loss
    )