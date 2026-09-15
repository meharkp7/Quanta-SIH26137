from __future__ import annotations

import numpy as np
import pytest

from src.learning.ppo_rollout import (
    PPOTransition,
    PPORolloutBuffer,
)


def make_transition(
    *,
    action: int = 0,
    reward: float = 1.0,
) -> PPOTransition:
    return PPOTransition(
        observation=np.arange(
            10,
            dtype=np.float32,
        ),
        action=action,
        log_probability=-0.5,
        value=2.0,
        reward=reward,
        terminated=False,
        truncated=False,
        action_mask=np.asarray(
            [True, True, True, True, True],
            dtype=bool,
        ),
    )


def test_transition_accepts_valid_data() -> None:
    transition = make_transition()

    assert transition.action == 0
    assert transition.reward == 1.0
    assert transition.observation.shape == (10,)
    assert transition.action_mask.shape == (5,)


def test_transition_arrays_are_immutable() -> None:
    transition = make_transition()

    assert transition.observation.flags.writeable is False
    assert transition.action_mask.flags.writeable is False

    with pytest.raises(ValueError):
        transition.observation[0] = 99.0


def test_invalid_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="action must"):
        make_transition(action=5)


def test_action_must_be_feasible() -> None:
    with pytest.raises(
        ValueError,
        match="not feasible",
    ):
        PPOTransition(
            observation=np.zeros(10, dtype=np.float32),
            action=2,
            log_probability=-1.0,
            value=0.0,
            reward=0.0,
            terminated=False,
            truncated=False,
            action_mask=np.asarray(
                [True, True, False, True, True],
                dtype=bool,
            ),
        )


def test_at_least_one_action_must_be_available() -> None:
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        PPOTransition(
            observation=np.zeros(10, dtype=np.float32),
            action=0,
            log_probability=-1.0,
            value=0.0,
            reward=0.0,
            terminated=False,
            truncated=False,
            action_mask=np.zeros(
                5,
                dtype=bool,
            ),
        )


def test_non_finite_observation_is_rejected() -> None:
    observation = np.zeros(10, dtype=np.float32)
    observation[0] = np.nan

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        PPOTransition(
            observation=observation,
            action=0,
            log_probability=-1.0,
            value=0.0,
            reward=0.0,
            terminated=False,
            truncated=False,
            action_mask=np.ones(
                5,
                dtype=bool,
            ),
        )


def test_buffer_capacity_is_enforced() -> None:
    buffer = PPORolloutBuffer(capacity=2)

    buffer.append(make_transition())
    buffer.append(make_transition())

    assert buffer.size == 2
    assert buffer.full is True

    with pytest.raises(RuntimeError, match="full"):
        buffer.append(make_transition())


def test_buffer_preserves_transition_order() -> None:
    buffer = PPORolloutBuffer(capacity=3)

    buffer.append(make_transition(action=0, reward=1.0))
    buffer.append(make_transition(action=2, reward=2.0))
    buffer.append(make_transition(action=4, reward=3.0))

    transitions = buffer.transitions()

    assert [t.action for t in transitions] == [0, 2, 4]
    assert [t.reward for t in transitions] == [
        1.0,
        2.0,
        3.0,
    ]


def test_as_arrays_has_expected_shapes() -> None:
    buffer = PPORolloutBuffer(capacity=3)

    buffer.append(make_transition(action=0))
    buffer.append(make_transition(action=2))
    buffer.append(make_transition(action=4))

    arrays = buffer.as_arrays()

    assert arrays["observations"].shape == (3, 10)
    assert arrays["actions"].shape == (3,)
    assert arrays["log_probabilities"].shape == (3,)
    assert arrays["values"].shape == (3,)
    assert arrays["rewards"].shape == (3,)
    assert arrays["terminated"].shape == (3,)
    assert arrays["truncated"].shape == (3,)
    assert arrays["action_masks"].shape == (3, 5)


def test_as_arrays_preserves_values() -> None:
    buffer = PPORolloutBuffer(capacity=2)

    buffer.append(
        make_transition(
            action=1,
            reward=4.0,
        )
    )

    buffer.append(
        make_transition(
            action=3,
            reward=-2.0,
        )
    )

    arrays = buffer.as_arrays()

    np.testing.assert_array_equal(
        arrays["actions"],
        np.asarray([1, 3]),
    )

    np.testing.assert_allclose(
        arrays["rewards"],
        np.asarray([4.0, -2.0]),
    )


def test_empty_buffer_cannot_be_converted() -> None:
    buffer = PPORolloutBuffer(capacity=5)

    with pytest.raises(
        RuntimeError,
        match="empty",
    ):
        buffer.as_arrays()


def test_clear_resets_buffer() -> None:
    buffer = PPORolloutBuffer(capacity=5)

    buffer.append(make_transition())

    assert buffer.size == 1

    buffer.clear()

    assert buffer.size == 0
    assert buffer.empty is True
    assert buffer.full is False


def test_extend_adds_multiple_transitions() -> None:
    buffer = PPORolloutBuffer(capacity=3)

    buffer.extend(
        [
            make_transition(action=0),
            make_transition(action=1),
            make_transition(action=2),
        ]
    )

    assert buffer.size == 3


def test_extend_cannot_exceed_capacity() -> None:
    buffer = PPORolloutBuffer(capacity=2)

    with pytest.raises(
        RuntimeError,
        match="capacity",
    ):
        buffer.extend(
            [
                make_transition(),
                make_transition(),
                make_transition(),
            ]
        )