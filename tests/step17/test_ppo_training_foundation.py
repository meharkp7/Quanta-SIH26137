from __future__ import annotations

import numpy as np
import pytest

from src.learning.ppo_curriculum import (
    get_curriculum_level,
    validate_curriculum_progression,
)
from src.learning.ppo_run_manifest import PPORunManifest
from src.learning.ppo_training_config import PPOTrainingConfig


def test_training_config_has_reproducible_defaults():
    config = PPOTrainingConfig()

    assert config.seed == 26137
    assert config.total_env_steps > 0
    assert config.rollout_steps > 0
    assert config.device == "cpu"


def test_training_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        PPOTrainingConfig(total_env_steps=0)

    with pytest.raises(ValueError):
        PPOTrainingConfig(rollout_steps=0)

    with pytest.raises(ValueError):
        PPOTrainingConfig(curriculum_level=-1)


def test_curriculum_levels_are_ordered():
    assert get_curriculum_level(0).name == "smoke"
    assert get_curriculum_level(1).name == "small_sumo"
    assert get_curriculum_level(2).name == "multi_event"
    assert get_curriculum_level(3).name == "large"


def test_curriculum_cannot_skip_levels():
    validate_curriculum_progression(0, 1)

    with pytest.raises(ValueError):
        validate_curriculum_progression(0, 2)


def test_manifest_round_trip(tmp_path):
    manifest = PPORunManifest(
        run_id="test-run",
        seed=26137,
        representation_version="ppo-representation-v1",
        forecaster_version="forecaster-v1",
        reward_version="reward-v1",
        simulator_version="fixture-v1",
        qpso_version="qpso-v1",
        curriculum_level=0,
        total_env_steps=100,
        rollout_steps=10,
        device="cpu",
    )

    path = tmp_path / "manifest.json"

    manifest.write(path)

    loaded = PPORunManifest.read(path)

    assert loaded == manifest