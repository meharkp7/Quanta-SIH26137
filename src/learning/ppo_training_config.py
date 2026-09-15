from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Configuration for reproducible on-policy PPO training."""

    seed: int = 26137

    total_env_steps: int = 10_000
    rollout_steps: int = 256
    max_episode_steps: int = 10

    checkpoint_interval_updates: int = 10
    evaluation_interval_updates: int = 10

    curriculum_level: int = 0

    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.total_env_steps <= 0:
            raise ValueError("total_env_steps must be positive")

        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")

        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")

        if self.checkpoint_interval_updates <= 0:
            raise ValueError(
                "checkpoint_interval_updates must be positive"
            )

        if self.evaluation_interval_updates <= 0:
            raise ValueError(
                "evaluation_interval_updates must be positive"
            )

        if self.curriculum_level < 0:
            raise ValueError(
                "curriculum_level must be non-negative"
            )

        if not self.device.strip():
            raise ValueError("device must not be empty")