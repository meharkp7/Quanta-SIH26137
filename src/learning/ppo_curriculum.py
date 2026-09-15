from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PPOCurriculumLevel:
    """One documented PPO training curriculum level."""

    level: int
    name: str
    max_episode_steps: int
    description: str


CURRICULUM: tuple[PPOCurriculumLevel, ...] = (
    PPOCurriculumLevel(
        level=0,
        name="smoke",
        max_episode_steps=10,
        description=(
            "Small deterministic environment used to verify the "
            "complete PPO rollout/update loop."
        ),
    ),
    PPOCurriculumLevel(
        level=1,
        name="small_sumo",
        max_episode_steps=30,
        description=(
            "Small SUMO maps with simple incidents and short episodes."
        ),
    ),
    PPOCurriculumLevel(
        level=2,
        name="multi_event",
        max_episode_steps=60,
        description=(
            "Multiple traffic events with increased service and "
            "deadline pressure."
        ),
    ),
    PPOCurriculumLevel(
        level=3,
        name="large",
        max_episode_steps=120,
        description=(
            "Larger fleet/network scenarios with multiple disruptions."
        ),
    ),
)


def get_curriculum_level(level: int) -> PPOCurriculumLevel:
    if level < 0 or level >= len(CURRICULUM):
        raise ValueError(
            f"unknown curriculum level: {level}"
        )

    return CURRICULUM[level]


def validate_curriculum_progression(
    previous_level: int,
    next_level: int,
) -> None:
    if next_level < previous_level:
        raise ValueError(
            "curriculum level cannot move backwards"
        )

    if next_level > previous_level + 1:
        raise ValueError(
            "curriculum cannot skip levels"
        )