"""Optional SUMO backend for Step 11 causal episodes."""

from __future__ import annotations

from pathlib import Path

from src.contracts.scenario import Scenario
from src.data.dynamic_episodes import DynamicEpisodeConfig
from src.sim.causal_logger import CausalSumoLogger
from src.sim.fixture import fixture_plan
from src.sim.sumo_runner import run_episode


def run_sumo_causal_episode(
    *,
    scenario: Scenario,
    output_dir: Path,
    config: DynamicEpisodeConfig,
    episode_id: str,
    split: str = "train",
) -> Path:
    output_dir = Path(output_dir)
    logger = CausalSumoLogger(scenario, interval_s=config.interval_s)
    run_episode(
        scenario,
        fixture_plan(scenario),
        output_dir / "sumo_run",
        end_time_s=float(config.duration_s),
        step_length_s=1.0,
        gui=False,
        on_step=logger.on_step,
    )
    logger.write(
        output_dir,
        episode_id=episode_id,
        config=config,
    )
    return output_dir
