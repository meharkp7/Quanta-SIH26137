"""Step 17 — production-style PPO training runner (>= 1,000 episodes).

This runner intentionally reuses the existing Step-17 episode runner and
Step-16 training loop. It adds only long-run orchestration, reproducibility,
metrics persistence, and checkpointing.

The default backend is the existing deterministic lightweight simulator used
for fast PPO development. This is a training/pretraining path; final claims
must still be made on SUMO/held-out scenarios as required by the project plan.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.contracts.scenario import Scenario
from src.learning.ppo_actor_critic import PPOActorCritic, PPOActorCriticConfig
from src.learning.ppo_controller import PPOController
from src.learning.ppo_episode_runner import PPOEpisodeRunner
from src.learning.ppo_policy_state import StructuredPPORepresentationProvider
from src.learning.ppo_policy_state_runtime import PPOPolicyStateRuntime
from src.learning.ppo_rollout import PPORolloutBuffer
from src.learning.ppo_rollout_collector import PPORolloutCollector
from src.learning.ppo_runtime import PPORuntime
from src.learning.ppo_trainer import PPOTrainer, PPOTrainerConfig
from src.learning.ppo_training_loop import PPOTrainingLoop
from src.sim.fixture import fixture_plan

# Development simulator is intentionally imported lazily so importing this
# module does not make the production package depend on test code.
from tests.step16.test_ppo_env import DeterministicSimulator


SCENARIO_CASES = (
    "feasible_reference",
    "early_arrival_wait",
    "closure_with_detour",
)


def load_scenarios(project_root: Path) -> list[Scenario]:
    root = project_root / "fixtures" / "step3" / "cases"
    scenarios: list[Scenario] = []
    for case in SCENARIO_CASES:
        path = root / case / "scenario.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing training scenario: {path}")
        scenarios.append(Scenario.model_validate(json.loads(path.read_text())))
    return scenarios


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_stack(scenario: Scenario, seed: int):
    env = __import__("src.learning.ppo_env", fromlist=["TrafficRoutingPPOEnv"]).TrafficRoutingPPOEnv(
        scenario,
        fixture_plan(scenario),
        simulator=DeterministicSimulator(),
    )

    state_runtime = PPOPolicyStateRuntime(scenario=scenario)
    provider = StructuredPPORepresentationProvider()

    model = PPOActorCritic(
        provider.spec,
        PPOActorCriticConfig(action_dim=5),
    )
    runtime = PPORuntime(
        model=model,
        representation_provider=provider,
        device="cpu",
        deterministic=False,
    )
    controller = PPOController(runtime=runtime, representation_provider=provider)

    buffer = PPORolloutBuffer(capacity=64)
    collector = PPORolloutCollector(
        env=env,
        controller=controller,
        representation_provider=provider,
        rollout_buffer=buffer,
    )
    runner = PPOEpisodeRunner(
        env=env,
        state_runtime=state_runtime,
        collector=collector,
    )

    trainer = PPOTrainer(
        model,
        trainer_config=PPOTrainerConfig(
            seed=seed,
            learning_rate=3e-4,
            epochs_per_rollout=4,
            minibatch_size=64,
        ),
        device="cpu",
    )
    loop = PPOTrainingLoop(
        collector=collector,
        trainer=trainer,
        rollout_buffer=buffer,
    )
    return runner, loop, trainer, model, buffer


def checkpoint(path: Path, *, episode: int, model, trainer, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "seed": seed,
            "representation_version": model.representation_spec.version,
            "representation_dim": model.representation_spec.total_dim,
            "trainer_config": {
                "learning_rate": trainer.trainer_config.learning_rate,
                "epochs_per_rollout": trainer.trainer_config.epochs_per_rollout,
                "minibatch_size": trainer.trainer_config.minibatch_size,
                "seed": trainer.trainer_config.seed,
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ppo_training"))
    args = parser.parse_args()

    if args.episodes < 1000:
        raise ValueError("Actual PPO training run must contain at least 1000 episodes")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")

    project_root = Path.cwd()
    scenarios = load_scenarios(project_root)
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "training_metrics.csv"
    manifest_path = args.output_dir / "training_manifest.json"
    latest_path = args.output_dir / "ppo_latest.pt"
    best_path = args.output_dir / "ppo_best.pt"

    manifest = {
        "episodes_requested": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "scenario_cases": list(SCENARIO_CASES),
        "forecaster": "persistence-pilot",
        "forecaster_frozen_within_update": True,
        "device": "cpu",
        "representation_dim": 332,
        "action_dim": 5,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # One policy/model is retained across all episodes. The environment and
    # state runtime are rebuilt per scenario so each rollout starts cleanly.
    # A single trainer/model is used, preserving optimizer state across updates.
    runner, loop, trainer, model, buffer = build_stack(scenarios[0], args.seed)

    fieldnames = [
        "episode", "scenario_id", "transitions", "episode_reward",
        "mean_reward", "total_loss", "policy_loss", "value_loss",
        "entropy", "approximate_kl", "clip_fraction", "ratio_mean",
        "mean_advantage", "mean_return", "gradient_norm", "wall_seconds",
    ]
    best_reward = float("-inf")
    start = time.perf_counter()

    with metrics_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            scenario = scenarios[(episode - 1) % len(scenarios)]

            # Rebuild only the environment-side stack when the scenario changes;
            # copy the learned model/optimizer into the new stack.
            if scenario is not runner.env.scenario:
                old_model = model
                runner, loop, trainer_new, _, buffer = build_stack(scenario, args.seed)
                trainer_new.model.load_state_dict(old_model.state_dict())
                trainer_new.optimizer.load_state_dict(trainer.optimizer.state_dict())
                trainer = trainer_new
                model = trainer.model

            if not buffer.empty:
                raise RuntimeError("rollout buffer was not empty before a fresh episode")

            t0 = time.perf_counter()
            result = runner.run_episode(
                episode_id=f"ppo-train-{episode:05d}",
                max_steps=args.max_steps,
            )
            iteration = loop.update()
            wall = time.perf_counter() - t0
            m = iteration.metrics

            row = {
                "episode": episode,
                "scenario_id": scenario.scenario_id,
                "transitions": iteration.transitions_collected,
                "episode_reward": result.total_reward,
                "mean_reward": m.mean_reward,
                "total_loss": m.total_loss,
                "policy_loss": m.policy_loss,
                "value_loss": m.value_loss,
                "entropy": m.entropy,
                "approximate_kl": m.approximate_kl,
                "clip_fraction": m.clip_fraction,
                "ratio_mean": m.ratio_mean,
                "mean_advantage": m.mean_advantage,
                "mean_return": m.mean_return,
                "gradient_norm": m.gradient_norm,
                "wall_seconds": wall,
            }
            writer.writerow(row)
            fh.flush()

            if result.total_reward > best_reward:
                best_reward = result.total_reward
                checkpoint(best_path, episode=episode, model=model, trainer=trainer, seed=args.seed)

            if episode % args.checkpoint_every == 0 or episode == args.episodes:
                checkpoint(latest_path, episode=episode, model=model, trainer=trainer, seed=args.seed)

            if episode == 1 or episode % 25 == 0 or episode == args.episodes:
                elapsed = time.perf_counter() - start
                print(
                    f"episode={episode:4d}/{args.episodes} "
                    f"scenario={scenario.scenario_id} "
                    f"reward={result.total_reward: .6f} "
                    f"loss={m.total_loss: .6f} "
                    f"elapsed={elapsed: .1f}s",
                    flush=True,
                )

    print(f"Training complete: {args.episodes} episodes")
    print(f"Metrics: {metrics_path}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
