"""Step 17 — real-backend PPO training on the Delhi SUMO corpus.

This is the production training launcher for the existing Step-16/17 PPO
stack.  Unlike ``train_ppo_1000.py``, this runner uses the real
``PPOSumoSimulatorBackend`` and the generated Delhi corpus.

For each corpus episode it:
  * loads the recorded map/scenario;
  * reconstructs the corpus generator's incumbent route deterministically;
  * loads the episode's real event timeline;
  * reveals affected vehicles only at/after each event's reveal time;
  * runs PPO through PPOEpisodeRunner + PPOTrainingLoop against real SUMO;
  * preserves model and optimizer state across scenario rebuilds;
  * writes per-episode metrics and best/latest checkpoints.

Always run ``--smoke-test`` before a long unattended run.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.contracts.scenario import Scenario
from src.learning.ppo_actor_critic import PPOActorCritic, PPOActorCriticConfig
from src.learning.ppo_controller import PPOController
from src.learning.ppo_env import PPOEnvConfig, TrafficRoutingPPOEnv
from src.learning.ppo_episode_runner import PPOEpisodeRunner
from src.learning.ppo_policy_state import StructuredPPORepresentationProvider
from src.learning.ppo_policy_state_runtime import PPOPolicyStateRuntime
from src.learning.ppo_rollout import PPORolloutBuffer
from src.learning.ppo_rollout_collector import PPORolloutCollector
from src.learning.ppo_runtime import PPORuntime
from src.learning.ppo_trainer import PPOTrainer, PPOTrainerConfig
from src.learning.ppo_training_loop import PPOTrainingLoop
from src.routing.initial_solution import InitialSolutionConfig
from src.routing.pipeline import RoutingPipeline, RoutingPipelineConfig
from src.routing.route_evaluator import RouteEvaluator
from src.runtime.scope_actions import JobImpact
from src.sim.incidents import IncidentConfig
from src.sim.ppo_sumo_backend import PPOSumoSimulatorBackend


_ROUTE_STRATEGIES = (
    ("earliest_deadline", RoutingPipelineConfig()),
    (
        "nearest_feasible",
        RoutingPipelineConfig(
            initial_solution=InitialSolutionConfig(
                heuristic="nearest_feasible",
                customer_ordering="nearest_feasible",
            )
        ),
    ),
    (
        "earliest_release",
        RoutingPipelineConfig(
            initial_solution=InitialSolutionConfig(
                heuristic="earliest_release",
                customer_ordering="earliest_release",
            )
        ),
    ),
    (
        "largest_demand",
        RoutingPipelineConfig(
            initial_solution=InitialSolutionConfig(
                heuristic="largest_demand",
                customer_ordering="largest_demand",
            )
        ),
    ),
    (
        "customer_id",
        RoutingPipelineConfig(
            initial_solution=InitialSolutionConfig(
                heuristic="customer_id",
                customer_ordering="customer_id",
            )
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class DelhiEpisodeSpec:
    episode_id: str
    map_index: int
    scenario_path: Path
    episode_dir: Path
    route_strategy: str
    route_strategy_attempt: int
    duration_s: int
    regime: str
    event_type: str | None
    zone: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_delhi_corpus(corpus_dir: Path, *, split: str = "train") -> list[DelhiEpisodeSpec]:
    manifest_path = corpus_dir / "corpus_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No corpus_manifest.json found at {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    map_records: dict[str, dict[str, Any]] = manifest["map_records"]
    specs: list[DelhiEpisodeSpec] = []

    for episode in manifest["episodes"]:
        if episode["split"] != split:
            continue
        map_index = int(episode["map_index"])
        map_record = map_records[str(map_index)]
        specs.append(
            DelhiEpisodeSpec(
                episode_id=str(episode["episode_id"]),
                map_index=map_index,
                scenario_path=corpus_dir / map_record["scenario_path"],
                episode_dir=corpus_dir / episode["episode_dir"],
                route_strategy=str(episode["route_strategy"]),
                route_strategy_attempt=int(episode["route_strategy_attempt"]),
                duration_s=int(episode["duration_s"]),
                regime=str(episode["regime"]),
                event_type=episode.get("event_type"),
                zone=str(map_record["zone"]),
            )
        )

    if not specs:
        raise RuntimeError(
            f"No episodes found for split={split!r}; "
            f"available counts={manifest.get('split_episode_counts')}"
        )
    return specs


def route_pipeline_config(strategy_name: str, attempt: int) -> RoutingPipelineConfig:
    for name, config in _ROUTE_STRATEGIES:
        if name == strategy_name:
            return config
    print(
        f"WARNING: unknown route_strategy={strategy_name!r}; "
        f"falling back to positional attempt {attempt}",
        flush=True,
    )
    return _ROUTE_STRATEGIES[attempt % len(_ROUTE_STRATEGIES)][1]


def build_route_plan(scenario: Scenario, spec: DelhiEpisodeSpec):
    result = RoutingPipeline(
        scenario,
        config=route_pipeline_config(spec.route_strategy, spec.route_strategy_attempt),
    ).solve(planning_time_s=0.0)
    if not result.feasible or not result.complete or result.final_route_plan is None:
        raise RuntimeError(
            f"Could not reconstruct route plan for {spec.episode_id}: "
            f"feasible={result.feasible} complete={result.complete} "
            f"errors={result.errors!r}"
        )
    return result.final_route_plan


def load_events(spec: DelhiEpisodeSpec) -> list[dict[str, Any]]:
    path = spec.episode_dir / "events.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def edges_for_parent_road(scenario: Scenario, parent_road_id: str) -> list[str]:
    return [
        str(edge.edge_id)
        for edge in scenario.edges
        if str(edge.parent_road_id) == str(parent_road_id)
    ]


def build_incident_configs(
    scenario: Scenario,
    events: list[dict[str, Any]],
) -> tuple[IncidentConfig, ...]:
    configs: list[IncidentConfig] = []
    for event in events:
        effect_start = float(event["effect_start_s"])
        effect_end = float(event["effect_end_s"])
        reveal_time = float(event["reveal_time_s"])
        duration = max(0.0, effect_end - effect_start)
        announce_lead = max(0.0, effect_start - reveal_time)

        for parent_road_id in event.get("affected_parent_road_ids", ()):
            for edge_id in edges_for_parent_road(scenario, parent_road_id):
                configs.append(
                    IncidentConfig(
                        incident_id=f"{event['event_id']}:{edge_id}",
                        edge_id=edge_id,
                        trigger_time_s=effect_start,
                        duration_s=duration,
                        announce_lead_s=announce_lead,
                    )
                )
    return tuple(configs)


def build_job_impacts_and_provider(
    scenario: Scenario,
    route_plan: Any,
    evaluation: Any,
    events: list[dict[str, Any]],
):
    physical_edges_by_vehicle = {
        str(v.vehicle_id): set(v.physical_route.edge_ids) if v.physical_route else set()
        for v in evaluation.vehicle_evaluations
    }
    request_to_vehicle = {
        str(customer_id): str(route.vehicle_id)
        for route in route_plan.vehicle_routes
        for customer_id in route.customer_ids
    }

    event_edges_and_reveal: list[tuple[set[str], float]] = []
    for event in events:
        edges: set[str] = set()
        for parent_road_id in event.get("affected_parent_road_ids", ()):
            edges.update(edges_for_parent_road(scenario, parent_road_id))
        if edges:
            event_edges_and_reveal.append((edges, float(event["reveal_time_s"])))

    def overlaps(vehicle_id: str) -> bool:
        vehicle_edges = physical_edges_by_vehicle.get(vehicle_id, set())
        return any(vehicle_edges & edges for edges, _ in event_edges_and_reveal)

    def earliest_reveal(vehicle_id: str) -> float:
        vehicle_edges = physical_edges_by_vehicle.get(vehicle_id, set())
        reveals = [
            reveal for edges, reveal in event_edges_and_reveal
            if vehicle_edges & edges
        ]
        return min(reveals) if reveals else float("inf")

    impacts = tuple(
        JobImpact(
            request_id=str(request.request_id),
            vehicle_id=request_to_vehicle[str(request.request_id)],
            zone_id=None,
            affected=overlaps(request_to_vehicle[str(request.request_id)]),
            deadline_slack_s=float(request.latest_service_start_s),
            route_overlap_fraction=(
                1.0 if overlaps(request_to_vehicle[str(request.request_id)]) else 0.0
            ),
        )
        for request in scenario.requests
        if str(request.request_id) in request_to_vehicle
    )

    def affected_vehicle_provider(sim_time_s: float) -> tuple[str, ...]:
        return tuple(
            sorted(
                vehicle_id
                for vehicle_id in physical_edges_by_vehicle
                if earliest_reveal(vehicle_id) <= float(sim_time_s)
                and overlaps(vehicle_id)
            )
        )

    return impacts, affected_vehicle_provider


def build_stack(
    spec: DelhiEpisodeSpec,
    corpus_dir: Path,
    seed: int,
    output_root: Path,
    step_length_s: float,
):
    scenario = Scenario.model_validate_json(
        spec.scenario_path.read_text(encoding="utf-8")
    )
    route_plan = build_route_plan(scenario, spec)
    evaluator = RouteEvaluator(scenario)
    evaluation = evaluator.evaluate(route_plan, planning_time_s=0.0)
    if not evaluation.feasible:
        raise RuntimeError(
            f"Reconstructed route plan for {spec.episode_id} is infeasible: "
            f"{evaluation.errors!r}"
        )

    events = load_events(spec)
    incident_configs = build_incident_configs(scenario, events)
    job_impacts, affected_vehicle_provider = build_job_impacts_and_provider(
        scenario, route_plan, evaluation, events
    )

    simulator = PPOSumoSimulatorBackend(
        scenario,
        route_plan,
        output_dir=output_root / "sumo_runs" / spec.episode_id,
        end_time_s=float(spec.duration_s),
        step_length_s=step_length_s,
        random_seed=seed,
        incident_configs=incident_configs,
        use_subscriptions=True,
    )
    env = TrafficRoutingPPOEnv(
        scenario,
        route_plan,
        simulator=simulator,
        job_impacts=job_impacts,
        affected_vehicle_provider=affected_vehicle_provider,
        config=PPOEnvConfig(
            decision_interval_s=60.0,
            episode_duration_s=float(spec.duration_s),
            qpso_particles=8,
            qpso_evaluations=16,
            qpso_seed=seed,
        ),
    )

    state_runtime = PPOPolicyStateRuntime(scenario=scenario)
    provider = StructuredPPORepresentationProvider()
    model = PPOActorCritic(provider.spec, PPOActorCriticConfig(action_dim=5))
    runtime = PPORuntime(
        model=model,
        representation_provider=provider,
        device="mps",
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
        device="mps",
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


def run_smoke_test(specs, corpus_dir: Path, seed: int, output_dir: Path, step_length_s: float) -> None:
    spec = specs[0]
    print(
        f"Smoke-testing one real SUMO episode: {spec.episode_id} "
        f"(zone={spec.zone}, regime={spec.regime}, event={spec.event_type})",
        flush=True,
    )
    runner, loop, trainer, model, buffer = build_stack(
        spec, corpus_dir, seed, output_dir, step_length_s
    )
    try:
        result = runner.run_episode(
            episode_id=f"smoke-{spec.episode_id}",
            max_steps=5,
        )
        print(f"Episode reward: {result.total_reward}", flush=True)
        print(f"Transitions collected: {buffer.size}", flush=True)
        iteration = loop.update()
        print(f"PPO update metrics: {iteration.metrics}", flush=True)
        print("SMOKE TEST PASSED", flush=True)
    finally:
        runner.env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus_delhi"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=26137)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--step-length-s", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ppo_training_delhi"))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.step_length_s <= 0:
        raise ValueError("--step-length-s must be positive")

    corpus_dir = args.corpus_dir.resolve()
    output_dir = args.output_dir.resolve()
    specs = load_delhi_corpus(corpus_dir, split=args.split)
    print(
        f"Loaded {len(specs)} {args.split}-split episodes from {corpus_dir}",
        flush=True,
    )
    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        run_smoke_test(specs, corpus_dir, args.seed, output_dir, args.step_length_s)
        return 0

    metrics_path = output_dir / "training_metrics.csv"
    manifest_path = output_dir / "training_manifest.json"
    latest_path = output_dir / "ppo_latest.pt"
    best_path = output_dir / "ppo_best.pt"

    manifest = {
        "episodes_requested": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "corpus_dir": str(corpus_dir),
        "corpus_split": args.split,
        "corpus_episode_count": len(specs),
        "backend": "sumo",
        "forecaster": "persistence-pilot",
        "forecaster_frozen_within_update": True,
        "device": "cpu",
        "representation_dim": 332,
        "action_dim": 5,
        "decision_interval_s": 60.0,
        "qpso_particles": 8,
        "qpso_evaluations": 16,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    runner, loop, trainer, model, buffer = build_stack(
        specs[0], corpus_dir, args.seed, output_dir, args.step_length_s
    )
    current_spec = specs[0]
    best_reward = float("-inf")
    start = time.perf_counter()

    fieldnames = [
        "episode", "episode_id", "map_index", "zone", "regime", "event_type",
        "transitions", "episode_reward", "mean_reward", "total_loss",
        "policy_loss", "value_loss", "entropy", "approximate_kl",
        "clip_fraction", "ratio_mean", "mean_advantage", "mean_return",
        "gradient_norm", "wall_seconds",
        "decision_seconds", "qpso_seconds", "other_episode_seconds",
        "qpso_calls", "qpso_evaluations", "steps_with_qpso",
        "decision_mean_seconds", "decision_max_seconds",
        "qpso_mean_seconds", "qpso_max_seconds",
        "affected_state_seconds", "action_mask_seconds",
        "scope_selection_seconds", "simulator_advance_seconds",
        "reward_seconds", "state_commit_seconds",
        "ppo_update_seconds",
    ]

    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            for episode in range(1, args.episodes + 1):
                spec = specs[(episode - 1) % len(specs)]

                if spec.episode_id != current_spec.episode_id:
                    old_model = model
                    old_trainer = trainer
                    runner.env.close()
                    runner, loop, trainer_new, _, buffer = build_stack(
                        spec, corpus_dir, args.seed, output_dir, args.step_length_s
                    )
                    trainer_new.model.load_state_dict(old_model.state_dict())
                    trainer_new.optimizer.load_state_dict(old_trainer.optimizer.state_dict())
                    trainer = trainer_new
                    model = trainer.model
                    current_spec = spec

                if not buffer.empty:
                    raise RuntimeError("rollout buffer was not empty before a fresh episode")

                t0 = time.perf_counter()
                result = runner.run_episode(
                    episode_id=f"ppo-train-{episode:05d}",
                    max_steps=args.max_steps,
                )
                # The environment already exposes per-decision timing in each
                # rollout step's info. Aggregate it here rather than adding
                # another timing layer inside the execution path.
                step_infos = [step.info for step in result.steps]
                decision_times = [
                    float(info.get("decision_elapsed_s", 0.0))
                    for info in step_infos
                ]
                qpso_times = [
                    float(info.get("qpso_elapsed_s", 0.0))
                    for info in step_infos
                ]
                qpso_calls = sum(
                    1 for info in step_infos if bool(info.get("qpso_called", False))
                )
                qpso_evaluations = sum(
                    int(info.get("qpso_evaluations", 0)) for info in step_infos
                )

                timing_keys = (
                    "affected_state_seconds",
                    "action_mask_seconds",
                    "scope_selection_seconds",
                    "qpso_seconds",
                    "simulator_advance_seconds",
                    "reward_seconds",
                    "state_commit_seconds",
                )
                phase_totals = {
                    key: float(sum(float(info.get("timing", {}).get(key, 0.0)) for info in step_infos))
                    for key in timing_keys
                }

                update_t0 = time.perf_counter()
                iteration = loop.update()
                ppo_update_seconds = time.perf_counter() - update_t0
                wall = time.perf_counter() - t0
                m = iteration.metrics

                decision_seconds = float(sum(decision_times))
                qpso_seconds = float(sum(qpso_times))
                other_episode_seconds = max(
                    0.0, wall - decision_seconds - ppo_update_seconds
                )

                writer.writerow({
                    "episode": episode,
                    "episode_id": spec.episode_id,
                    "map_index": spec.map_index,
                    "zone": spec.zone,
                    "regime": spec.regime,
                    "event_type": spec.event_type,
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
                    "decision_seconds": decision_seconds,
                    "qpso_seconds": qpso_seconds,
                    "other_episode_seconds": other_episode_seconds,
                    "qpso_calls": qpso_calls,
                    "qpso_evaluations": qpso_evaluations,
                    "steps_with_qpso": qpso_calls,
                    "decision_mean_seconds": (
                        decision_seconds / len(decision_times) if decision_times else 0.0
                    ),
                    "decision_max_seconds": max(decision_times, default=0.0),
                    "qpso_mean_seconds": (
                        qpso_seconds / len(qpso_times) if qpso_times else 0.0
                    ),
                    "qpso_max_seconds": max(qpso_times, default=0.0),
                    "affected_state_seconds": phase_totals["affected_state_seconds"],
                    "action_mask_seconds": phase_totals["action_mask_seconds"],
                    "scope_selection_seconds": phase_totals["scope_selection_seconds"],
                    "simulator_advance_seconds": phase_totals["simulator_advance_seconds"],
                    "reward_seconds": phase_totals["reward_seconds"],
                    "state_commit_seconds": phase_totals["state_commit_seconds"],
                    "ppo_update_seconds": ppo_update_seconds,
                })
                fh.flush()

                if result.total_reward > best_reward:
                    best_reward = result.total_reward
                    checkpoint(
                        best_path,
                        episode=episode,
                        model=model,
                        trainer=trainer,
                        seed=args.seed,
                    )

                if episode % args.checkpoint_every == 0 or episode == args.episodes:
                    checkpoint(
                        latest_path,
                        episode=episode,
                        model=model,
                        trainer=trainer,
                        seed=args.seed,
                    )

                if episode == 1 or episode % 25 == 0 or episode == args.episodes:
                    elapsed = time.perf_counter() - start
                    print(
                        f"episode={episode:4d}/{args.episodes} "
                        f"zone={spec.zone} regime={spec.regime} "
                        f"event={spec.event_type or 'none'} "
                        f"reward={result.total_reward: .6f} "
                        f"loss={m.total_loss: .6f} "
                        f"elapsed={elapsed: .1f}s",
                        flush=True,
                    )
    finally:
        runner.env.close()

    print(f"Training complete: {args.episodes} episodes")
    print(f"Metrics: {metrics_path}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
