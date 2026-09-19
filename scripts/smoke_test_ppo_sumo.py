"""Headless smoke test for the Step-16 PPO/SUMO integration."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Let SUMO/TraCI locate the SUMO installation when the shell has only
# `sumo` on PATH. This also removes the XML-validation warning emitted by
# sumolib when SUMO_HOME is absent.
if not os.environ.get("SUMO_HOME"):
    sumo_bin = shutil.which("sumo")
    if sumo_bin:
        os.environ["SUMO_HOME"] = str(Path(sumo_bin).resolve().parent.parent)

import numpy as np

from src.contracts.scenario import Scenario
from src.learning.ppo_env import PPOEnvConfig, TrafficRoutingPPOEnv
from src.sim.fixture import fixture_plan
from src.sim.ppo_sumo_backend import PPOSumoSimulatorBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=ROOT / "fixtures/step3/base/scenario.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/ppo_sumo_smoke")
    parser.add_argument("--action", type=int, default=0, choices=range(5))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    scenario = Scenario.model_validate_json(args.scenario.read_text())
    initial_plan = fixture_plan(scenario)

    simulator = PPOSumoSimulatorBackend(
        scenario,
        initial_plan,
        output_dir=args.output,
        end_time_s=600.0,
        step_length_s=1.0,
        random_seed=26137,
        gui=args.gui,
        use_subscriptions=True,
    )
    env = TrafficRoutingPPOEnv(
        scenario,
        initial_plan,
        simulator=simulator,
        config=PPOEnvConfig(
            decision_interval_s=60.0,
            episode_duration_s=600.0,
            qpso_particles=8,
            qpso_evaluations=16,
            qpso_seed=26137,
        ),
    )

    try:
        obs, info = env.reset(seed=26137)
        assert obs.shape == (32,), f"unexpected observation shape: {obs.shape}"
        assert np.isfinite(obs).all(), "reset observation contains non-finite values"

        records = []
        for index in range(args.steps):
            if simulator.done:
                break
            next_obs, reward, terminated, truncated, step_info = env.step(args.action)
            assert next_obs.shape == (32,), f"unexpected next observation shape: {next_obs.shape}"
            assert np.isfinite(next_obs).all(), "next observation contains non-finite values"
            assert np.isfinite(reward), f"non-finite reward: {reward}"
            records.append({
                "step": index,
                "action": args.action,
                "reward": float(reward),
                "sim_time_s": float(env.sim_time_s),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "requested_action": step_info.get("requested_action"),
                "proposed_action": step_info.get("proposed_action"),
                "executed_action": step_info.get("executed_action"),
                "overridden": bool(step_info.get("overridden", False)),
                "override_reason": step_info.get("override_reason"),
                "affected_vehicle_ids": list(step_info.get("affected_vehicle_ids", [])),
                "mutable_request_ids": list(step_info.get("mutable_request_ids", [])),
                "qpso_called": bool(step_info.get("qpso_called", False)),
                "qpso_evaluations": int(step_info.get("qpso_evaluations", 0)),
                "qpso_elapsed_s": float(step_info.get("qpso_elapsed_s", 0.0)),
                "route_changed": bool(step_info.get("route_changed", False)),
                "reward_components": step_info.get("reward_components"),
            })
            if terminated or truncated:
                break

        state = simulator.latest_state
        # These are the canonical keys exposed by PPOSumoSimulatorBackend.
        # `mean_speed_ratio` is intentionally used here: the backend does not
        # invent a generic `speed_ratio` scalar or a synthetic `vehicles` blob.
        required = {
            "vehicle_loads",
            "vehicle_capacities",
            "vehicle_current_edge_ids",
            "deadline_slacks_s",
            "active_event_count",
            "congestion_exposure",
            "mean_speed_ratio",
            "remaining_work",
            "completed_work",
            "failed_work",
            "edge_observations",
        }
        missing = required - set(state)
        assert not missing, f"latest_state missing keys: {sorted(missing)}"

        print(json.dumps({
            "status": "PASS",
            "scenario_id": scenario.scenario_id,
            "action": args.action,
            "steps_executed": len(records),
            "records": records,
            "final_sim_time_s": float(simulator.sim_time_s),
            "final_done": bool(simulator.done),
        }, indent=2, default=str))
    finally:
        env.close()


if __name__ == "__main__":
    main()
