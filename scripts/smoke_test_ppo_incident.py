"""Incident-driven Step-16 PPO/SUMO integration smoke test."""
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

if not os.environ.get("SUMO_HOME"):
    sumo_bin = shutil.which("sumo")
    if sumo_bin:
        os.environ["SUMO_HOME"] = str(Path(sumo_bin).resolve().parent.parent)

from src.contracts.scenario import Scenario
from src.learning.ppo_env import PPOEnvConfig, TrafficRoutingPPOEnv
from src.routing.initial_solution import InitialSolutionBuilder
from src.routing.route_evaluator import RouteEvaluator
from src.runtime.scope_actions import JobImpact
from src.sim.ppo_sumo_backend import PPOSumoSimulatorBackend


def build_job_impacts(scenario, initial_plan, evaluation, incident_edge: str) -> tuple[JobImpact, ...]:
    physical_edges_by_vehicle = {
        str(v.vehicle_id): set(v.physical_route.edge_ids) if v.physical_route else set()
        for v in evaluation.vehicle_evaluations
    }
    request_to_vehicle = {
        str(customer_id): str(route.vehicle_id)
        for route in initial_plan.vehicle_routes
        for customer_id in route.customer_ids
    }
    return tuple(
        JobImpact(
            request_id=str(request.request_id),
            vehicle_id=request_to_vehicle[str(request.request_id)],
            zone_id=None,
            affected=incident_edge in physical_edges_by_vehicle.get(
                request_to_vehicle[str(request.request_id)], set()
            ),
            deadline_slack_s=float(request.latest_service_start_s),
            route_overlap_fraction=(
                1.0
                if incident_edge in physical_edges_by_vehicle.get(
                    request_to_vehicle[str(request.request_id)], set()
                )
                else 0.0
            ),
        )
        for request in scenario.requests
        if str(request.request_id) in request_to_vehicle
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=ROOT / "fixtures/step3/base/scenario.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/ppo_incident_smoke")
    parser.add_argument("--action", type=int, default=1, choices=range(5))
    parser.add_argument("--warmup-s", type=float, default=60.0)
    parser.add_argument("--incident-edge", default="E23")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    scenario = Scenario.model_validate_json(args.scenario.read_text())
    evaluator = RouteEvaluator(scenario)
    initial_solution = InitialSolutionBuilder(scenario, evaluator).build(planning_time_s=0.0)
    if not initial_solution.feasible:
        raise RuntimeError("InitialSolutionBuilder produced an infeasible plan: " + "; ".join(initial_solution.errors))

    initial_plan = initial_solution.route_plan
    evaluation = initial_solution.evaluation
    job_impacts = build_job_impacts(scenario, initial_plan, evaluation, args.incident_edge)
    affected_vehicle_ids = tuple(sorted({j.vehicle_id for j in job_impacts if j.affected}))
    mutable_request_ids = tuple(sorted(j.request_id for j in job_impacts if j.affected))

    print("\n=== INCIDENT PREFLIGHT ===")
    print("incident_edge:", args.incident_edge)
    print("affected_vehicle_ids:", affected_vehicle_ids)
    print("mutable_request_ids:", mutable_request_ids)
    if not affected_vehicle_ids or not mutable_request_ids:
        raise RuntimeError(
            f"Incident edge {args.incident_edge!r} does not occur in the initial physical route plan. "
            f"affected_vehicle_ids={affected_vehicle_ids}, mutable_request_ids={mutable_request_ids}. "
            "Choose a fixture/scenario whose initial route traverses the incident edge."
        )

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

    def affected_vehicle_provider(_sim_time_s):
        return affected_vehicle_ids

    env = TrafficRoutingPPOEnv(
        scenario,
        initial_plan,
        simulator=simulator,
        job_impacts=job_impacts,
        affected_vehicle_provider=affected_vehicle_provider,
        config=PPOEnvConfig(
            decision_interval_s=60.0,
            episode_duration_s=600.0,
            qpso_particles=8,
            qpso_evaluations=16,
            qpso_seed=26137,
        ),
    )

    try:
        env.reset(seed=26137)
        # Advance to t=60 without taking a replanning action. The built-in
        # Step-6 incident triggers at t=50 and is therefore closed/visible.
        simulator.advance(args.warmup_s)
        env._sim_time_s = simulator.sim_time_s

        closed_edges = list(simulator.closed_edge_ids)
        if args.incident_edge not in closed_edges:
            raise RuntimeError(
                f"Expected incident edge {args.incident_edge!r} to be closed after warmup; "
                f"closed_edges={closed_edges}"
            )

        print("\n=== BEFORE REPLAN ===")
        print("closed_edges:", closed_edges)
        print("affected_vehicle_ids:", affected_vehicle_ids)
        print("mutable_request_ids:", mutable_request_ids)

        _, _, terminated, truncated, info = env.step(args.action)

        result = {
            "status": "PASS",
            "scenario_id": str(scenario.scenario_id),
            "incident_edge": args.incident_edge,
            "closed_edges_before_action": closed_edges,
            "warmup_s": args.warmup_s,
            "requested_action": info.get("requested_action"),
            "executed_action": info.get("executed_action"),
            "overridden": info.get("overridden"),
            "override_reason": info.get("override_reason"),
            "affected_vehicle_ids": info.get("affected_vehicle_ids"),
            "mutable_request_ids": info.get("mutable_request_ids"),
            "qpso_called": info.get("qpso_called"),
            "qpso_evaluations": info.get("qpso_evaluations"),
            "route_changed": info.get("route_changed"),
            "sim_time_s": info.get("sim_time_s"),
            "terminated": terminated,
            "truncated": truncated,
            # Diagnostics for *why* terminated fired (all-work-delivered
            # vs. simulator giving up), since terminated alone can't tell
            # these apart and they mean very different things.
            "remaining_work": info.get("remaining_work"),
            "simulator_done": info.get("simulator_done"),
            "reward": info.get("reward_components"),
        }
        print(json.dumps(result, indent=2, default=str))
    finally:
        env.close()


if __name__ == "__main__":
    main()