"""First executable Step 10 operating loop.

The loop deliberately uses only Step-10 baselines:

    scenario -> observation snapshot -> persistence forecast -> rule scope
    -> QPSO -> independent validation -> commit -> SUMO advance -> outcomes

GNN/Transformer forecasting and PPO scope selection are not substituted here;
they remain explicit baselines until their later steps are integrated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.platform.service import PlatformService, SolveOptions
from src.platform.catalog import PROJECT_ROOT, load_scenario
from src.contracts.routing import RoutePlan as ExecutableRoutePlan
from src.contracts.routing import StopLeg, VehicleRoute


@dataclass
class LoopState:
    scenario_id: str
    closed_edge_ids: tuple[str, ...]
    state_version: str
    route_version: str
    forecast_mode: str
    scope_action: str
    committed: bool
    plan: dict[str, list[str]]
    evaluation: dict
    solve: dict
    sumo_episode: dict | None = None
    notes: list[str] = field(default_factory=list)


class DemoLoop:
    def __init__(self, service: PlatformService | None = None) -> None:
        self.service = service or PlatformService()

    def run(
        self,
        scenario_id: str = "S3_BASE",
        *,
        closed_edge_ids: Sequence[str] = (),
        particles: int = 12,
        evaluations: int = 40,
        seed: int = 7,
        method: str = "qpso",
        execute_sumo: bool = True,
        output_dir: Path | None = None,
    ) -> LoopState:
        closed = tuple(closed_edge_ids)
        state_version = f"{scenario_id}:t0:graph:{','.join(sorted(closed)) or 'open'}"
        route_version = f"step10-{seed}-t0"
        notes = [
            "Forecast = persistence of the latest visible free-flow speeds (baseline, not a trained GNN/Transformer).",
            "Scope = rule policy (GLOBAL if a road closure is visible, otherwise KEEP).",
            f"Observation captured at simulation time 0.0s with state_version={state_version}.",
        ]
        action = "GLOBAL" if closed else "KEEP"
        notes.append(
            "Closed roads are visible, so the baseline requests a GLOBAL replan." if closed
            else "No incident is visible, so the baseline keeps the current routing scope."
        )

        solve = self.service.solve(
            SolveOptions(
                method=method,
                particles=particles,
                evaluations=evaluations,
                seed=seed,
                closed_edge_ids=closed,
            ),
            scenario_id=scenario_id,
            include_route_plan=True,
        )
        route_plan = solve.pop("route_plan_object", None)
        evaluation_object = solve.pop("evaluation_object", None)
        notes.append(
            f"{solve['method']} returned status={solve['status']} in {solve['elapsed_s']:.2f}s."
        )
        if not solve["evaluation"]["feasible"] or route_plan is None:
            notes.append("Independent validator rejected the candidate; no plan was committed or dispatched.")
            return LoopState(
                scenario_id=scenario_id,
                closed_edge_ids=closed,
                state_version=state_version,
                route_version=route_version,
                forecast_mode="persistence",
                scope_action=action,
                committed=False,
                plan=solve["plan"],
                evaluation=solve["evaluation"],
                solve=solve,
                notes=notes,
            )

        executable = self._to_executable_plan(
            evaluation_object,
            scenario_id=scenario_id,
            state_version=state_version,
            route_version=route_version,
        )
        scenario = load_scenario(scenario_id)
        # Revalidate the exact logical plan against the same closed-road state
        # immediately before commit. This is the Step-10 stale-plan gate.
        checked = self.service.validate(
            scenario_id,
            solve["plan"],
            closed_edge_ids=closed,
        )
        if not checked["validation"]["feasible"]:
            notes.append("Pre-commit revalidation failed; candidate was discarded as stale/unsafe.")
            return LoopState(
                scenario_id=scenario_id,
                closed_edge_ids=closed,
                state_version=state_version,
                route_version=route_version,
                forecast_mode="persistence",
                scope_action=action,
                committed=False,
                plan=solve["plan"],
                evaluation=checked["validation"],
                solve=solve,
                notes=notes,
            )

        notes.append(
            f"Plan committed as {route_version} against {state_version}; execution prefix is empty at t=0.0s."
        )
        sumo_episode = None
        if execute_sumo:
            from src.sim.sumo_runner import run_episode
            output = output_dir or (PROJECT_ROOT / "artifacts" / "step10_complete_loop")
            output.mkdir(parents=True, exist_ok=True)
            progress = []
            episode = run_episode(
                scenario,
                executable,
                output,
                gui=False,
                on_step=lambda step: progress.append({
                    "t": step.sim_time_s,
                    "vehicles": len(step.vehicles),
                    "service_begin": list(step.service_begin_events),
                    "service_complete": list(step.service_complete_events),
                    "teleports": list(step.teleported_vehicle_ids),
                }),
            )
            sumo_episode = episode.to_dict()
            (output / "step10_loop_manifest.json").write_text(
                json.dumps({
                    "scenario_id": scenario_id,
                    "state_version": state_version,
                    "route_version": route_version,
                    "forecast_mode": "persistence",
                    "scope_action": action,
                    "committed": True,
                    "solve": solve,
                    "progress": progress,
                    "episode": sumo_episode,
                }, indent=2),
                encoding="utf-8",
            )
            notes.append(
                f"SUMO advanced the committed plan for {episode.total_steps} steps; "
                f"delivered={episode.delivered_count}, pending={episode.pending_count}, "
                f"teleports={episode.teleport_events}."
            )
        else:
            notes.append("SUMO execution was disabled by the caller; commit was completed but no simulator outcome was produced.")

        return LoopState(
            scenario_id=scenario_id,
            closed_edge_ids=closed,
            state_version=state_version,
            route_version=route_version,
            forecast_mode="persistence",
            scope_action=action,
            committed=True,
            plan=solve["plan"],
            evaluation=checked["validation"],
            solve=solve,
            sumo_episode=sumo_episode,
            notes=notes,
        )

    @staticmethod
    def _to_executable_plan(
        evaluation,
        *,
        scenario_id: str,
        state_version: str,
        route_version: str,
    ) -> ExecutableRoutePlan:
        """Bridge evaluated logical routes to the SUMO execution contract.

        The optimizer decides customer order. The independent evaluator
        supplies the physical directed legs and their timing. This function
        only packages those already-validated results for SUMO.
        """
        if evaluation is None or not evaluation.feasible:
            raise ValueError("Only a feasible independent evaluation can be dispatched")

        vehicles = []
        for vehicle_eval in evaluation.vehicle_evaluations:
            physical = vehicle_eval.physical_route
            if physical is None:
                raise ValueError(
                    f"Vehicle {vehicle_eval.vehicle_id!r} has no physical route"
                )

            stops = list(vehicle_eval.stops)
            legs = []
            departure = 0.0
            for index, leg in enumerate(physical.legs):
                if index < len(stops):
                    stop = stops[index]
                    arrival = float(stop.arrival_time_s)
                    departure = max(0.0, arrival - float(leg.travel_time_s))
                    from_stop = leg.from_node
                    to_stop = leg.to_node
                    service_departure = float(stop.service_end_time_s)
                else:
                    from_stop = leg.from_node
                    to_stop = leg.to_node
                    arrival = departure + float(leg.travel_time_s)
                    service_departure = arrival

                legs.append(
                    StopLeg(
                        from_stop_id=str(from_stop),
                        to_stop_id=str(to_stop),
                        physical_edge_ids=tuple(leg.edge_ids),
                        travel_time_s=float(leg.travel_time_s),
                        distance_m=float(leg.distance_m),
                        congestion_exposure=float(leg.congestion_delay_s),
                        departure_time_s=float(departure),
                        arrival_time_s=float(arrival),
                    )
                )
                departure = service_departure

            vehicles.append(
                VehicleRoute(
                    vehicle_id=str(vehicle_eval.vehicle_id),
                    customer_order=tuple(vehicle_eval.customer_ids),
                    legs=tuple(legs),
                    expected_departure_s=0.0,
                    expected_return_s=float(vehicle_eval.elapsed_time_s),
                    expected_driving_time_s=float(vehicle_eval.total_travel_time_s),
                    expected_waiting_time_s=float(vehicle_eval.total_waiting_time_s),
                    expected_service_time_s=float(vehicle_eval.total_service_time_s),
                    served_request_ids=tuple(vehicle_eval.customer_ids),
                    frozen_prefix_edge_ids=(),
                )
            )

        return ExecutableRoutePlan(
            scenario_id=scenario_id,
            state_version=state_version,
            route_version=route_version,
            generated_at_s=0.0,
            vehicle_routes=tuple(vehicles),
            objective_value=float(evaluation.objective_value),
            objective_time=float(evaluation.total_elapsed_time_s),
            objective_distance=float(evaluation.total_distance_m),
            objective_congestion=float(sum(
                leg.congestion_exposure
                for vehicle in vehicles
                for leg in vehicle.legs
            )),
            route_change_penalty=0.0,
            changed_vehicle_ids=tuple(vehicle.vehicle_id for vehicle in vehicles),
            generated_by="step10_qpso_persistence_rule",
        )

    def as_dict(self, state: LoopState) -> dict:
        return {
            "scenario_id": state.scenario_id,
            "closed_edge_ids": list(state.closed_edge_ids),
            "state_version": state.state_version,
            "route_version": state.route_version,
            "forecast_mode": state.forecast_mode,
            "scope_action": state.scope_action,
            "committed": state.committed,
            "plan": state.plan,
            "evaluation": state.evaluation,
            "solve": state.solve,
            "sumo_episode": state.sumo_episode,
            "notes": state.notes,
        }
