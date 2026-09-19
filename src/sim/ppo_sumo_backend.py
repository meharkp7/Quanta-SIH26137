"""
Step 16 — SUMO/TraCI backend for the PPO environment.

This adapter is the execution bridge between ``TrafficRoutingPPOEnv`` and
the existing Step-6 SUMO stack.  It deliberately reuses:

    SumoExporter -> RouteBuilder -> TraciAdapter -> SimState
                         + IncidentManager

The PPO environment remains responsible for reward semantics and decision
semantics; this class is responsible for simulator lifecycle, measured state,
and safe live route application.
"""

from __future__ import annotations

import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.sim.incidents import IncidentConfig, IncidentManager, STEP6_CLOSURE_CONFIG
from src.sim.route_builder import RouteBuilder
from src.sim.sim_state import SimState
from src.sim.sumo_exporter import SumoExporter
from src.sim.traci_adaptor import SimStepOutput, TraciAdapter

logger = logging.getLogger(__name__)


class PPOSumoSimulatorBackend:
    """
    Concrete simulator backend expected by ``TrafficRoutingPPOEnv``.

    The backend advances SUMO by a requested control interval, while
    internally stepping TraCI at ``step_length_s``.  It exposes only
    measurements that have already been realized by the simulator through
    ``latest_state``.

    Live replanning is conservative:
      * all candidate delivery routes are materialized and preflighted first;
      * a vehicle must still be active and its current SUMO edge must occur in
        the candidate route;
      * vehicles currently inside a programmed stop are not modified;
      * TraCI route/stop mutation APIs must be available;
      * failure leaves the incumbent plan as the authoritative plan.
    """

    def __init__(
        self,
        scenario: Any,
        initial_plan: Any,
        *,
        output_dir: Path,
        end_time_s: float = 600.0,
        step_length_s: float = 1.0,
        random_seed: int = 26137,
        gui: bool = False,
        incident_configs: Sequence[IncidentConfig] | None = None,
        background_duration_s: float = 0.0,
        background_interval_s: float = 60.0,
        background_start_s: float = 0.0,
        background_target_edges: Sequence[str] | None = None,
        background_blocked_intervals: Sequence[tuple[float, float]] | None = None,
        background_seed: int | None = None,
        use_subscriptions: bool = True,
    ) -> None:
        if end_time_s <= 0:
            raise ValueError("end_time_s must be positive")
        if step_length_s <= 0:
            raise ValueError("step_length_s must be positive")
        if end_time_s < step_length_s:
            raise ValueError("end_time_s must be >= step_length_s")

        self.scenario = scenario
        self.initial_plan = initial_plan
        self.current_plan = initial_plan
        self.output_dir = Path(output_dir).resolve()
        self.end_time_s = float(end_time_s)
        self.step_length_s = float(step_length_s)
        self.random_seed = int(random_seed)
        self.gui = bool(gui)
        self.use_subscriptions = bool(use_subscriptions)

        self._incident_configs = tuple(
            incident_configs
            if incident_configs is not None
            else (STEP6_CLOSURE_CONFIG,)
        )

        self._background_duration_s = float(background_duration_s)
        self._background_interval_s = float(background_interval_s)
        self._background_start_s = float(background_start_s)
        self._background_target_edges = tuple(background_target_edges or ())
        self._background_blocked_intervals = tuple(
            (float(start), float(end))
            for start, end in (background_blocked_intervals or ())
            if float(end) > float(start)
        )
        self._background_seed = (
            self.random_seed
            if background_seed is None
            else int(background_seed)
        )

        self._exporter: SumoExporter | None = None
        self._adapter: TraciAdapter | None = None
        self._sim_state: SimState | None = None
        self._incident_manager: IncidentManager | None = None
        self._edge_mapping: dict[str, str] = {}
        self._node_mapping: dict[str, str] = {}
        # Vehicles represented in SimState. SUMO may also contain background
        # traffic; those vehicles are intentionally outside delivery-state
        # bookkeeping.
        self._controlled_vehicle_ids: frozenset[str] = frozenset()

        self._sim_time_s = 0.0
        self._latest_state: dict[str, Any] = {}
        self._last_step: SimStepOutput | None = None
        self._episode_started = False

        self._total_teleports = 0
        self._total_failed = 0
        self._last_interval_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # SimulatorBackend contract
    # ------------------------------------------------------------------

    def reset(self, scenario: Any, route_plan: Any) -> None:
        """Start a fresh SUMO episode for ``scenario`` and ``route_plan``."""
        self.close()

        self.scenario = scenario
        self.initial_plan = route_plan
        self.current_plan = route_plan
        self.output_dir.mkdir(parents=True, exist_ok=True)

        network_dir = self.output_dir / "network"
        routes_dir = self.output_dir / "routes"

        self._exporter = SumoExporter(
            scenario,
            output_dir=network_dir,
        )
        net_path = self._exporter.export()

        self._edge_mapping = dict(self._exporter.edge_mapping)
        self._node_mapping = dict(self._exporter.node_mapping)

        route_builder = RouteBuilder(
            scenario,
            route_plan,
            {
                "edge_mapping": self._edge_mapping,
                "node_mapping": self._node_mapping,
            },
            background_duration_s=self._background_duration_s,
            background_interval_s=self._background_interval_s,
            background_start_s=self._background_start_s,
            background_target_edges=list(self._background_target_edges),
            background_blocked_intervals=list(
                self._background_blocked_intervals
            ),
            background_seed=self._background_seed,
        )
        rou_path = route_builder.build(routes_dir)

        stop_map = self._build_stop_request_map(route_plan)
        self._controlled_vehicle_ids = frozenset(stop_map)

        self._sim_state = SimState(scenario)
        self._sim_state.load_route_cargo(route_plan)
        self._sync_scenario_runtime_state()

        self._incident_manager = IncidentManager(
            configs=list(self._incident_configs),
            edge_mapping=self._edge_mapping,
        )

        self._adapter = TraciAdapter(
            net_path=net_path,
            rou_path=rou_path,
            output_dir=self.output_dir / "sumo_output",
            edge_ids=[edge.edge_id for edge in scenario.edges],
            stop_to_request_map=stop_map,
            end_time_s=self.end_time_s,
            step_length_s=self.step_length_s,
            gui=self.gui,
            random_seed=self.random_seed,
            use_subscriptions=self.use_subscriptions,
            service_earliest={
                request.request_id: max(
                    float(request.earliest_service_start_s),
                    float(request.release_s),
                )
                for request in scenario.requests
            },
        )

        self._adapter.__enter__()
        self._episode_started = True
        self._sim_time_s = float(self._adapter._traci.simulation.getTime())

        self._total_teleports = 0
        self._total_failed = 0
        self._last_interval_metrics = {}
        self._latest_state = self._build_latest_state(
            samples=(),
            failed_delta=0,
            incident_events=(),
        )

    def advance(self, duration_s: float) -> Mapping[str, Any]:
        """
        Advance SUMO by ``duration_s`` and return only realized measurements.

        The requested duration is converted to a number of TraCI steps.  A
        final partial step is rejected rather than silently changing the
        simulator clock.
        """
        if not self._episode_started or self._adapter is None:
            raise RuntimeError("advance() called before reset()")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")

        steps_float = float(duration_s) / self.step_length_s
        steps = int(round(steps_float))
        if not np.isclose(steps_float, steps, atol=1e-9):
            raise ValueError(
                "duration_s must be an integer multiple of step_length_s; "
                f"duration_s={duration_s}, step_length_s={self.step_length_s}"
            )

        start_failed = (
            self._sim_state.failed_count()
            if self._sim_state is not None
            else 0
        )
        outputs: list[SimStepOutput] = []
        incident_events: list[tuple[str, str]] = []

        for _ in range(steps):
            if self._adapter.done:
                break

            next_time_s = (
                float(self._adapter._traci.simulation.getTime())
                + self.step_length_s
            )

            if self._incident_manager is not None:
                incident_events.extend(
                    self._incident_manager.step(
                        self._adapter._traci,
                        next_time_s,
                    )
                )

            step = self._adapter.step()
            outputs.append(step)
            self._last_step = step
            self._sim_time_s = float(step.sim_time_s)

            self._update_sim_state(step)
            self._mark_expired_requests()

            self._total_teleports += len(step.teleported_vehicle_ids)
            if step.teleported_vehicle_ids:
                raise RuntimeError(
                    "SUMO teleportation detected for PPO episode: "
                    f"{step.teleported_vehicle_ids}"
                )

        failed_now = (
            self._sim_state.failed_count()
            if self._sim_state is not None
            else 0
        )
        failed_delta = max(0, failed_now - start_failed)

        self._latest_state = self._build_latest_state(
            samples=outputs,
            failed_delta=failed_delta,
            incident_events=incident_events,
        )

        self._last_interval_metrics = {
            "failed_delta": float(failed_delta),
            "teleports": float(
                sum(len(x.teleported_vehicle_ids) for x in outputs)
            ),
            "sim_steps": float(len(outputs)),
        }

        return {
            "incremental_operating_cost": 0.0,
            "congestion_exposure": float(
                self._latest_state["congestion_exposure"]
            ),
            "service_failures": float(failed_delta),
            "remaining_work": float(
                self._latest_state["remaining_work"]
            ),
            "edge_observations": tuple(
                self._latest_state["edge_observations"]
            ),
            "vehicle_loads": tuple(
                self._latest_state["vehicle_loads"]
            ),
            "incident_events": tuple(incident_events),
            "teleports": int(
                sum(len(x.teleported_vehicle_ids) for x in outputs)
            ),
        }

    @property
    def sim_time_s(self) -> float:
        return float(self._sim_time_s)

    @property
    def done(self) -> bool:
        if self._adapter is None:
            return False
        return bool(self._adapter.done)

    @property
    def latest_state(self) -> Mapping[str, Any]:
        return self._latest_state

    @property
    def closed_edge_ids(self) -> tuple[str, ...]:
        """Currently closed contract edge IDs visible to the planner."""
        if self._incident_manager is None:
            return ()
        return tuple(self._incident_manager.closed_edge_ids())

    # ------------------------------------------------------------------
    # Live replanning
    # ------------------------------------------------------------------

    def apply_route_plan(
        self,
        route_plan: Any,
        *,
        commitments: Any = None,
        planning_time_s: float | None = None,
    ) -> bool:
        """
        Apply a candidate logical route plan to the running SUMO instance.

        This is intentionally conservative.  If a safe live mutation cannot
        be proven, ``False`` is returned and the caller must retain the
        incumbent plan.

        `commitments` and `planning_time_s` matter: a mid-episode candidate
        must be re-validated as of *now* (some onboard customers may
        already be delivered), not as if the vehicle were starting fresh
        from the depot. Omitting them here previously caused
        `RouteBuilder` to double-count already-completed deliveries
        against vehicle capacity, spuriously rejecting valid candidates.
        Defaults to the simulator's own current sim time when
        `planning_time_s` is not given, since that is almost always what a
        caller wants for a live mutation.
        """
        if not self._episode_started or self._adapter is None:
            raise RuntimeError(
                "apply_route_plan() called before reset()"
            )

        if self._sim_state is None:
            raise RuntimeError("simulation state is unavailable")

        if planning_time_s is None:
            planning_time_s = self._sim_time_s

        traci = self._adapter._traci
        if traci is None:
            raise RuntimeError("TraCI connection is unavailable")

        if not hasattr(traci.vehicle, "setRoute"):
            logger.warning("TraCI vehicle.setRoute is unavailable")
            return False

        route_dir = self.output_dir / "replans" / f"t_{int(self._sim_time_s)}"
        if route_dir.exists():
            shutil.rmtree(route_dir)
        route_dir.mkdir(parents=True, exist_ok=True)

        try:
            route_xml = RouteBuilder(
                self.scenario,
                route_plan,
                {
                    "edge_mapping": self._edge_mapping,
                    "node_mapping": self._node_mapping,
                },
                planning_time_s=planning_time_s,
                commitments=commitments,
            ).build(route_dir)

            candidate_routes, candidate_stops = self._read_delivery_routes(
                route_xml
            )
        except Exception as exc:
            # exc_info=True: the previous bare exception message hid the
            # real failure site for non-ValueError bugs (e.g. an
            # AttributeError from an unrelated code path), making a
            # second, different failure indistinguishable from this one
            # in the logs. Always log the full traceback here.
            logger.warning(
                "Candidate route plan could not be materialized: %s",
                exc,
                exc_info=True,
            )
            return False

        current_snapshots = {
            item.vehicle_id: item
            for item in self._sim_state.snapshot_vehicles()
        }

        # Preflight every changed vehicle before mutating SUMO.
        mutations: list[tuple[str, list[str], list[dict[str, Any]]]] = []

        for vehicle_id, new_edges in candidate_routes.items():
            if vehicle_id not in current_snapshots:
                return False

            if not new_edges:
                return False

            try:
                active_ids = set(traci.vehicle.getIDList())
            except Exception:
                return False

            if vehicle_id not in active_ids:
                # A vehicle that has already arrived cannot be replanned.
                continue

            try:
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                stop_state = int(traci.vehicle.getStopState(vehicle_id))
            except Exception:
                return False

            # Never interrupt an active programmed customer service stop.
            if stop_state & 1:
                continue

            current_edge = current_edge or ""
            if current_edge.startswith(":"):
                # SUMO internal junction edges are transient.  The next normal
                # route edge is the stable alignment point.
                current_route = list(traci.vehicle.getRoute(vehicle_id))
                route_index = int(traci.vehicle.getRouteIndex(vehicle_id))
                stable_current = next(
                    (
                        edge
                        for edge in current_route[route_index:]
                        if not str(edge).startswith(":")
                    ),
                    None,
                )
                if stable_current is None:
                    return False
                current_edge = stable_current

            try:
                start_index = new_edges.index(current_edge)
            except ValueError:
                logger.info(
                    "Rejecting live replan for vehicle %r: candidate route "
                    "does not contain current edge %r",
                    vehicle_id,
                    current_edge,
                )
                return False

            suffix = new_edges[start_index:]
            if not suffix or suffix[0] != current_edge:
                return False

            mutations.append(
                (
                    vehicle_id,
                    suffix,
                    candidate_stops.get(vehicle_id, []),
                )
            )

        if not mutations:
            return True

        # Stop mutation is version-sensitive in TraCI. traci==1.27.1 (this
        # repo's pinned version) does not expose removeStop at all, so the
        # old check here (removeStop + setStop) was permanently False and
        # every live replan silently fell back to the incumbent route,
        # regardless of whether the candidate was otherwise valid. Confirmed
        # directly against the installed traci package. replaceStop +
        # insertStop are this version's real equivalent (see
        # _replace_future_stops) and are what is actually used below.
        if not (
            hasattr(traci.vehicle, "replaceStop")
            and hasattr(traci.vehicle, "insertStop")
        ):
            logger.warning(
                "Live replanning requires TraCI vehicle.replaceStop/"
                "insertStop; incumbent route retained"
            )
            return False

        old_routes: dict[str, list[str]] = {}
        applied: list[str] = []

        try:
            for vehicle_id, suffix, _ in mutations:
                # Capture both the route and the vehicle's index into it, so
                # rollback can restore only the remaining portion -- the
                # forward mutation above already does this alignment
                # (start_index/suffix); restoring the *full* historical
                # route unaligned to the vehicle's current position is the
                # same class of bug the pos=-1 fix above addresses, and can
                # itself raise "No connection between edge X and edge Y"
                # during rollback.
                full_route = list(traci.vehicle.getRoute(vehicle_id))
                try:
                    route_index = int(traci.vehicle.getRouteIndex(vehicle_id))
                except Exception:
                    route_index = 0
                old_routes[vehicle_id] = full_route[max(route_index, 0):] or full_route
                traci.vehicle.setRoute(vehicle_id, suffix)
                applied.append(vehicle_id)

            for vehicle_id, _, stops in mutations:
                self._replace_future_stops(
                    traci,
                    vehicle_id,
                    stops,
                )
        except Exception as exc:
            logger.exception(
                "Live route mutation failed; attempting route rollback"
            )
            for vehicle_id in applied:
                old_route = old_routes.get(vehicle_id)
                if old_route:
                    try:
                        traci.vehicle.setRoute(
                            vehicle_id,
                            old_route,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to rollback route for vehicle %r",
                            vehicle_id,
                        )
            return False

        self.current_plan = route_plan
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the current SUMO connection, if one is active."""
        if self._adapter is not None:
            try:
                self._adapter.__exit__(None, None, None)
            except Exception:
                logger.exception("Error closing PPO SUMO backend")
        self._adapter = None
        self._episode_started = False
        self._sim_state = None
        self._incident_manager = None
        self._last_step = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _build_stop_request_map(
        route_plan: Any,
    ) -> dict[str, list[tuple[int, str]]]:
        mapping: dict[str, list[tuple[int, str]]] = {}
        for vehicle_route in route_plan.vehicle_routes:
            customer_ids = getattr(
                vehicle_route,
                "customer_order",
                getattr(vehicle_route, "customer_ids", ()),
            )
            mapping[str(vehicle_route.vehicle_id)] = [
                (index, str(request_id))
                for index, request_id in enumerate(customer_ids)
            ]
        return mapping

    def _sync_scenario_runtime_state(self) -> None:
        """Publish the authoritative SimState vehicle snapshot to Scenario.

        Step-7/QPSO receives ``scenario`` directly, while SUMO state lives in
        ``SimState``.  Replanning must therefore see the same runtime vehicle
        position that the live simulator sees.
        """
        if self._sim_state is None or self.scenario is None:
            return

        snapshots = {
            str(item.vehicle_id): item
            for item in self._sim_state.snapshot_vehicles()
        }

        # Scenario/Vehicle contracts are frozen Pydantic models.  Never mutate
        # fleet members in place; publish a new immutable fleet snapshot instead.
        # This keeps the planner's Scenario view synchronized with SUMO without
        # violating the contract's immutability guarantees.
        updated_fleet = []
        changed = False

        for vehicle in self.scenario.fleet:
            snapshot = snapshots.get(str(vehicle.vehicle_id))
            if snapshot is None:
                updated_fleet.append(vehicle)
                continue

            updated = vehicle.model_copy(update={
                "current_edge_id": snapshot.current_edge_id,
                "current_node_id": snapshot.current_node_id,
                "distance_remaining_m": max(
                    0.0, float(snapshot.distance_remaining_m)
                ),
                "onboard_request_ids": tuple(snapshot.onboard_request_ids),
                "remaining_load": max(
                    0.0, float(snapshot.remaining_load)
                ),
                "executed_prefix_edge_ids": tuple(
                    snapshot.executed_prefix_edge_ids
                ),
            })
            updated_fleet.append(updated)
            changed = changed or updated != vehicle

        if changed:
            self.scenario = self.scenario.model_copy(
                update={"fleet": tuple(updated_fleet)}
            )

    def _update_sim_state(self, step: SimStepOutput) -> None:
        if self._sim_state is None:
            return

        controlled_vehicle_ids = self._controlled_vehicle_ids

        for vehicle in step.vehicles:
            # TraciAdapter reports both controlled delivery vehicles and any
            # SUMO background traffic. Only the former belong to SimState.
            if vehicle.vehicle_id not in controlled_vehicle_ids:
                continue
            self._sim_state.update_vehicle_position(
                vehicle.vehicle_id,
                current_edge_id=vehicle.edge_id,
                # SimState derives the downstream graph node from the
                # physical SUMO edge when TraCI does not expose one.
                current_node_id=None,
                distance_remaining_m=max(
                    0.0,
                    float(vehicle.distance_remaining_m),
                ),
                sim_time_s=float(step.sim_time_s),
            )

        self._sync_scenario_runtime_state()

        for vehicle_id, request_id in step.service_begin_events:
            if vehicle_id not in controlled_vehicle_ids:
                continue
            self._sim_state.begin_service(
                vehicle_id,
                request_id,
                float(step.sim_time_s),
            )

        for vehicle_id, request_id in step.service_complete_events:
            if vehicle_id not in controlled_vehicle_ids:
                continue
            self._sim_state.complete_service(
                vehicle_id,
                request_id,
                float(step.sim_time_s),
            )

        for vehicle_id in step.arrived_vehicle_ids:
            if vehicle_id not in controlled_vehicle_ids:
                continue
            self._sim_state.mark_vehicle_finished(vehicle_id)

    def _mark_expired_requests(self) -> None:
        if self._sim_state is None:
            return

        request_map = {
            request.request_id: request
            for request in self.scenario.requests
        }

        for snapshot in self._sim_state.snapshot_requests():
            if str(snapshot.state.value) != "pending":
                continue

            request = request_map.get(snapshot.request_id)
            if request is None:
                continue

            if (
                self._sim_time_s
                > float(request.latest_service_start_s)
                + float(request.service_duration_s)
            ):
                self._sim_state.mark_failed(
                    snapshot.request_id,
                    reason=(
                        f"service window expired at "
                        f"t={self._sim_time_s:.1f}s"
                    ),
                    sim_time_s=self._sim_time_s,
                )

        self._sync_scenario_runtime_state()

    def _build_latest_state(
        self,
        *,
        samples: Sequence[SimStepOutput],
        failed_delta: int,
        incident_events: Sequence[tuple[str, str]],
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "vehicle_loads": (),
            "vehicle_capacities": tuple(
                float(vehicle.capacity)
                for vehicle in self.scenario.fleet
            ),
            "vehicle_current_edge_ids": (),
            "vehicle_current_node_ids": (),
            "vehicle_distance_remaining_m": (),
            "vehicle_onboard_request_ids": (),
            "vehicle_executed_prefix_edge_ids": (),
            "deadline_slacks_s": (),
            "active_event_count": len(
                self._incident_manager.closed_edge_ids()
                if self._incident_manager is not None
                else ()
            ),
            "congestion_exposure": 0.0,
            "mean_speed_ratio": 1.0,
            "route_change_fraction": 0.0,
            "affected_route_fraction": 0.0,
            "remaining_work": float(
                self._sim_state.pending_count()
                if self._sim_state is not None
                else len(self.scenario.requests)
            ),
            "completed_work": float(
                self._sim_state.delivered_count()
                if self._sim_state is not None
                else 0
            ),
            "failed_work": float(
                self._sim_state.failed_count()
                if self._sim_state is not None
                else 0
            ),
            "service_failures": float(failed_delta),
            "edge_observations": [],
            "incident_events": tuple(incident_events),
        }

        if self._sim_state is not None:
            vehicles = self._sim_state.snapshot_vehicles()
            state["vehicle_loads"] = tuple(
                float(vehicle.remaining_load)
                for vehicle in vehicles
            )
            state["vehicle_current_edge_ids"] = tuple(
                vehicle.current_edge_id
                for vehicle in vehicles
            )
            state["vehicle_current_node_ids"] = tuple(
                vehicle.current_node_id
                for vehicle in vehicles
            )
            state["vehicle_distance_remaining_m"] = tuple(
                float(vehicle.distance_remaining_m)
                for vehicle in vehicles
            )
            state["vehicle_onboard_request_ids"] = tuple(
                vehicle.onboard_request_ids
                for vehicle in vehicles
            )
            state["vehicle_executed_prefix_edge_ids"] = tuple(
                vehicle.executed_prefix_edge_ids
                for vehicle in vehicles
            )

            request_map = {
                request.request_id: request
                for request in self.scenario.requests
            }
            slacks = []
            for request_snapshot in self._sim_state.snapshot_requests():
                if request_snapshot.state.value in {
                    "pending",
                    "onboard",
                }:
                    request = request_map.get(request_snapshot.request_id)
                    if request is not None:
                        slacks.append(
                            float(request.latest_service_start_s)
                            - self._sim_time_s
                        )
            state["deadline_slacks_s"] = tuple(slacks)

        edge_rows: dict[str, list[tuple[float, float, int, int]]] = {}
        for sample in samples:
            for edge in sample.edges:
                edge_rows.setdefault(edge.edge_id, []).append(
                    (
                        float(edge.mean_speed_mps),
                        float(edge.occupancy),
                        int(edge.vehicle_count),
                        int(edge.halting_count),
                    )
                )

        edge_observations = []
        speed_ratios = []
        occupancy_values = []
        halting_values = []

        closed_edges = set(
            self._incident_manager.closed_edge_ids()
            if self._incident_manager is not None
            else ()
        )

        for edge in self.scenario.edges:
            rows = edge_rows.get(edge.edge_id, [])
            if rows:
                speed = float(np.mean([row[0] for row in rows]))
                occupancy = float(np.mean([row[1] for row in rows]))
                vehicle_count = int(
                    round(np.mean([row[2] for row in rows]))
                )
                halting_count = int(
                    round(np.mean([row[3] for row in rows]))
                )
                speed_ratio = speed / float(edge.speed_limit_mps)
                speed_ratios.append(
                    float(np.clip(speed_ratio, 0.0, 2.0))
                )
                occupancy_values.append(
                    float(np.clip(occupancy, 0.0, 1.0))
                )
                halting_values.append(
                    max(0, halting_count)
                )
                edge_observations.append(
                    {
                        "edge_id": edge.edge_id,
                        "observed_speed_mps": speed
                        if edge.edge_id not in closed_edges
                        else None,
                        "observed_travel_time_s": (
                            float(edge.length_m) / speed
                            if (
                                edge.edge_id not in closed_edges
                                and speed > 0.05
                            )
                            else None
                        ),
                        "observation_age_s": 0.0,
                        "missing": False,
                        "known_closed": edge.edge_id in closed_edges,
                        "occupancy": occupancy,
                        "vehicle_count": vehicle_count,
                        "halting_count": halting_count,
                    }
                )
            else:
                edge_observations.append(
                    {
                        "edge_id": edge.edge_id,
                        "observed_speed_mps": None,
                        "observed_travel_time_s": None,
                        "observation_age_s": float(self._sim_time_s),
                        "missing": True,
                        "known_closed": edge.edge_id in closed_edges,
                        "occupancy": None,
                        "vehicle_count": 0,
                        "halting_count": None,
                    }
                )

        state["edge_observations"] = edge_observations

        state["mean_speed_ratio"] = (
            float(np.mean(speed_ratios))
            if speed_ratios
            else 1.0
        )

        if occupancy_values:
            state["congestion_exposure"] = float(
                np.mean(occupancy_values)
            )
        elif halting_values:
            state["congestion_exposure"] = float(
                np.mean(halting_values)
                / max(1.0, len(self.scenario.fleet))
            )

        return state

    @staticmethod
    def _read_delivery_routes(
        route_xml: Path,
    ) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
        tree = ET.parse(route_xml)
        root = tree.getroot()
        routes: dict[str, list[str]] = {}
        stops: dict[str, list[dict[str, Any]]] = {}

        for vehicle in root.findall("vehicle"):
            vehicle_id = vehicle.get("id")
            if not vehicle_id:
                continue

            route_el = vehicle.find("route")
            if route_el is None:
                continue

            edges = [
                edge
                for edge in (route_el.get("edges") or "").split()
                if edge
            ]
            routes[vehicle_id] = edges

            vehicle_stops = []
            for stop in vehicle.findall("stop"):
                edge = stop.get("edge")
                if not edge:
                    continue
                vehicle_stops.append(
                    {
                        "edge": edge,
                        "duration": float(stop.get("duration", "0")),
                        "until": float(stop.get("until", "0")),
                        "trip_id": stop.get("tripId"),
                    }
                )
            stops[vehicle_id] = vehicle_stops

        return routes, stops

    @staticmethod
    def _replace_future_stops(
        traci: Any,
        vehicle_id: str,
        stops: Sequence[Mapping[str, Any]],
    ) -> None:
        """
        Replace future programmed stops.

        traci==1.27.1 (this repo's pinned version) does not expose the
        older vehicle.removeStop API at all -- confirmed directly:
        hasattr(traci.vehicle, "removeStop") is False on this version, so
        the previous removeStop/setStop implementation could never
        actually run; every live replan silently fell back to the
        incumbent route. This version's real API instead exposes
        replaceStop (remove-without-replacement when edgeID="") and
        insertStop (add a brand new stop at a given index), which achieve
        the same effect: repeatedly remove the first remaining stop until
        SUMO reports none left, then insert the candidate stops in order.
        This is only called after the vehicle has been proven not to be
        in an active stop.
        """
        while True:
            try:
                remaining = traci.vehicle.getStops(vehicle_id, 0)
            except Exception:
                break
            if not remaining:
                break
            try:
                traci.vehicle.replaceStop(
                    vehicle_id,
                    0,
                    "",
                )
            except Exception:
                break

        for index, stop in enumerate(stops):
            edge = str(stop["edge"])
            # insertStop requires a real numeric stop-end position on the
            # lane; nothing upstream of this (route_builder.py only ever
            # wrote SUMO route-file XML, where SUMO computes a sensible
            # default itself) carries a position for a stop, and the
            # previous hardcoded pos=-1 is not a valid "auto" sentinel for
            # this API -- on a short edge it produces exactly the observed
            # "End position on lane must be after start position", because
            # SUMO derives startPos from pos and a nonsensical negative pos
            # can put startPos ahead of endPos. Query the edge's real
            # length and place the stop near its end, matching where SUMO
            # itself would place an unpositioned stop from route XML.
            try:
                lane_length = float(traci.lane.getLength(f"{edge}_0"))
            except Exception:
                lane_length = 0.0
            min_stop_length = 5.0
            if lane_length > min_stop_length:
                pos = lane_length - 0.1
            else:
                pos = max(lane_length, 0.1)

            traci.vehicle.insertStop(
                vehicle_id,
                index,
                edge,
                pos=pos,
                laneIndex=0,
                duration=float(stop.get("duration", 0.0)),
                until=float(stop.get("until", 0.0)),
            )