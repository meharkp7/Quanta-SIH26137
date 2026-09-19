"""
Application-side delivery state machine.

Maintains the canonical state of each request (pending → onboard → delivered)
and each vehicle's load/cargo, independently of SUMO's internal state.
State is updated ONLY on confirmed SUMO service events (via TraCI stop
detection), not on route arrival alone — preventing duplicate service.
Key Rule : "Mark a delivery complete only after the intended service event."
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RequestState(str, enum.Enum):
    PENDING = "pending"
    ONBOARD = "onboard"
    DELIVERED = "delivered"
    FAILED = "failed"          # window missed, vehicle lost, etc.


class VehicleSimState(str, enum.Enum):
    IDLE = "idle"
    EN_ROUTE = "en_route"
    STOPPED = "stopped"        # at a customer stop
    RETURNING = "returning"
    FINISHED = "finished"


# ---------------------------------------------------------------------------
# Immutable snapshots (returned to callers / logged)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RequestSnapshot:
    request_id: str
    state: RequestState
    assigned_vehicle_id: str | None
    service_start_s: float | None     # set when service begins
    service_end_s: float | None       # set when service completes
    was_late: bool                    # service started after latest_service_start_s


@dataclasses.dataclass(frozen=True)
class VehicleSnapshot:
    vehicle_id: str
    state: VehicleSimState
    current_edge_id: str | None
    current_node_id: str | None
    distance_remaining_m: float
    onboard_request_ids: tuple[str, ...]
    remaining_load: float
    delivered_count: int
    executed_prefix_edge_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Mutable internal records
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _RequestRecord:
    request_id: str
    demand: float
    earliest_service_start_s: float
    latest_service_start_s: float
    service_duration_s: float
    state: RequestState = RequestState.PENDING
    assigned_vehicle_id: str | None = None
    service_start_s: float | None = None
    service_end_s: float | None = None
    was_late: bool = False


@dataclasses.dataclass
class _VehicleRecord:
    vehicle_id: str
    capacity: float
    state: VehicleSimState = VehicleSimState.IDLE
    current_edge_id: str | None = None
    current_node_id: str | None = None
    distance_remaining_m: float = 0.0
    onboard_request_ids: list[str] = dataclasses.field(default_factory=list)
    remaining_load: float = 0.0
    delivered_count: int = 0
    executed_prefix_edge_ids: list[str] = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# SimState — the main class
# ---------------------------------------------------------------------------

class SimState:
    """
    Tracks delivery state for one simulation episode.

    Lifecycle
    ---------
    1. Instantiate from the scenario at episode start.
    2. Call ``update_vehicle_position`` on each TraCI step.
    3. Call ``begin_service`` when a vehicle's stop begins.
    4. Call ``complete_service`` when the stop duration elapses.
    5. Call ``mark_failed`` for any request that cannot be served.
    6. Read ``snapshot_requests`` / ``snapshot_vehicles`` for observations.

    Invariants
    ----------
    * A request transitions PENDING → ONBOARD only on ``begin_service``.
    * A request transitions ONBOARD → DELIVERED only on ``complete_service``.
    * No state transition is applied twice (idempotent guard).
    * ``remaining_load`` never exceeds ``capacity`` (guarded on onboard).
    """

    def __init__(self, scenario: Any) -> None:
        """
        Initialise from a Scenario contract.

        ``scenario`` is typed as ``Any`` to avoid a hard import cycle if this
        module is loaded before the contracts package is on sys.path; at
        runtime it is always a ``src.contracts.scenario.Scenario``.
        """
        self._scenario = scenario

        self._requests: dict[str, _RequestRecord] = {
            r.request_id: _RequestRecord(
                request_id=r.request_id,
                demand=float(r.demand),
                earliest_service_start_s=float(r.earliest_service_start_s),
                latest_service_start_s=float(r.latest_service_start_s),
                service_duration_s=float(r.service_duration_s),
            )
            for r in scenario.requests
        }
        self._vehicles: dict[str, _VehicleRecord] = {
            v.vehicle_id: _VehicleRecord(
                vehicle_id=v.vehicle_id,
                capacity=float(v.capacity),
            )
            for v in scenario.fleet
        }

    def load_route_cargo(self, route_plan) -> None:
        """Assign and load delivery cargo at the depot before departure."""
        seen = set()
        for route in route_plan.vehicle_routes:
            veh = self._vehicles[route.vehicle_id]
            request_ids = tuple(
                getattr(route, "customer_ids", ())
                or getattr(route, "customer_order", ())
            )
            load = sum(self._requests[rid].demand for rid in request_ids)
            if load > veh.capacity:
                raise ValueError(f"Vehicle {route.vehicle_id} exceeds capacity")
            for rid in request_ids:
                if rid in seen:
                    raise ValueError(f"Duplicate cargo assignment: {rid}")
                seen.add(rid)
                self._requests[rid].assigned_vehicle_id = route.vehicle_id
            veh.onboard_request_ids = list(request_ids)
            veh.remaining_load = load
        if seen != set(self._requests):
            raise ValueError("Every request must have a cargo owner")

    # ------------------------------------------------------------------
    # Position updates (called every TraCI step)
    # ------------------------------------------------------------------

    def update_vehicle_position(
        self,
        vehicle_id: str,
        *,
        current_edge_id: str | None,
        current_node_id: str | None,
        distance_remaining_m: float,
        sim_time_s: float,
    ) -> None:
        """Record the latest position reported by TraCI for a vehicle."""
        rec = self._vehicles.get(vehicle_id)
        if rec is None:
            logger.warning("update_vehicle_position: unknown vehicle %r", vehicle_id)
            return

        rec.current_edge_id = current_edge_id

        # TraCI reports the edge a vehicle is currently traversing.  The
        # routing/evaluation layer needs a graph node from which the mutable
        # continuation begins.  When TraCI does not provide an explicit node,
        # use the edge's downstream node rather than silently falling back to
        # the vehicle depot/start node.  This is the first valid decision
        # point after the vehicle completes its current physical edge.
        if current_node_id is None and current_edge_id is not None:
            edge = next(
                (
                    candidate
                    for candidate in self._scenario.edges
                    if str(candidate.edge_id) == str(current_edge_id)
                ),
                None,
            )
            if edge is not None:
                current_node_id = edge.to_node

        rec.current_node_id = current_node_id
        rec.distance_remaining_m = distance_remaining_m

        # Update executed prefix: append edge if it's new
        if (
            current_edge_id is not None
            and (
                not rec.executed_prefix_edge_ids
                or rec.executed_prefix_edge_ids[-1] != current_edge_id
            )
        ):
            rec.executed_prefix_edge_ids.append(current_edge_id)

        if rec.state == VehicleSimState.IDLE and current_edge_id is not None:
            rec.state = VehicleSimState.EN_ROUTE

    # ------------------------------------------------------------------
    # Service events (called by TraCI adapter on stop detection)
    # ------------------------------------------------------------------

    def begin_service(
        self,
        vehicle_id: str,
        request_id: str,
        sim_time_s: float,
    ) -> None:
        """
        Called when a vehicle starts its SUMO stop for request_id.

        Transitions the request to ONBOARD and notes service start time.
        Does nothing if the request is already being served (idempotent).
        """
        veh = self._vehicles.get(vehicle_id)
        req = self._requests.get(request_id)

        if veh is None:
            logger.error("begin_service: unknown vehicle %r", vehicle_id)
            return
        if req is None:
            logger.error("begin_service: unknown request %r", request_id)
            return
        if req.state != RequestState.PENDING:
            logger.warning(
                "begin_service: request %r already in state %s, ignoring",
                request_id, req.state,
            )
            return

        if req.assigned_vehicle_id not in (None, vehicle_id):
            raise ValueError("Service vehicle does not own this cargo")
        loaded = request_id in veh.onboard_request_ids

        # Guard: check capacity
        if not loaded and veh.remaining_load + req.demand > veh.capacity + 1e-9:
            logger.error(
                "begin_service: loading request %r (demand=%.1f) would exceed "
                "vehicle %r capacity %.1f (current load %.1f)",
                request_id, req.demand, vehicle_id, veh.capacity, veh.remaining_load,
            )
            return

        req.state = RequestState.ONBOARD
        req.assigned_vehicle_id = vehicle_id
        req.service_start_s = sim_time_s
        req.was_late = sim_time_s > req.latest_service_start_s + 1e-6

        if not loaded:
            veh.onboard_request_ids.append(request_id)
            veh.remaining_load += req.demand
        veh.state = VehicleSimState.STOPPED

        if req.was_late:
            logger.warning(
                "request %r service started LATE at %.1fs (latest=%.1fs)",
                request_id, sim_time_s, req.latest_service_start_s,
            )
        else:
            logger.info(
                "request %r service begun by vehicle %r at %.1fs",
                request_id, vehicle_id, sim_time_s,
            )

    def complete_service(
        self,
        vehicle_id: str,
        request_id: str,
        sim_time_s: float,
    ) -> None:
        """
        Called when the SUMO stop duration has elapsed.

        Transitions the request to DELIVERED and unloads cargo.
        Does nothing if the request is not currently ONBOARD (idempotent).
        """
        veh = self._vehicles.get(vehicle_id)
        req = self._requests.get(request_id)

        if veh is None:
            logger.error("complete_service: unknown vehicle %r", vehicle_id)
            return
        if req is None:
            logger.error("complete_service: unknown request %r", request_id)
            return
        if req.state != RequestState.ONBOARD:
            logger.warning(
                "complete_service: request %r not ONBOARD (state=%s), ignoring",
                request_id, req.state,
            )
            return

        if req.assigned_vehicle_id != vehicle_id:
            raise ValueError("Service vehicle does not own this cargo")
        if sim_time_s < req.service_start_s + req.service_duration_s - 1e-6:
            raise ValueError("Service duration has not elapsed")
        req.state = RequestState.DELIVERED
        req.service_end_s = sim_time_s

        if request_id in veh.onboard_request_ids:
            veh.onboard_request_ids.remove(request_id)
        veh.remaining_load = max(0.0, veh.remaining_load - req.demand)
        veh.delivered_count += 1

        if not veh.onboard_request_ids:
            veh.state = VehicleSimState.EN_ROUTE   # back on road

        logger.info(
            "request %r DELIVERED by vehicle %r at %.1fs",
            request_id, vehicle_id, sim_time_s,
        )

    def mark_failed(
        self,
        request_id: str,
        reason: str,
        sim_time_s: float,
    ) -> None:
        """Mark a request as failed (e.g., window missed, vehicle lost)."""
        req = self._requests.get(request_id)
        if req is None:
            logger.error("mark_failed: unknown request %r", request_id)
            return
        if req.state in (RequestState.DELIVERED, RequestState.FAILED):
            return
        req.state = RequestState.FAILED
        logger.warning(
            "request %r marked FAILED at %.1fs: %s", request_id, sim_time_s, reason
        )

    def mark_vehicle_finished(self, vehicle_id: str) -> None:
        """Mark a vehicle as having completed its route and returned to depot."""
        rec = self._vehicles.get(vehicle_id)
        if rec:
            rec.state = VehicleSimState.FINISHED

    # ------------------------------------------------------------------
    # Snapshots (read-only views for Observation / logging)
    # ------------------------------------------------------------------

    def snapshot_requests(self) -> list[RequestSnapshot]:
        return [
            RequestSnapshot(
                request_id=r.request_id,
                state=r.state,
                assigned_vehicle_id=r.assigned_vehicle_id,
                service_start_s=r.service_start_s,
                service_end_s=r.service_end_s,
                was_late=r.was_late,
            )
            for r in self._requests.values()
        ]

    def snapshot_vehicles(self) -> list[VehicleSnapshot]:
        return [
            VehicleSnapshot(
                vehicle_id=v.vehicle_id,
                state=v.state,
                current_edge_id=v.current_edge_id,
                current_node_id=v.current_node_id,
                distance_remaining_m=v.distance_remaining_m,
                onboard_request_ids=tuple(v.onboard_request_ids),
                remaining_load=v.remaining_load,
                delivered_count=v.delivered_count,
                executed_prefix_edge_ids=tuple(v.executed_prefix_edge_ids),
            )
            for v in self._vehicles.values()
        ]

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def all_delivered(self) -> bool:
        return all(
            r.state == RequestState.DELIVERED for r in self._requests.values()
        )

    def pending_count(self) -> int:
        return sum(
            1 for r in self._requests.values() if r.state in (RequestState.PENDING, RequestState.ONBOARD)
        )

    def delivered_count(self) -> int:
        return sum(
            1 for r in self._requests.values() if r.state == RequestState.DELIVERED
        )

    def failed_count(self) -> int:
        return sum(
            1 for r in self._requests.values() if r.state == RequestState.FAILED
        )

    def summary(self) -> dict:
        return {
            "pending": self.pending_count(),
            "delivered": self.delivered_count(),
            "failed": self.failed_count(),
            "vehicles": {
                v.vehicle_id: v.state.value
                for v in self._vehicles.values()
            },
        }
