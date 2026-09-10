"""Independent route evaluation for the Step 4 routing stack.

The route evaluator is the feasibility-truth layer of the routing system.

Responsibilities
----------------
- resolve logical customer sequences onto the directed road network;
- construct physical directed routes;
- perform time-dependent shortest-path selection;
- propagate vehicle arrival/service/departure times;
- enforce request release times;
- enforce service-start time windows;
- account for waiting;
- enforce vehicle capacity;
- enforce directed connectivity;
- enforce depot return;
- detect unknown vehicles;
- detect unknown, duplicate and unserved requests;
- expose detailed per-stop and per-vehicle diagnostics;
- expose objective components for constructive heuristics and optimizers;
- support dynamic travel-time providers;
- remain independent of QPSO, PSO, ALNS, DRL, GNN and Transformer code.

The evaluator does not optimize routes.

All optimization and learning components must treat this module as the
authoritative feasibility and objective-evaluation layer.

Contract semantics
------------------
The implementation follows the V1.2 scenario contract.

Request:
    request_id
    original_customer_id
    original_x
    original_y
    access_node_id
    access_distance_m
    demand
    known_at_s
    release_s
    earliest_service_start_s
    latest_service_start_s
    service_duration_s
    status

Vehicle:
    vehicle_id
    capacity
    start_node_id
    depot_node_id
    current_edge_id
    current_node_id
    distance_remaining_m
    onboard_request_ids
    remaining_load
    executed_prefix_edge_ids
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from src.contracts.core_types import (
    CustomerId,
    DemandUnits,
    DistanceM,
    RoadEdgeId,
    RoadNodeId,
    TimeS,
    VehicleId,
)
from src.contracts.scenario import RoadEdge
from src.routing.path_builder import (
    DirectedPathBuilder,
    PathNotFoundError,
    PathResult,
)
from src.routing.route_plan import RoutePlan, VehicleRoute
from src.routing.route_types import PhysicalRoute, RouteLeg


TravelTimeProvider = Callable[[RoadEdge, TimeS], TimeS]


@dataclass(frozen=True)
class RouteEvaluationConfig:
    """Configuration controlling route evaluation and objective weighting."""

    distance_weight: float = 1.0
    travel_time_weight: float = 1.0
    waiting_time_weight: float = 1.0
    service_time_weight: float = 0.0

    lateness_penalty: float = 10_000.0
    capacity_penalty: float = 10_000.0
    connectivity_penalty: float = 10_000.0
    duplicate_customer_penalty: float = 10_000.0
    unserved_customer_penalty: float = 10_000.0
    unknown_vehicle_penalty: float = 10_000.0

    allow_waiting: bool = True

    # V1.2 does not require a vehicle start-time field. This value therefore
    # provides the planning/simulation epoch unless the vehicle model later
    # exposes an explicit start_time_s field.
    default_start_time_s: TimeS = 0.0

    def __post_init__(self) -> None:
        numeric_fields = (
            "distance_weight",
            "travel_time_weight",
            "waiting_time_weight",
            "service_time_weight",
            "lateness_penalty",
            "capacity_penalty",
            "connectivity_penalty",
            "duplicate_customer_penalty",
            "unserved_customer_penalty",
            "unknown_vehicle_penalty",
            "default_start_time_s",
        )

        for field_name in numeric_fields:
            value = float(getattr(self, field_name))

            if not isfinite(value):
                raise ValueError(
                    f"{field_name} must be finite"
                )

            if field_name.endswith("_penalty") and value < 0.0:
                raise ValueError(
                    f"{field_name} must be non-negative"
                )

            if field_name.endswith("_weight") and value < 0.0:
                raise ValueError(
                    f"{field_name} must be non-negative"
                )


class VehicleEvaluationCollection(tuple):
    """Tuple-like collection supporting deterministic vehicle-ID lookup."""

    def __getitem__(self, key):
        if isinstance(key, str):
            for item in self:
                if str(item.vehicle_id) == key:
                    return item
            raise KeyError(key)
        return super().__getitem__(key)


@dataclass(frozen=True)
class StopEvaluation:
    """Detailed evaluation of one customer service stop."""

    customer_id: CustomerId
    node_id: RoadNodeId
    edge_ids: tuple[RoadEdgeId, ...]
    distance_m: DistanceM
    travel_time_s: TimeS

    arrival_time_s: TimeS
    waiting_time_s: TimeS
    service_start_time_s: TimeS
    service_end_time_s: TimeS

    demand_units: DemandUnits
    load_before_units: DemandUnits
    load_after_units: DemandUnits

    release_time_s: TimeS
    earliest_time_s: TimeS
    latest_time_s: TimeS

    release_feasible: bool
    time_window_feasible: bool

    @property
    def departure_time_s(self) -> TimeS:
        """Time at which service completion allows route continuation."""

        return self.service_end_time_s

    @property
    def lateness_s(self) -> TimeS:
        """Positive service-start lateness beyond the latest allowed time."""

        return max(
            0.0,
            float(self.service_start_time_s)
            - float(self.latest_time_s),
        )

    # Compatibility-style semantic aliases retained as properties so that
    # downstream routing/analysis code can use descriptive names without
    # duplicating stored state.

    @property
    def release_time(self) -> TimeS:
        return self.release_time_s

    @property
    def earliest_service_start_s(self) -> TimeS:
        return self.earliest_time_s

    @property
    def latest_service_start_s(self) -> TimeS:
        return self.latest_time_s

    @property
    def service_start_s(self) -> TimeS:
        return self.service_start_time_s

    @property
    def service_end_s(self) -> TimeS:
        return self.service_end_time_s


@dataclass(frozen=True)
class VehicleRouteEvaluation:
    """Complete evaluation of one vehicle's logical route."""

    vehicle_id: VehicleId

    physical_route: PhysicalRoute | None
    stops: tuple[StopEvaluation, ...]

    feasible: bool
    connectivity_feasible: bool
    capacity_feasible: bool
    time_window_feasible: bool

    capacity_units: DemandUnits
    final_load_units: DemandUnits
    maximum_load_units: DemandUnits

    total_distance_m: DistanceM
    total_travel_time_s: TimeS
    total_waiting_time_s: TimeS
    total_service_time_s: TimeS
    elapsed_time_s: TimeS
    lateness_s: TimeS

    errors: tuple[str, ...]

    @property
    def customer_ids(self) -> tuple[CustomerId, ...]:
        """Customers represented by the evaluated physical stops."""

        return tuple(
            stop.customer_id
            for stop in self.stops
        )

    @property
    def total_elapsed_time_s(self) -> TimeS:
        """Explicit alias for elapsed route duration."""

        return self.elapsed_time_s

    @property
    def total_lateness_s(self) -> TimeS:
        """Explicit alias for aggregate lateness."""

        return self.lateness_s

    @property
    def service_starts_s(self) -> Mapping[CustomerId, float]:
        return {stop.customer_id: float(stop.service_start_time_s) for stop in self.stops}

    @property
    def service_ends_s(self) -> Mapping[CustomerId, float]:
        return {stop.customer_id: float(stop.service_end_time_s) for stop in self.stops}

    @property
    def objective_components(self) -> Mapping[str, float]:
        """Raw objective components before configured weighting."""

        return {
            "distance_m": float(self.total_distance_m),
            "travel_time_s": float(self.total_travel_time_s),
            "waiting_time_s": float(self.total_waiting_time_s),
            "service_time_s": float(self.total_service_time_s),
            "elapsed_time_s": float(self.elapsed_time_s),
            "lateness_s": float(self.lateness_s),
        }


@dataclass(frozen=True)
class RoutePlanEvaluation:
    """Evaluation of a complete multi-vehicle logical route plan."""

    route_plan: RoutePlan
    vehicle_evaluations: tuple[VehicleRouteEvaluation, ...]

    feasible: bool

    all_requests_served: bool
    customer_uniqueness_feasible: bool

    served_customer_ids: tuple[CustomerId, ...]
    unserved_customer_ids: tuple[CustomerId, ...]
    duplicate_customer_ids: tuple[CustomerId, ...]
    unknown_request_ids: tuple[CustomerId, ...]
    unknown_vehicle_ids: tuple[VehicleId, ...]

    total_distance_m: DistanceM
    total_travel_time_s: TimeS
    total_waiting_time_s: TimeS
    total_service_time_s: TimeS
    total_elapsed_time_s: TimeS
    total_lateness_s: TimeS

    objective_value: float

    errors: tuple[str, ...]

    @property
    def assignment_feasible(self) -> bool:
        """Whether customer assignment itself is valid and complete."""

        return (
            self.all_requests_served
            and self.customer_uniqueness_feasible
            and not self.unknown_request_ids
            and not self.unknown_vehicle_ids
        )

    @property
    def capacity_feasible(self) -> bool:
        """Whether every vehicle route respects capacity."""

        return all(
            evaluation.capacity_feasible
            for evaluation in self.vehicle_evaluations
        )

    @property
    def connectivity_feasible(self) -> bool:
        """Whether every vehicle route is physically connected."""

        return all(
            evaluation.connectivity_feasible
            for evaluation in self.vehicle_evaluations
        )

    @property
    def time_window_feasible(self) -> bool:
        """Whether every evaluated service stop satisfies its window."""

        return all(
            evaluation.time_window_feasible
            for evaluation in self.vehicle_evaluations
        )

    @property
    def assigned_request_ids(self) -> tuple[CustomerId, ...]:
        return self.served_customer_ids

    @property
    def unassigned_request_ids(self) -> tuple[CustomerId, ...]:
        return self.unserved_customer_ids

    @property
    def duplicate_request_ids(self) -> tuple[CustomerId, ...]:
        return self.duplicate_customer_ids

    @property
    def total_travel_time(self) -> TimeS:
        """Semantic alias for total travel time."""

        return self.total_travel_time_s

    @property
    def total_waiting_time(self) -> TimeS:
        """Semantic alias for total waiting time."""

        return self.total_waiting_time_s

    @property
    def total_lateness(self) -> TimeS:
        """Semantic alias for total lateness."""

        return self.total_lateness_s

    @property
    def objective_components(self) -> Mapping[str, float]:
        """Raw objective components before penalty aggregation."""

        return {
            "distance_m": float(self.total_distance_m),
            "travel_time_s": float(self.total_travel_time_s),
            "waiting_time_s": float(self.total_waiting_time_s),
            "service_time_s": float(self.total_service_time_s),
            "elapsed_time_s": float(self.total_elapsed_time_s),
            "lateness_s": float(self.total_lateness_s),
        }


class RouteEvaluator:
    """Evaluate logical RoutePlans against a V1.2 Scenario.

    RouteEvaluator is deliberately independent of optimization algorithms.

    A route plan enters as customer assignments. The evaluator converts each
    logical customer transition into a directed physical path and propagates
    time and load through that path.

    When a dynamic TravelTimeProvider is supplied, path selection and route
    traversal both use the same time-dependent edge-cost semantics. This is
    important: a dynamic provider must not merely be applied after a static
    shortest path has already been selected.
    """

    def __init__(
        self,
        scenario: Any,
        *,
        config: RouteEvaluationConfig | None = None,
        travel_time_provider: TravelTimeProvider | None = None,
        closed_edge_ids: Iterable[RoadEdgeId] = (),
    ) -> None:
        self.scenario = scenario
        self.config = config or RouteEvaluationConfig()
        self.travel_time_provider = travel_time_provider

        self._closed_edge_ids = frozenset(
            closed_edge_ids
        )

        self._edges = tuple(
            self._scenario_edges()
        )

        self._requests = tuple(
            self._scenario_requests()
        )

        self._vehicles = tuple(
            self._scenario_vehicles()
        )

        self._request_by_id: dict[
            CustomerId,
            Any,
        ] = {
            self._request_id(request): request
            for request in self._requests
        }

        self._vehicle_by_id: dict[
            VehicleId,
            Any,
        ] = {
            self._vehicle_id(vehicle): vehicle
            for vehicle in self._vehicles
        }

        if len(self._request_by_id) != len(self._requests):
            raise ValueError(
                "Scenario contains duplicate request IDs"
            )

        if len(self._vehicle_by_id) != len(self._vehicles):
            raise ValueError(
                "Scenario contains duplicate vehicle IDs"
            )

        self.path_builder = DirectedPathBuilder(
            self._edges,
            closed_edge_ids=self._closed_edge_ids,
            travel_time_provider=self.travel_time_provider,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def evaluate(
        self,
        route_plan: RoutePlan,
        *,
        planning_time_s: TimeS | None = None,
    ) -> RoutePlanEvaluation:
        """Evaluate an entire multi-vehicle route plan.

        ``planning_time_s`` is the planning/simulation epoch. It does not
        mutate the Scenario or vehicle state. It is useful when evaluating
        dynamic snapshots at different points in a simulation.
        """

        if not isinstance(route_plan, RoutePlan):
            raise TypeError(
                "route_plan must be a RoutePlan"
            )

        evaluation_time = (
            self.config.default_start_time_s
            if planning_time_s is None
            else float(planning_time_s)
        )

        if not isfinite(float(evaluation_time)):
            raise ValueError(
                "planning_time_s must be finite"
            )

        vehicle_evaluations = VehicleEvaluationCollection(
            self._evaluate_vehicle_route(
                vehicle_route,
                planning_time_s=evaluation_time,
            )
            for vehicle_route in route_plan.vehicle_routes
        )

        served_customer_ids = (
            route_plan.unique_customer_ids()
        )

        duplicate_customer_ids = (
            route_plan.duplicate_customer_ids()
        )

        unknown_request_ids = tuple(
            dict.fromkeys(
                customer_id
                for customer_id in route_plan.all_customer_ids()
                if customer_id not in self._request_by_id
            )
        )

        required_customer_ids = tuple(
            self._request_by_id.keys()
        )

        served_set = set(
            served_customer_ids
        )

        unserved_customer_ids = tuple(
            customer_id
            for customer_id in required_customer_ids
            if customer_id not in served_set
        )

        unknown_vehicle_ids = tuple(
            sorted(
                (
                    vehicle_route.vehicle_id
                    for vehicle_route
                    in route_plan.vehicle_routes
                    if vehicle_route.vehicle_id
                    not in self._vehicle_by_id
                ),
                key=str,
            )
        )

        all_requests_served = (
            not unserved_customer_ids
        )

        customer_uniqueness_feasible = (
            not duplicate_customer_ids
        )

        errors: list[str] = []

        for vehicle_evaluation in vehicle_evaluations:
            errors.extend(
                vehicle_evaluation.errors
            )

        if unknown_vehicle_ids:
            errors.append(
                "Unknown vehicle assignments: "
                + ", ".join(
                    str(vehicle_id)
                    for vehicle_id in unknown_vehicle_ids
                )
            )

        if unknown_request_ids:
            errors.append(
                "Unknown request assignments: "
                + ", ".join(str(customer_id) for customer_id in unknown_request_ids)
            )

        if duplicate_customer_ids:
            errors.append(
                "Duplicate customer assignments: "
                + ", ".join(
                    str(customer_id)
                    for customer_id in duplicate_customer_ids
                )
            )

        if unserved_customer_ids:
            errors.append(
                "Unserved customers: "
                + ", ".join(
                    str(customer_id)
                    for customer_id in unserved_customer_ids
                )
            )

        total_distance = sum(
            evaluation.total_distance_m
            for evaluation in vehicle_evaluations
        )

        total_travel_time = sum(
            evaluation.total_travel_time_s
            for evaluation in vehicle_evaluations
        )

        total_waiting = sum(
            evaluation.total_waiting_time_s
            for evaluation in vehicle_evaluations
        )

        total_service = sum(
            evaluation.total_service_time_s
            for evaluation in vehicle_evaluations
        )

        total_elapsed = sum(
            evaluation.elapsed_time_s
            for evaluation in vehicle_evaluations
        )

        total_lateness = sum(
            evaluation.lateness_s
            for evaluation in vehicle_evaluations
        )

        objective = self._objective_value(
            distance_m=total_distance,
            travel_time_s=total_travel_time,
            waiting_time_s=total_waiting,
            service_time_s=total_service,
            lateness_s=total_lateness,
            vehicle_evaluations=vehicle_evaluations,
            duplicate_customer_count=(
                len(duplicate_customer_ids)
            ),
            unserved_customer_count=(
                len(unserved_customer_ids)
            ),
            unknown_vehicle_count=(
                len(unknown_vehicle_ids)
            ),
        )

        # ``feasible`` answers whether the supplied candidate itself is
        # internally legal. Completeness is exposed separately through
        # ``all_requests_served``/``assignment_feasible`` so partial plans
        # can be evaluated during construction, repair, and optimization.
        feasible = (
            all(
                evaluation.feasible
                for evaluation in vehicle_evaluations
            )
            and customer_uniqueness_feasible
            and not unknown_request_ids
            and not unknown_vehicle_ids
        )

        return RoutePlanEvaluation(
            route_plan=route_plan,
            vehicle_evaluations=vehicle_evaluations,
            feasible=feasible,
            all_requests_served=all_requests_served,
            customer_uniqueness_feasible=(
                customer_uniqueness_feasible
            ),
            served_customer_ids=served_customer_ids,
            unserved_customer_ids=(
                unserved_customer_ids
            ),
            duplicate_customer_ids=(
                duplicate_customer_ids
            ),
            unknown_request_ids=unknown_request_ids,
            unknown_vehicle_ids=unknown_vehicle_ids,
            total_distance_m=float(total_distance),
            total_travel_time_s=float(
                total_travel_time
            ),
            total_waiting_time_s=float(
                total_waiting
            ),
            total_service_time_s=float(
                total_service
            ),
            total_elapsed_time_s=float(
                total_elapsed
            ),
            total_lateness_s=float(
                total_lateness
            ),
            objective_value=float(objective),
            errors=tuple(errors),
        )

    def evaluate_vehicle_route(
        self,
        vehicle_route: VehicleRoute,
        *,
        planning_time_s: TimeS | None = None,
    ) -> VehicleRouteEvaluation:
        """Evaluate one vehicle route independently."""

        evaluation_time = (
            self.config.default_start_time_s
            if planning_time_s is None
            else float(planning_time_s)
        )

        return self._evaluate_vehicle_route(
            vehicle_route,
            planning_time_s=evaluation_time,
        )

    def with_network_state(
        self,
        *,
        closed_edge_ids: Iterable[RoadEdgeId] = (),
        travel_time_provider: TravelTimeProvider | None = None,
    ) -> RouteEvaluator:
        """Create a new evaluator for a network-state snapshot.

        The Scenario remains immutable. This method is therefore appropriate
        for simulation epochs where incidents or traffic conditions change.
        """

        provider = (
            self.travel_time_provider
            if travel_time_provider is None
            else travel_time_provider
        )

        return RouteEvaluator(
            self.scenario,
            config=self.config,
            travel_time_provider=provider,
            closed_edge_ids=closed_edge_ids,
        )

    # ==================================================================
    # Vehicle evaluation
    # ==================================================================

    def _evaluate_vehicle_route(
        self,
        vehicle_route: VehicleRoute,
        *,
        planning_time_s: TimeS,
    ) -> VehicleRouteEvaluation:
        vehicle = self._vehicle_by_id.get(
            vehicle_route.vehicle_id
        )

        if vehicle is None:
            return self._unknown_vehicle_evaluation(
                vehicle_route
            )

        start_node = self._vehicle_start_node(
            vehicle
        )

        end_node = self._vehicle_end_node(
            vehicle
        )

        capacity = float(
            self._vehicle_capacity(vehicle)
        )

        vehicle_start_time = float(
            planning_time_s
        )

        current_node = start_node
        current_time = vehicle_start_time

        current_load = 0.0
        maximum_load = 0.0

        total_distance = 0.0
        total_travel_time = 0.0
        total_waiting = 0.0
        total_service = 0.0
        total_lateness = 0.0

        physical_legs: list[RouteLeg] = []
        stop_results: list[StopEvaluation] = []

        connectivity_feasible = True
        capacity_feasible = True
        time_window_feasible = True

        errors: list[str] = []

        # --------------------------------------------------------------
        # Customer visits
        # --------------------------------------------------------------

        for customer_id in vehicle_route.customer_ids:
            request = self._request_by_id.get(
                customer_id
            )

            if request is None:
                errors.append(
                    f"Unknown customer_id={customer_id!r}"
                )

                connectivity_feasible = False
                time_window_feasible = False
                continue

            customer_node = self._request_node(
                request
            )

            # CRITICAL:
            # The departure time is passed into the path builder so a
            # time-dependent provider can influence PATH SELECTION itself.
            #
            # We must not select a free-flow path first and then merely
            # re-time that path afterwards.
            try:
                path = self.path_builder.shortest_path(
                    current_node,
                    customer_node,
                    departure_time_s=current_time,
                    travel_time_provider=(
                        self.travel_time_provider
                    ),
                )

            except PathNotFoundError as exc:
                connectivity_feasible = False

                errors.append(
                    f"Vehicle {vehicle_route.vehicle_id!r}: "
                    f"cannot reach customer {customer_id!r} "
                    f"from node {current_node!r}: {exc}"
                )

                continue

            except ValueError as exc:
                connectivity_feasible = False

                errors.append(
                    f"Vehicle {vehicle_route.vehicle_id!r}: "
                    f"invalid dynamic travel-time evaluation "
                    f"while reaching customer "
                    f"{customer_id!r}: {exc}"
                )

                continue

            physical_legs.append(
                self._path_to_leg(path)
            )

            distance = float(
                path.distance_m
            )

            # PathResult.travel_time_s is already the time-dependent
            # traversal cost selected by DirectedPathBuilder.
            travel_time = float(
                path.travel_time_s
            )

            if travel_time < 0.0 or not isfinite(
                travel_time
            ):
                connectivity_feasible = False

                errors.append(
                    f"Invalid travel time for transition "
                    f"{current_node!r} -> "
                    f"{customer_node!r}: "
                    f"{travel_time}"
                )

                continue

            arrival_time = (
                current_time
                + travel_time
            )

            release_time = float(
                self._request_release_time(request)
            )

            earliest = float(
                self._request_earliest_time(request)
            )

            latest = float(
                self._request_latest_time(request)
            )

            service_time = float(
                self._request_service_time(request)
            )

            demand = float(
                self._request_demand(request)
            )

            # ----------------------------------------------------------
            # Release time + service-start window
            # ----------------------------------------------------------

            minimum_service_start = max(
                release_time,
                earliest,
            )

            service_start = arrival_time
            waiting_time = 0.0

            if service_start < minimum_service_start:
                if self.config.allow_waiting:
                    waiting_time = (
                        minimum_service_start
                        - service_start
                    )

                    service_start = (
                        minimum_service_start
                    )

                else:
                    errors.append(
                        f"Customer {customer_id!r} requires "
                        f"{minimum_service_start - service_start} "
                        f"seconds of waiting, but waiting is disabled"
                    )

            service_end = (
                service_start
                + service_time
            )

            # ----------------------------------------------------------
            # Capacity
            # ----------------------------------------------------------

            load_before = current_load
            current_load += demand

            maximum_load = max(
                maximum_load,
                current_load,
            )

            stop_capacity_feasible = (
                current_load <= capacity
            )

            if not stop_capacity_feasible:
                capacity_feasible = False

                errors.append(
                    f"Vehicle {vehicle_route.vehicle_id!r} "
                    f"exceeds capacity at customer "
                    f"{customer_id!r}: "
                    f"{current_load} > {capacity}"
                )

            # ----------------------------------------------------------
            # Release/window validation
            # ----------------------------------------------------------

            release_feasible = (
                service_start >= release_time
            )

            window_feasible = (
                earliest
                <= service_start
                <= latest
            )

            if not release_feasible:
                time_window_feasible = False

                errors.append(
                    f"Customer {customer_id!r} service starts "
                    f"before release time {release_time}: "
                    f"{service_start}"
                )

            lateness = max(
                0.0,
                service_start - latest,
            )

            if not window_feasible:
                time_window_feasible = False
                total_lateness += lateness

                errors.append(
                    f"Customer {customer_id!r} service starts "
                    f"outside time window "
                    f"[{earliest}, {latest}]: "
                    f"{service_start}"
                )

            stop_results.append(
                StopEvaluation(
                    customer_id=customer_id,
                    node_id=customer_node,
                    edge_ids=tuple(path.edge_ids),
                    distance_m=float(distance),
                    travel_time_s=float(travel_time),
                    arrival_time_s=float(
                        arrival_time
                    ),
                    waiting_time_s=float(
                        waiting_time
                    ),
                    service_start_time_s=float(
                        service_start
                    ),
                    service_end_time_s=float(
                        service_end
                    ),
                    demand_units=float(
                        demand
                    ),
                    load_before_units=float(
                        load_before
                    ),
                    load_after_units=float(
                        current_load
                    ),
                    release_time_s=float(
                        release_time
                    ),
                    earliest_time_s=float(
                        earliest
                    ),
                    latest_time_s=float(
                        latest
                    ),
                    release_feasible=(
                        release_feasible
                    ),
                    time_window_feasible=(
                        window_feasible
                    ),
                )
            )

            total_distance += distance
            total_travel_time += travel_time
            total_waiting += waiting_time
            total_service += service_time

            current_time = service_end
            current_node = customer_node

        # --------------------------------------------------------------
        # Depot return
        # --------------------------------------------------------------

        try:
            return_path = self.path_builder.shortest_path(
                current_node,
                end_node,
                departure_time_s=current_time,
                travel_time_provider=(
                    self.travel_time_provider
                ),
            )

            physical_legs.append(
                self._path_to_leg(
                    return_path
                )
            )

            return_distance = float(
                return_path.distance_m
            )

            return_travel_time = float(
                return_path.travel_time_s
            )

            if (
                return_travel_time < 0.0
                or not isfinite(return_travel_time)
            ):
                raise ValueError(
                    "Invalid depot-return travel time: "
                    f"{return_travel_time}"
                )

            total_distance += return_distance
            total_travel_time += return_travel_time

            current_time += return_travel_time

        except PathNotFoundError as exc:
            connectivity_feasible = False

            errors.append(
                f"Vehicle {vehicle_route.vehicle_id!r} "
                f"cannot return from node "
                f"{current_node!r} to depot/end node "
                f"{end_node!r}: {exc}"
            )

        except ValueError as exc:
            connectivity_feasible = False

            errors.append(
                f"Vehicle {vehicle_route.vehicle_id!r} "
                f"has invalid depot-return travel time: {exc}"
            )

        # --------------------------------------------------------------
        # Physical route
        # --------------------------------------------------------------

        physical_route = PhysicalRoute.from_legs(
            vehicle_id=vehicle_route.vehicle_id,
            start_node=start_node,
            end_node=end_node,
            legs=physical_legs,
        )

        elapsed_time = (
            current_time
            - vehicle_start_time
        )

        feasible = (
            connectivity_feasible
            and capacity_feasible
            and time_window_feasible
            and not any(
                "Unknown customer" in error
                for error in errors
            )
        )

        return VehicleRouteEvaluation(
            vehicle_id=vehicle_route.vehicle_id,
            physical_route=physical_route,
            stops=tuple(stop_results),
            feasible=feasible,
            connectivity_feasible=(
                connectivity_feasible
            ),
            capacity_feasible=(
                capacity_feasible
            ),
            time_window_feasible=(
                time_window_feasible
            ),
            capacity_units=float(
                capacity
            ),
            final_load_units=float(
                current_load
            ),
            maximum_load_units=float(
                maximum_load
            ),
            total_distance_m=float(
                total_distance
            ),
            total_travel_time_s=float(
                total_travel_time
            ),
            total_waiting_time_s=float(
                total_waiting
            ),
            total_service_time_s=float(
                total_service
            ),
            elapsed_time_s=float(
                elapsed_time
            ),
            lateness_s=float(
                total_lateness
            ),
            errors=tuple(errors),
        )

    def _unknown_vehicle_evaluation(
        self,
        vehicle_route: VehicleRoute,
    ) -> VehicleRouteEvaluation:
        """Create a deterministic failure result for an unknown vehicle."""

        return VehicleRouteEvaluation(
            vehicle_id=vehicle_route.vehicle_id,
            physical_route=None,
            stops=(),
            feasible=False,
            connectivity_feasible=False,
            capacity_feasible=False,
            time_window_feasible=False,
            capacity_units=0.0,
            final_load_units=0.0,
            maximum_load_units=0.0,
            total_distance_m=0.0,
            total_travel_time_s=0.0,
            total_waiting_time_s=0.0,
            total_service_time_s=0.0,
            elapsed_time_s=0.0,
            lateness_s=0.0,
            errors=(
                f"Unknown vehicle_id="
                f"{vehicle_route.vehicle_id!r}",
            ),
        )

    # ==================================================================
    # Dynamic travel-time handling
    # ==================================================================

    def _path_travel_time(
        self,
        path: PathResult,
        departure_time_s: TimeS,
    ) -> TimeS:
        """Return the dynamic travel time of an already-selected path.

        This method is retained as a utility for callers that need to
        re-evaluate a known physical path.

        Normal route evaluation should use PathBuilder's dynamic shortest
        path directly because dynamic costs must influence route selection.
        """

        if self.travel_time_provider is None:
            return float(
                path.travel_time_s
            )

        current_time = float(
            departure_time_s
        )

        total = 0.0

        for edge_id in path.edge_ids:
            edge = self.path_builder.graph.edge_for(
                edge_id
            )

            travel_time = float(
                self.travel_time_provider(
                    edge,
                    current_time,
                )
            )

            if (
                travel_time < 0.0
                or not isfinite(travel_time)
            ):
                raise ValueError(
                    "travel_time_provider returned an invalid "
                    f"travel time for edge {edge_id!r}: "
                    f"{travel_time}"
                )

            total += travel_time
            current_time += travel_time

        return float(total)

    @staticmethod
    def _path_to_leg(
        path: PathResult,
    ) -> RouteLeg:
        """Convert a physical path result into a route leg."""

        if not path.node_ids:
            raise ValueError(
                "PathResult contains no nodes"
            )

        return RouteLeg.from_sequence(
            from_node=path.node_ids[0],
            to_node=path.node_ids[-1],
            edge_ids=path.edge_ids,
            distance_m=path.distance_m,
            travel_time_s=path.travel_time_s,
        )

    # ==================================================================
    # Objective
    # ==================================================================

    def _objective_value(
        self,
        *,
        distance_m: float,
        travel_time_s: float,
        waiting_time_s: float,
        service_time_s: float,
        lateness_s: float,
        vehicle_evaluations: Iterable[
            VehicleRouteEvaluation
        ],
        duplicate_customer_count: int,
        unserved_customer_count: int,
        unknown_vehicle_count: int,
    ) -> float:
        """Calculate the configured scalar route objective."""

        value = (
            self.config.distance_weight
            * distance_m
            + self.config.travel_time_weight
            * travel_time_s
            + self.config.waiting_time_weight
            * waiting_time_s
            + self.config.service_time_weight
            * service_time_s
            + self.config.lateness_penalty
            * lateness_s
        )

        capacity_violations = sum(
            1
            for evaluation in vehicle_evaluations
            if not evaluation.capacity_feasible
        )

        connectivity_violations = sum(
            1
            for evaluation in vehicle_evaluations
            if not evaluation.connectivity_feasible
        )

        value += (
            self.config.capacity_penalty
            * capacity_violations
        )

        value += (
            self.config.connectivity_penalty
            * connectivity_violations
        )

        value += (
            self.config.duplicate_customer_penalty
            * duplicate_customer_count
        )

        value += (
            self.config.unserved_customer_penalty
            * unserved_customer_count
        )

        value += (
            self.config.unknown_vehicle_penalty
            * unknown_vehicle_count
        )

        return float(value)

    # ==================================================================
    # Scenario collections
    # ==================================================================

    def _scenario_edges(
        self,
    ) -> Iterable[RoadEdge]:
        return self._collection(
            self.scenario,
            "edges",
        )

    def _scenario_requests(
        self,
    ) -> Iterable[Any]:
        return self._collection(
            self.scenario,
            "requests",
        )

    def _scenario_vehicles(
        self,
    ) -> Iterable[Any]:
        return self._collection(
            self.scenario,
            "fleet",
        )

    @staticmethod
    def _collection(
        scenario: Any,
        name: str,
    ) -> Iterable[Any]:
        if not hasattr(
            scenario,
            name,
        ):
            raise AttributeError(
                f"Scenario does not expose {name!r}"
            )

        value = getattr(
            scenario,
            name,
        )

        if value is None:
            raise AttributeError(
                f"Scenario.{name} is None"
            )

        return value

    # ==================================================================
    # V1.2 request accessors
    # ==================================================================

    @staticmethod
    def _request_id(
        request: Any,
    ) -> CustomerId:
        return request.request_id

    @staticmethod
    def _request_node(
        request: Any,
    ) -> RoadNodeId:
        return request.access_node_id

    @staticmethod
    def _request_demand(
        request: Any,
    ) -> DemandUnits:
        return request.demand

    @staticmethod
    def _request_service_time(
        request: Any,
    ) -> TimeS:
        return request.service_duration_s

    @staticmethod
    def _request_release_time(
        request: Any,
    ) -> TimeS:
        return request.release_s

    @staticmethod
    def _request_earliest_time(
        request: Any,
    ) -> TimeS:
        return request.earliest_service_start_s

    @staticmethod
    def _request_latest_time(
        request: Any,
    ) -> TimeS:
        return request.latest_service_start_s

    # ==================================================================
    # V1.2 vehicle accessors
    # ==================================================================

    @staticmethod
    def _vehicle_id(
        vehicle: Any,
    ) -> VehicleId:
        return vehicle.vehicle_id

    @staticmethod
    def _vehicle_start_node(
        vehicle: Any,
    ) -> RoadNodeId:
        return vehicle.start_node_id

    @staticmethod
    def _vehicle_end_node(
        vehicle: Any,
    ) -> RoadNodeId:
        return vehicle.depot_node_id

    @staticmethod
    def _vehicle_capacity(
        vehicle: Any,
    ) -> DemandUnits:
        return vehicle.capacity