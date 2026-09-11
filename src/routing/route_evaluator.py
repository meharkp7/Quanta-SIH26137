"""
Independent route evaluation for the Step 5 routing stack.

The evaluator is the feasibility-truth layer of the routing system.

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
- enforce current vehicle state;
- enforce frozen/committed route prefixes;
- expose structured violations and objective components;
- support immutable CostView snapshots;
- remain independent of optimization and learning algorithms.

The evaluator does not optimize routes.
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
from src.routing.cost_view import CostView
from src.routing.evaluator_state import (
    CommitmentSnapshot,
    EvaluationConstraints,
    VehicleCommitment,
)
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
    commitment_penalty: float = 10_000.0
    depot_penalty: float = 10_000.0

    allow_waiting: bool = True

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
            "commitment_penalty",
            "depot_penalty",
            "default_start_time_s",
        )

        for field_name in numeric_fields:
            value = float(getattr(self, field_name))

            if not isfinite(value):
                raise ValueError(
                    f"{field_name} must be finite"
                )

            if (
                field_name.endswith("_penalty")
                or field_name.endswith("_weight")
            ) and value < 0.0:
                raise ValueError(
                    f"{field_name} must be non-negative"
                )

        if self.default_start_time_s < 0.0:
            raise ValueError(
                "default_start_time_s must be non-negative"
            )


@dataclass(frozen=True)
class EvaluationViolation:
    """
    Structured evaluator diagnostic.

    name is a stable machine-readable category.
    magnitude is the measured violation amount.
    vehicle_id/customer_id are optional contextual identifiers.
    message is human-readable diagnostic text.
    """

    name: str
    magnitude: float = 0.0
    count: int = 1
    vehicle_id: VehicleId | None = None
    customer_id: CustomerId | None = None
    message: str = ""

    def __post_init__(self) -> None:
        magnitude = float(self.magnitude)

        if not isfinite(magnitude) or magnitude < 0.0:
            raise ValueError(
                "violation magnitude must be finite and non-negative"
            )

        if self.count < 1:
            raise ValueError(
                "violation count must be >= 1"
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
        return self.service_end_time_s

    @property
    def lateness_s(self) -> TimeS:
        return max(
            0.0,
            float(self.service_start_time_s)
            - float(self.latest_time_s),
        )

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
    commitment_feasible: bool
    depot_feasible: bool

    capacity_units: DemandUnits
    initial_load_units: DemandUnits
    final_load_units: DemandUnits
    maximum_load_units: DemandUnits

    total_distance_m: DistanceM
    total_travel_time_s: TimeS
    total_waiting_time_s: TimeS
    total_service_time_s: TimeS
    elapsed_time_s: TimeS
    lateness_s: TimeS

    violations: tuple[EvaluationViolation, ...]
    errors: tuple[str, ...]

    @property
    def customer_ids(self) -> tuple[CustomerId, ...]:
        return tuple(
            stop.customer_id
            for stop in self.stops
        )

    @property
    def total_elapsed_time_s(self) -> TimeS:
        return self.elapsed_time_s

    @property
    def total_lateness_s(self) -> TimeS:
        return self.lateness_s

    @property
    def service_starts_s(self) -> Mapping[CustomerId, float]:
        return {
            stop.customer_id: float(
                stop.service_start_time_s
            )
            for stop in self.stops
        }

    @property
    def service_ends_s(self) -> Mapping[CustomerId, float]:
        return {
            stop.customer_id: float(
                stop.service_end_time_s
            )
            for stop in self.stops
        }

    @property
    def objective_components(self) -> Mapping[str, float]:
        return {
            "distance_m": float(self.total_distance_m),
            "travel_time_s": float(
                self.total_travel_time_s
            ),
            "waiting_time_s": float(
                self.total_waiting_time_s
            ),
            "service_time_s": float(
                self.total_service_time_s
            ),
            "elapsed_time_s": float(
                self.elapsed_time_s
            ),
            "lateness_s": float(
                self.lateness_s
            ),
        }


@dataclass(frozen=True)
class RoutePlanEvaluation:
    """Evaluation of a complete multi-vehicle logical route plan."""

    route_plan: RoutePlan
    vehicle_evaluations: tuple[
        VehicleRouteEvaluation,
        ...,
    ]

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

    violations: tuple[EvaluationViolation, ...]
    errors: tuple[str, ...]

    @property
    def assignment_feasible(self) -> bool:
        return (
            self.all_requests_served
            and self.customer_uniqueness_feasible
            and not self.unknown_request_ids
            and not self.unknown_vehicle_ids
        )

    @property
    def capacity_feasible(self) -> bool:
        return all(
            evaluation.capacity_feasible
            for evaluation in self.vehicle_evaluations
        )

    @property
    def connectivity_feasible(self) -> bool:
        return all(
            evaluation.connectivity_feasible
            for evaluation in self.vehicle_evaluations
        )

    @property
    def time_window_feasible(self) -> bool:
        return all(
            evaluation.time_window_feasible
            for evaluation in self.vehicle_evaluations
        )

    @property
    def commitment_feasible(self) -> bool:
        return all(
            evaluation.commitment_feasible
            for evaluation in self.vehicle_evaluations
        )

    @property
    def depot_feasible(self) -> bool:
        return all(
            evaluation.depot_feasible
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
        return self.total_travel_time_s

    @property
    def total_waiting_time(self) -> TimeS:
        return self.total_waiting_time_s

    @property
    def total_lateness(self) -> TimeS:
        return self.total_lateness_s

    @property
    def objective_components(self) -> Mapping[str, float]:
        return {
            "distance_m": float(self.total_distance_m),
            "travel_time_s": float(
                self.total_travel_time_s
            ),
            "waiting_time_s": float(
                self.total_waiting_time_s
            ),
            "service_time_s": float(
                self.total_service_time_s
            ),
            "elapsed_time_s": float(
                self.total_elapsed_time_s
            ),
            "lateness_s": float(
                self.total_lateness_s
            ),
        }


class RouteEvaluator:
    """
    Independent feasibility/objective evaluator.

    Optimizers must not be embedded here.

    The evaluator can evaluate against an explicit CostView and immutable
    constraint/commitment snapshots without mutating the evaluator itself.
    """

    def __init__(
        self,
        scenario: Any,
        *,
        config: RouteEvaluationConfig | None = None,
        travel_time_provider: TravelTimeProvider | None = None,
        closed_edge_ids: Iterable[RoadEdgeId] = (),
        cost_view: CostView | None = None,
    ) -> None:
        self.scenario = scenario
        self.config = (
            config or RouteEvaluationConfig()
        )

        self.travel_time_provider = (
            travel_time_provider
        )

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

        self._request_by_id = {
            self._request_id(request): request
            for request in self._requests
        }

        self._vehicle_by_id = {
            self._vehicle_id(vehicle): vehicle
            for vehicle in self._vehicles
        }

        if len(self._request_by_id) != len(
            self._requests
        ):
            raise ValueError(
                "Scenario contains duplicate request IDs"
            )

        if len(self._vehicle_by_id) != len(
            self._vehicles
        ):
            raise ValueError(
                "Scenario contains duplicate vehicle IDs"
            )

        self.cost_view = cost_view

        self.path_builder = DirectedPathBuilder(
            self._edges,
            closed_edge_ids=self._closed_edge_ids,
            travel_time_provider=(
                self.travel_time_provider
            ),
            cost_view=self.cost_view,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def evaluate(
        self,
        route_plan: RoutePlan,
        cost_view: CostView | None = None,
        constraints: EvaluationConstraints | None = None,
        commitments: CommitmentSnapshot | None = None,
        *,
        planning_time_s: TimeS | None = None,
    ) -> RoutePlanEvaluation:
        """
        Evaluate a complete route plan.

        Step 5C interface:

            evaluate(
                routes,
                cost_view,
                constraints,
                commitments,
            )

        planning_time_s remains as a compatibility argument for existing
        callers.
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

        if not isfinite(
            float(evaluation_time)
        ):
            raise ValueError(
                "planning_time_s must be finite"
            )

        if evaluation_time < 0.0:
            raise ValueError(
                "planning_time_s must be non-negative"
            )

        active_constraints = (
            constraints
            if constraints is not None
            else EvaluationConstraints.from_config(
                self.config
            )
        )

        active_cost_view = (
            cost_view
            if cost_view is not None
            else self.cost_view
        )

        active_commitments = (
            commitments
            if commitments is not None
            else CommitmentSnapshot.from_scenario(
                self._vehicles,
                default_time_s=evaluation_time,
            )
        )

        path_builder = self._make_path_builder(
            active_cost_view
        )

        vehicle_evaluations = VehicleEvaluationCollection(
            self._evaluate_vehicle_route(
                vehicle_route,
                planning_time_s=evaluation_time,
                cost_view=active_cost_view,
                constraints=active_constraints,
                commitment=active_commitments.for_vehicle(
                    vehicle_route.vehicle_id
                ),
                path_builder=path_builder,
            )
            for vehicle_route
            in route_plan.vehicle_routes
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
                for customer_id
                in route_plan.all_customer_ids()
                if customer_id
                not in self._request_by_id
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
            for customer_id
            in required_customer_ids
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
        violations: list[EvaluationViolation] = []

        for evaluation in vehicle_evaluations:
            errors.extend(
                evaluation.errors
            )
            violations.extend(
                evaluation.violations
            )

        if unknown_vehicle_ids:
            message = (
                "Unknown vehicle assignments: "
                + ", ".join(
                    str(vehicle_id)
                    for vehicle_id
                    in unknown_vehicle_ids
                )
            )

            errors.append(message)

            violations.append(
                EvaluationViolation(
                    name="unknown_vehicle",
                    magnitude=float(
                        len(unknown_vehicle_ids)
                    ),
                    count=len(
                        unknown_vehicle_ids
                    ),
                    message=message,
                )
            )

        if unknown_request_ids:
            message = (
                "Unknown request assignments: "
                + ", ".join(
                    str(customer_id)
                    for customer_id
                    in unknown_request_ids
                )
            )

            errors.append(message)

            violations.append(
                EvaluationViolation(
                    name="unknown_request",
                    magnitude=float(
                        len(unknown_request_ids)
                    ),
                    count=len(
                        unknown_request_ids
                    ),
                    message=message,
                )
            )

        if duplicate_customer_ids:
            message = (
                "Duplicate customer assignments: "
                + ", ".join(
                    str(customer_id)
                    for customer_id
                    in duplicate_customer_ids
                )
            )

            errors.append(message)

            violations.append(
                EvaluationViolation(
                    name="duplicate_customer",
                    magnitude=float(
                        len(duplicate_customer_ids)
                    ),
                    count=len(
                        duplicate_customer_ids
                    ),
                    message=message,
                )
            )

        if (
            active_constraints.require_all_requests_served
            and unserved_customer_ids
        ):
            message = (
                "Unserved customers: "
                + ", ".join(
                    str(customer_id)
                    for customer_id
                    in unserved_customer_ids
                )
            )

            errors.append(message)

            violations.append(
                EvaluationViolation(
                    name="unserved_customer",
                    magnitude=float(
                        len(unserved_customer_ids)
                    ),
                    count=len(
                        unserved_customer_ids
                    ),
                    message=message,
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
            duplicate_customer_count=len(
                duplicate_customer_ids
            ),
            unserved_customer_count=len(
                unserved_customer_ids
            ),
            unknown_vehicle_count=len(
                unknown_vehicle_ids
            ),
            commitment_violation_count=sum(
                not evaluation.commitment_feasible
                for evaluation
                in vehicle_evaluations
            ),
            depot_violation_count=sum(
                not evaluation.depot_feasible
                for evaluation
                in vehicle_evaluations
            ),
        )

        assignment_failure = (
            not customer_uniqueness_feasible
            or bool(unknown_request_ids)
            or bool(unknown_vehicle_ids)
        )

        if active_constraints.require_all_requests_served:
            assignment_failure = (
                assignment_failure
                or not all_requests_served
            )

        feasible = (
            all(
                evaluation.feasible
                for evaluation
                in vehicle_evaluations
            )
            and not assignment_failure
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
            total_elapsed_time_s=float(
                total_elapsed
            ),
            total_lateness_s=float(
                total_lateness
            ),
            objective_value=float(
                objective
            ),
            violations=tuple(
                violations
            ),
            errors=tuple(errors),
        )

    def evaluate_vehicle_route(
        self,
        vehicle_route: VehicleRoute,
        *,
        planning_time_s: TimeS | None = None,
        cost_view: CostView | None = None,
        constraints: EvaluationConstraints | None = None,
        commitment: VehicleCommitment | None = None,
    ) -> VehicleRouteEvaluation:
        """Evaluate one vehicle route independently."""

        evaluation_time = (
            self.config.default_start_time_s
            if planning_time_s is None
            else float(planning_time_s)
        )

        active_constraints = (
            constraints
            if constraints is not None
            else EvaluationConstraints.from_config(
                self.config
            )
        )

        active_cost_view = (
            cost_view
            if cost_view is not None
            else self.cost_view
        )

        path_builder = self._make_path_builder(
            active_cost_view
        )

        return self._evaluate_vehicle_route(
            vehicle_route,
            planning_time_s=evaluation_time,
            cost_view=active_cost_view,
            constraints=active_constraints,
            commitment=commitment,
            path_builder=path_builder,
        )

    def with_network_state(
        self,
        *,
        closed_edge_ids: Iterable[RoadEdgeId] = (),
        travel_time_provider: TravelTimeProvider | None = None,
        cost_view: CostView | None = None,
    ) -> "RouteEvaluator":
        """
        Create an immutable evaluator for a network-state snapshot.
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
            cost_view=(
                self.cost_view
                if cost_view is None
                else cost_view
            ),
        )

    # ==================================================================
    # Vehicle evaluation
    # ==================================================================

    def _evaluate_vehicle_route(
        self,
        vehicle_route: VehicleRoute,
        *,
        planning_time_s: TimeS,
        cost_view: CostView | None,
        constraints: EvaluationConstraints,
        commitment: VehicleCommitment | None,
        path_builder: DirectedPathBuilder,
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

        violations: list[EvaluationViolation] = []
        errors: list[str] = []

        commitment_feasible = True

        # --------------------------------------------------------------
        # Resolve immutable state snapshot.
        # --------------------------------------------------------------

        if commitment is not None:
            if (
                commitment.vehicle_id
                != vehicle_route.vehicle_id
            ):
                commitment_feasible = False

                message = (
                    "Commitment vehicle_id does not match "
                    "route vehicle_id"
                )

                errors.append(message)

                violations.append(
                    EvaluationViolation(
                        name="commitment_vehicle_mismatch",
                        vehicle_id=vehicle_route.vehicle_id,
                        message=message,
                    )
                )

            current_node = commitment.current_node_id

            vehicle_start_time = float(
                commitment.current_time_s
            )
            current_time = vehicle_start_time

            current_load = float(
                commitment.current_load_units
            )

            onboard_request_ids = set(
                commitment.onboard_request_ids
            )

            committed_customer_ids = tuple(
                commitment.committed_customer_ids
            )

            frozen_prefix_edge_ids = tuple(
                commitment.frozen_prefix_edge_ids
            )

        else:
            current_node = start_node

            vehicle_start_time = float(
                planning_time_s
            )
            current_time = vehicle_start_time

            current_load = 0.0
            onboard_request_ids = set()
            committed_customer_ids = ()
            frozen_prefix_edge_ids = ()

        if current_load < -constraints.load_tolerance:
            commitment_feasible = False

            message = (
                f"Vehicle {vehicle_route.vehicle_id!r} "
                f"has negative current load "
                f"{current_load}"
            )

            errors.append(message)

            violations.append(
                EvaluationViolation(
                    name="negative_current_load",
                    magnitude=abs(current_load),
                    vehicle_id=vehicle_route.vehicle_id,
                    message=message,
                )
            )

        if current_load > capacity + constraints.load_tolerance:
            commitment_feasible = False

            excess = current_load - capacity

            message = (
                f"Vehicle {vehicle_route.vehicle_id!r} "
                f"starts above capacity: "
                f"{current_load} > {capacity}"
            )

            errors.append(message)

            violations.append(
                EvaluationViolation(
                    name="initial_capacity_exceeded",
                    magnitude=excess,
                    vehicle_id=vehicle_route.vehicle_id,
                    message=message,
                )
            )

        initial_load = max(
            0.0,
            current_load,
        )

        maximum_load = initial_load

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
        depot_feasible = True

        # --------------------------------------------------------------
        # Validate commitments before traversing mutable continuation.
        # --------------------------------------------------------------

        route_customer_ids = tuple(
            vehicle_route.customer_ids
        )

        if constraints.require_commitments:
            if committed_customer_ids:
                expected_prefix = (
                    committed_customer_ids
                )

                actual_prefix = route_customer_ids[
                    : len(expected_prefix)
                ]

                if actual_prefix != expected_prefix:
                    commitment_feasible = False

                    message = (
                        f"Vehicle {vehicle_route.vehicle_id!r} "
                        f"does not preserve committed customer prefix. "
                        f"expected={expected_prefix!r}, "
                        f"actual={actual_prefix!r}"
                    )

                    errors.append(message)

                    violations.append(
                        EvaluationViolation(
                            name="committed_customer_prefix",
                            magnitude=float(
                                len(expected_prefix)
                                - sum(
                                    a == b
                                    for a, b in zip(
                                        actual_prefix,
                                        expected_prefix,
                                    )
                                )
                            ),
                            vehicle_id=vehicle_route.vehicle_id,
                            message=message,
                        )
                    )

            if frozen_prefix_edge_ids:
                candidate_edges = tuple(
                    edge_id
                    for leg in vehicle_route.legs
                    for edge_id in leg.physical_edge_ids
                )

                prefix = candidate_edges[
                    : len(frozen_prefix_edge_ids)
                ]

                if prefix != frozen_prefix_edge_ids:
                    # A route generated from a rolling state normally starts
                    # AFTER the frozen physical prefix, so an absent prefix is
                    # legal when current_node_id already represents the
                    # post-prefix state. If the candidate explicitly carries
                    # physical edges, however, they must agree.
                    if candidate_edges:
                        commitment_feasible = False

                        message = (
                            f"Vehicle {vehicle_route.vehicle_id!r} "
                            "violates frozen physical prefix"
                        )

                        errors.append(message)

                        violations.append(
                            EvaluationViolation(
                                name="frozen_physical_prefix",
                                magnitude=float(
                                    len(
                                        frozen_prefix_edge_ids
                                    )
                                ),
                                vehicle_id=(
                                    vehicle_route.vehicle_id
                                ),
                                message=message,
                            )
                        )

        # --------------------------------------------------------------
        # Customer visits.
        # --------------------------------------------------------------

        for customer_id in route_customer_ids:
            request = self._request_by_id.get(
                customer_id
            )

            if request is None:
                message = (
                    f"Unknown customer_id="
                    f"{customer_id!r}"
                )

                errors.append(message)

                connectivity_feasible = False
                time_window_feasible = False

                violations.append(
                    EvaluationViolation(
                        name="unknown_request",
                        vehicle_id=vehicle_route.vehicle_id,
                        customer_id=customer_id,
                        message=message,
                    )
                )

                continue

            customer_node = self._request_node(
                request
            )

            try:
                path = path_builder.shortest_path(
                    current_node,
                    customer_node,
                    departure_time_s=current_time,
                )

            except PathNotFoundError as exc:
                connectivity_feasible = False

                message = (
                    f"Vehicle {vehicle_route.vehicle_id!r}: "
                    f"cannot reach customer "
                    f"{customer_id!r} from node "
                    f"{current_node!r}: {exc}"
                )

                errors.append(message)

                violations.append(
                    EvaluationViolation(
                        name="connectivity",
                        magnitude=1.0,
                        vehicle_id=(
                            vehicle_route.vehicle_id
                        ),
                        customer_id=customer_id,
                        message=message,
                    )
                )

                continue

            except ValueError as exc:
                connectivity_feasible = False

                message = (
                    f"Vehicle {vehicle_route.vehicle_id!r}: "
                    f"invalid dynamic travel-time evaluation "
                    f"for customer {customer_id!r}: {exc}"
                )

                errors.append(message)

                violations.append(
                    EvaluationViolation(
                        name="invalid_edge_cost",
                        magnitude=1.0,
                        vehicle_id=(
                            vehicle_route.vehicle_id
                        ),
                        customer_id=customer_id,
                        message=message,
                    )
                )

                continue

            physical_legs.append(
                self._path_to_leg(path)
            )

            distance = float(
                path.distance_m
            )

            travel_time = float(
                path.travel_time_s
            )

            if (
                travel_time < 0.0
                or not isfinite(travel_time)
            ):
                connectivity_feasible = False

                message = (
                    f"Invalid travel time for transition "
                    f"{current_node!r} -> "
                    f"{customer_node!r}: "
                    f"{travel_time}"
                )

                errors.append(message)

                violations.append(
                    EvaluationViolation(
                        name="invalid_travel_time",
                        magnitude=1.0,
                        vehicle_id=(
                            vehicle_route.vehicle_id
                        ),
                        customer_id=customer_id,
                        message=message,
                    )
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

            minimum_service_start = max(
                release_time,
                earliest,
            )

            service_start = arrival_time
            waiting_time = 0.0

            if (
                service_start
                < minimum_service_start
                - constraints.time_tolerance
            ):
                required_wait = (
                    minimum_service_start
                    - service_start
                )

                if constraints.allow_waiting:
                    waiting_time = required_wait
                    service_start = (
                        minimum_service_start
                    )
                else:
                    time_window_feasible = False

                    message = (
                        f"Customer {customer_id!r} "
                        f"requires {required_wait} seconds "
                        "of waiting, but waiting is disabled"
                    )

                    errors.append(message)

                    violations.append(
                        EvaluationViolation(
                            name="waiting_disabled",
                            magnitude=required_wait,
                            vehicle_id=(
                                vehicle_route.vehicle_id
                            ),
                            customer_id=customer_id,
                            message=message,
                        )
                    )

            service_end = (
                service_start
                + service_time
            )

            load_before = current_load

            current_load += demand

            maximum_load = max(
                maximum_load,
                current_load,
            )

            stop_capacity_feasible = (
                current_load
                <= capacity
                + constraints.load_tolerance
            )

            if constraints.enforce_capacity:
                if not stop_capacity_feasible:
                    capacity_feasible = False

                    excess = (
                        current_load
                        - capacity
                    )

                    message = (
                        f"Vehicle {vehicle_route.vehicle_id!r} "
                        f"exceeds capacity at customer "
                        f"{customer_id!r}: "
                        f"{current_load} > {capacity}"
                    )

                    errors.append(message)

                    violations.append(
                        EvaluationViolation(
                            name="capacity",
                            magnitude=max(
                                0.0,
                                excess,
                            ),
                            vehicle_id=(
                                vehicle_route.vehicle_id
                            ),
                            customer_id=customer_id,
                            message=message,
                        )
                    )

            release_feasible = (
                service_start
                >= release_time
                - constraints.time_tolerance
            )

            window_feasible = (
                earliest
                - constraints.time_tolerance
                <= service_start
                <= latest
                + constraints.time_tolerance
            )

            if constraints.enforce_releases:
                if not release_feasible:
                    time_window_feasible = False

                    magnitude = (
                        release_time
                        - service_start
                    )

                    message = (
                        f"Customer {customer_id!r} "
                        f"service starts before release "
                        f"time {release_time}: "
                        f"{service_start}"
                    )

                    errors.append(message)

                    violations.append(
                        EvaluationViolation(
                            name="release",
                            magnitude=max(
                                0.0,
                                magnitude,
                            ),
                            vehicle_id=(
                                vehicle_route.vehicle_id
                            ),
                            customer_id=customer_id,
                            message=message,
                        )
                    )

            if constraints.enforce_time_windows:
                if not window_feasible:
                    time_window_feasible = False

                    lateness = max(
                        0.0,
                        service_start
                        - latest,
                    )

                    total_lateness += (
                        lateness
                    )

                    message = (
                        f"Customer {customer_id!r} "
                        f"service starts outside time "
                        f"window [{earliest}, {latest}]: "
                        f"{service_start}"
                    )

                    errors.append(message)

                    violations.append(
                        EvaluationViolation(
                            name="time_window",
                            magnitude=lateness,
                            vehicle_id=(
                                vehicle_route.vehicle_id
                            ),
                            customer_id=customer_id,
                            message=message,
                        )
                    )

            stop_results.append(
                StopEvaluation(
                    customer_id=customer_id,
                    node_id=customer_node,
                    edge_ids=tuple(
                        path.edge_ids
                    ),
                    distance_m=distance,
                    travel_time_s=travel_time,
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
                    release_time_s=release_time,
                    earliest_time_s=earliest,
                    latest_time_s=latest,
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
        # Depot return.
        # --------------------------------------------------------------

        if constraints.require_depot_return:
            try:
                return_path = (
                    path_builder.shortest_path(
                        current_node,
                        end_node,
                        departure_time_s=current_time,
                    )
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
                    or not isfinite(
                        return_travel_time
                    )
                ):
                    raise ValueError(
                        "Invalid depot-return "
                        f"travel time: "
                        f"{return_travel_time}"
                    )

                total_distance += (
                    return_distance
                )

                total_travel_time += (
                    return_travel_time
                )

                current_time += (
                    return_travel_time
                )

            except PathNotFoundError as exc:
                depot_feasible = False
                connectivity_feasible = False

                message = (
                    f"Vehicle "
                    f"{vehicle_route.vehicle_id!r} "
                    f"cannot return from node "
                    f"{current_node!r} to depot "
                    f"{end_node!r}: {exc}"
                )

                errors.append(message)

                violations.append(
                    EvaluationViolation(
                        name="depot_return",
                        magnitude=1.0,
                        vehicle_id=(
                            vehicle_route.vehicle_id
                        ),
                        message=message,
                    )
                )

            except ValueError as exc:
                depot_feasible = False
                connectivity_feasible = False

                message = (
                    f"Vehicle "
                    f"{vehicle_route.vehicle_id!r} "
                    f"has invalid depot-return "
                    f"travel time: {exc}"
                )

                errors.append(message)

                violations.append(
                    EvaluationViolation(
                        name="depot_return",
                        magnitude=1.0,
                        vehicle_id=(
                            vehicle_route.vehicle_id
                        ),
                        message=message,
                    )
                )

        # --------------------------------------------------------------
        # Physical route.
        # --------------------------------------------------------------

        physical_route = PhysicalRoute.from_legs(
            vehicle_id=vehicle_route.vehicle_id,
            start_node=(
                commitment.current_node_id
                if commitment is not None
                else start_node
            ),
            end_node=end_node,
            legs=physical_legs,
        )

        elapsed_time = (
            current_time
            - vehicle_start_time
        )

        feasible = (
            connectivity_feasible
            and (
                capacity_feasible
                if constraints.enforce_capacity
                else True
            )
            and (
                time_window_feasible
                if (
                    constraints.enforce_time_windows
                    or constraints.enforce_releases
                )
                else True
            )
            and (
                commitment_feasible
                if constraints.require_commitments
                else True
            )
            and (
                depot_feasible
                if constraints.require_depot_return
                else True
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
            commitment_feasible=(
                commitment_feasible
            ),
            depot_feasible=(
                depot_feasible
            ),
            capacity_units=float(
                capacity
            ),
            initial_load_units=float(
                initial_load
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
            violations=tuple(
                violations
            ),
            errors=tuple(errors),
        )

    def _unknown_vehicle_evaluation(
        self,
        vehicle_route: VehicleRoute,
    ) -> VehicleRouteEvaluation:
        message = (
            f"Unknown vehicle_id="
            f"{vehicle_route.vehicle_id!r}"
        )

        violation = EvaluationViolation(
            name="unknown_vehicle",
            magnitude=1.0,
            vehicle_id=vehicle_route.vehicle_id,
            message=message,
        )

        return VehicleRouteEvaluation(
            vehicle_id=vehicle_route.vehicle_id,
            physical_route=None,
            stops=(),
            feasible=False,
            connectivity_feasible=False,
            capacity_feasible=False,
            time_window_feasible=False,
            commitment_feasible=False,
            depot_feasible=False,
            capacity_units=0.0,
            initial_load_units=0.0,
            final_load_units=0.0,
            maximum_load_units=0.0,
            total_distance_m=0.0,
            total_travel_time_s=0.0,
            total_waiting_time_s=0.0,
            total_service_time_s=0.0,
            elapsed_time_s=0.0,
            lateness_s=0.0,
            violations=(violation,),
            errors=(message,),
        )

    # ==================================================================
    # Path / CostView
    # ==================================================================

    def _make_path_builder(
        self,
        cost_view: CostView | None,
    ) -> DirectedPathBuilder:
        """
        Build a per-evaluation path builder.

        This is intentional: a caller-supplied CostView must never mutate
        shared evaluator state.
        """

        return DirectedPathBuilder(
            self._edges,
            closed_edge_ids=self._closed_edge_ids,
            travel_time_provider=(
                self.travel_time_provider
                if cost_view is None
                else None
            ),
            cost_view=cost_view,
        )

    def _path_travel_time(
        self,
        path: PathResult,
        departure_time_s: TimeS,
        *,
        cost_view: CostView | None = None,
    ) -> TimeS:
        """Re-evaluate an already selected physical path."""

        active_cost_view = (
            cost_view
            if cost_view is not None
            else self.cost_view
        )

        if active_cost_view is not None:
            return float(
                active_cost_view.path_cost(
                    path.edge_ids,
                    departure_time_s,
                ).actual_travel_time_s
            )

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
                    "travel_time_provider returned "
                    "an invalid travel time for edge "
                    f"{edge_id!r}: {travel_time}"
                )

            total += travel_time
            current_time += travel_time

        return float(total)

    @staticmethod
    def _path_to_leg(
        path: PathResult,
    ) -> RouteLeg:
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
        commitment_violation_count: int,
        depot_violation_count: int,
    ) -> float:
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
            for evaluation
            in vehicle_evaluations
            if not evaluation.capacity_feasible
        )

        connectivity_violations = sum(
            1
            for evaluation
            in vehicle_evaluations
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

        value += (
            self.config.commitment_penalty
            * commitment_violation_count
        )

        value += (
            self.config.depot_penalty
            * depot_violation_count
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
    # Request accessors
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
    # Vehicle accessors
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