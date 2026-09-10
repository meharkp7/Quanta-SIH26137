"""Deterministic route-plan repair for Step 4.

Repair converts an imperfect candidate RoutePlan into a more feasible
candidate without modifying the underlying Scenario.

The repair layer is deliberately separate from optimization. Constructive
heuristics, ALNS, QPSO, local search, DRL, or other optimizers may produce
candidate route plans that require repair before evaluation or execution.

Supported repairs:
- remove unknown customers;
- remove duplicate customer assignments;
- relocate customers between vehicles;
- repair capacity violations;
- repair directed-road connectivity violations;
- repair service-start time-window violations.

All operations are deterministic and scenario-immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.contracts.core_types import (
    CustomerId,
    VehicleId,
)
from src.routing.route_evaluator import (
    RouteEvaluator,
    RoutePlanEvaluation,
)
from src.routing.route_plan import (
    RoutePlan,
    VehicleRoute,
)


@dataclass(frozen=True)
class RouteRepairConfig:
    """Controls deterministic route repair."""

    max_passes: int = 100

    remove_unknown_customers: bool = True
    remove_duplicates: bool = True

    repair_capacity: bool = True
    repair_connectivity: bool = True
    repair_time_windows: bool = True

    allow_cross_vehicle_relocation: bool = True

    # Backwards-compatible spelling retained for callers that used the
    # earlier public configuration name. If provided, it takes precedence.
    enable_cross_vehicle_relocation: bool | None = None

    distance_weight: float = 1.0
    travel_time_weight: float = 1.0
    waiting_time_weight: float = 0.25
    lateness_weight: float = 100.0

    prioritize_feasibility: bool = True


@dataclass(frozen=True)
class RepairAction:
    """One deterministic modification made to a candidate route plan."""

    action: str
    customer_id: CustomerId | None
    source_vehicle_id: VehicleId | None
    destination_vehicle_id: VehicleId | None
    source_position: int | None
    destination_position: int | None
    reason: str


@dataclass(frozen=True)
class RouteRepairResult:
    """Complete output of the repair process."""

    original_route_plan: RoutePlan
    repaired_route_plan: RoutePlan

    original_evaluation: RoutePlanEvaluation
    repaired_evaluation: RoutePlanEvaluation

    actions: tuple[RepairAction, ...]

    changed: bool
    feasible: bool
    complete: bool

    passes_used: int

    unresolved_customer_ids: tuple[
        CustomerId,
        ...
    ]

    unresolved_reasons: tuple[str, ...]


class RouteRepairer:
    """Repair candidate RoutePlans using deterministic local operations."""

    def __init__(
        self,
        scenario: Any,
        evaluator: RouteEvaluator,
        *,
        config: RouteRepairConfig | None = None,
    ) -> None:
        self.scenario = scenario
        self.evaluator = evaluator
        self.config = config or RouteRepairConfig()

        if self.config.max_passes < 1:
            raise ValueError(
                "max_passes must be at least 1"
            )

        self._allow_cross_vehicle_relocation = (
            self.config.allow_cross_vehicle_relocation
            if self.config.enable_cross_vehicle_relocation is None
            else self.config.enable_cross_vehicle_relocation
        )

        self._requests = tuple(
            self._scenario_requests()
        )

        self._vehicles = tuple(
            self._scenario_vehicles()
        )

        self._request_by_id = {
            self._customer_id(request): request
            for request in self._requests
        }

        self._known_customer_ids = tuple(
            self._request_by_id.keys()
        )

        self._known_customer_set = set(
            self._known_customer_ids
        )

        self._vehicle_ids = tuple(
            self._vehicle_id(vehicle)
            for vehicle in self._vehicles
        )

        self._vehicle_id_set = set(
            self._vehicle_ids
        )
        self._planning_time_s = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def repair(
        self,
        route_plan: RoutePlan,
        *,
        planning_time_s: float = 0.0,
    ) -> RouteRepairResult:
        """Apply deterministic feasibility-oriented repair passes."""

        self._planning_time_s = float(planning_time_s)

        original_evaluation = (
            self.evaluator.evaluate(
                route_plan,
                planning_time_s=self._planning_time_s,
            )
        )

        current_plan = route_plan
        actions: list[RepairAction] = []

        passes_used = 0

        for pass_number in range(
            1,
            self.config.max_passes + 1,
        ):
            passes_used = pass_number
            changed_this_pass = False

            if self.config.remove_unknown_customers:
                (
                    current_plan,
                    repair_actions,
                    changed,
                ) = self._remove_unknown_customers(
                    current_plan
                )

                if changed:
                    actions.extend(
                        repair_actions
                    )
                    changed_this_pass = True

            if self.config.remove_duplicates:
                (
                    current_plan,
                    repair_actions,
                    changed,
                ) = self._remove_duplicates(
                    current_plan
                )

                if changed:
                    actions.extend(
                        repair_actions
                    )
                    changed_this_pass = True

            if self.config.repair_capacity:
                (
                    current_plan,
                    repair_actions,
                    changed,
                ) = self._repair_capacity(
                    current_plan
                )

                if changed:
                    actions.extend(
                        repair_actions
                    )
                    changed_this_pass = True

            if self.config.repair_connectivity:
                (
                    current_plan,
                    repair_actions,
                    changed,
                ) = self._repair_connectivity(
                    current_plan
                )

                if changed:
                    actions.extend(
                        repair_actions
                    )
                    changed_this_pass = True

            if self.config.repair_time_windows:
                (
                    current_plan,
                    repair_actions,
                    changed,
                ) = self._repair_time_windows(
                    current_plan
                )

                if changed:
                    actions.extend(
                        repair_actions
                    )
                    changed_this_pass = True

            if not changed_this_pass:
                break

        repaired_evaluation = (
            self.evaluator.evaluate(
                current_plan,
                planning_time_s=self._planning_time_s,
            )
        )

        (
            unresolved_customer_ids,
            unresolved_reasons,
        ) = self._unresolved_violations(
            repaired_evaluation
        )

        feasible = (
            repaired_evaluation.feasible
        )

        complete = (
            not repaired_evaluation.unserved_customer_ids
            and not repaired_evaluation.duplicate_customer_ids
        )

        return RouteRepairResult(
            original_route_plan=route_plan,
            repaired_route_plan=current_plan,
            original_evaluation=original_evaluation,
            repaired_evaluation=repaired_evaluation,
            actions=tuple(actions),
            changed=current_plan != route_plan,
            feasible=feasible,
            complete=complete,
            passes_used=passes_used,
            unresolved_customer_ids=(
                unresolved_customer_ids
            ),
            unresolved_reasons=(
                unresolved_reasons
            ),
        )

    # ------------------------------------------------------------------
    # Unknown customers
    # ------------------------------------------------------------------

    def _remove_unknown_customers(
        self,
        route_plan: RoutePlan,
    ) -> tuple[
        RoutePlan,
        list[RepairAction],
        bool,
    ]:
        actions: list[RepairAction] = []
        changed = False
        new_routes: list[VehicleRoute] = []

        for route in route_plan.vehicle_routes:
            retained: list[CustomerId] = []

            for position, customer_id in enumerate(
                route.customer_ids
            ):
                if customer_id in self._known_customer_set:
                    retained.append(customer_id)
                    continue

                actions.append(
                    RepairAction(
                        action=(
                            "remove_unknown_customer"
                        ),
                        customer_id=customer_id,
                        source_vehicle_id=(
                            route.vehicle_id
                        ),
                        destination_vehicle_id=None,
                        source_position=position,
                        destination_position=None,
                        reason=(
                            "customer does not exist "
                            "in scenario"
                        ),
                    )
                )

                changed = True

            new_routes.append(
                VehicleRoute.from_sequence(
                    vehicle_id=route.vehicle_id,
                    customer_ids=retained,
                )
            )

        if not changed:
            return (
                route_plan,
                actions,
                False,
            )

        return (
            RoutePlan.from_routes(new_routes),
            actions,
            True,
        )

    # ------------------------------------------------------------------
    # Duplicate assignments
    # ------------------------------------------------------------------

    def _remove_duplicates(
        self,
        route_plan: RoutePlan,
    ) -> tuple[
        RoutePlan,
        list[RepairAction],
        bool,
    ]:
        actions: list[RepairAction] = []
        changed = False

        seen: set[CustomerId] = set()
        new_routes: list[VehicleRoute] = []

        for route in route_plan.vehicle_routes:
            retained: list[CustomerId] = []

            for position, customer_id in enumerate(
                route.customer_ids
            ):
                if customer_id not in seen:
                    seen.add(customer_id)
                    retained.append(customer_id)
                    continue

                actions.append(
                    RepairAction(
                        action=(
                            "remove_duplicate_customer"
                        ),
                        customer_id=customer_id,
                        source_vehicle_id=(
                            route.vehicle_id
                        ),
                        destination_vehicle_id=None,
                        source_position=position,
                        destination_position=None,
                        reason=(
                            "customer already assigned "
                            "to another route"
                        ),
                    )
                )

                changed = True

            new_routes.append(
                VehicleRoute.from_sequence(
                    vehicle_id=route.vehicle_id,
                    customer_ids=retained,
                )
            )

        if not changed:
            return (
                route_plan,
                actions,
                False,
            )

        return (
            RoutePlan.from_routes(new_routes),
            actions,
            True,
        )

    # ------------------------------------------------------------------
    # Capacity repair
    # ------------------------------------------------------------------

    def _repair_capacity(
        self,
        route_plan: RoutePlan,
    ) -> tuple[
        RoutePlan,
        list[RepairAction],
        bool,
    ]:
        actions: list[RepairAction] = []
        current_plan = route_plan
        changed = False

        evaluation = self.evaluator.evaluate(
            current_plan,
            planning_time_s=self._planning_time_s,
        )

        for vehicle_evaluation in (
            evaluation.vehicle_evaluations
        ):
            if vehicle_evaluation.capacity_feasible:
                continue

            source_vehicle_id = (
                vehicle_evaluation.vehicle_id
            )

            route = current_plan.route_for(
                source_vehicle_id
            )

            candidates = sorted(
                enumerate(
                    route.customer_ids
                ),
                key=lambda item: (
                    self._request_demand(item[1]),
                    -float(self._request_latest(item[1])),
                    str(item[1]),
                    item[0],
                ),
            )

            for source_position, customer_id in candidates:
                if self._vehicle_route_is_capacity_feasible(
                    current_plan,
                    source_vehicle_id,
                ):
                    break

                destination = (
                    self._best_destination_for_customer(
                        current_plan,
                        source_vehicle_id=(
                            source_vehicle_id
                        ),
                        customer_id=customer_id,
                        reason="capacity",
                    )
                )

                if destination is None:
                    continue

                (
                    destination_vehicle_id,
                    destination_position,
                ) = destination

                current_plan = self._move_customer(
                    current_plan,
                    source_vehicle_id=(
                        source_vehicle_id
                    ),
                    customer_id=customer_id,
                    destination_vehicle_id=(
                        destination_vehicle_id
                    ),
                    destination_position=(
                        destination_position
                    ),
                )

                actions.append(
                    RepairAction(
                        action="relocate_customer",
                        customer_id=customer_id,
                        source_vehicle_id=(
                            source_vehicle_id
                        ),
                        destination_vehicle_id=(
                            destination_vehicle_id
                        ),
                        source_position=source_position,
                        destination_position=(
                            destination_position
                        ),
                        reason=(
                            "repair vehicle capacity "
                            "violation"
                        ),
                    )
                )

                changed = True

                evaluation = (
                    self.evaluator.evaluate(
                        current_plan,
                        planning_time_s=self._planning_time_s,
                    )
                )

                vehicle_evaluation = next(
                    result
                    for result in (
                        evaluation.vehicle_evaluations
                    )
                    if result.vehicle_id
                    == source_vehicle_id
                )

        return (
            current_plan,
            actions,
            changed,
        )

    # ------------------------------------------------------------------
    # Connectivity repair
    # ------------------------------------------------------------------

    def _repair_connectivity(
        self,
        route_plan: RoutePlan,
    ) -> tuple[
        RoutePlan,
        list[RepairAction],
        bool,
    ]:
        """Relocate stops from routes with directed-path failures."""

        actions: list[RepairAction] = []
        current_plan = route_plan
        changed = False

        evaluation = self.evaluator.evaluate(
            current_plan,
            planning_time_s=self._planning_time_s,
        )

        for vehicle_evaluation in (
            evaluation.vehicle_evaluations
        ):
            if vehicle_evaluation.connectivity_feasible:
                continue

            source_vehicle_id = (
                vehicle_evaluation.vehicle_id
            )

            route = current_plan.route_for(
                source_vehicle_id
            )

            # Try customers in deterministic route order.
            for source_position, customer_id in enumerate(
                route.customer_ids
            ):
                destination = (
                    self._best_destination_for_customer(
                        current_plan,
                        source_vehicle_id=(
                            source_vehicle_id
                        ),
                        customer_id=customer_id,
                        reason="connectivity",
                    )
                )

                if destination is None:
                    continue

                (
                    destination_vehicle_id,
                    destination_position,
                ) = destination

                candidate_plan = self._move_customer(
                    current_plan,
                    source_vehicle_id=(
                        source_vehicle_id
                    ),
                    customer_id=customer_id,
                    destination_vehicle_id=(
                        destination_vehicle_id
                    ),
                    destination_position=(
                        destination_position
                    ),
                )

                candidate_evaluation = (
                    self.evaluator.evaluate(
                        candidate_plan,
                        planning_time_s=self._planning_time_s,
                    )
                )

                candidate_source = next(
                    result
                    for result in (
                        candidate_evaluation
                        .vehicle_evaluations
                    )
                    if result.vehicle_id
                    == source_vehicle_id
                )

                # Do not move a customer if the source route remains
                # connectivity-infeasible.
                if not candidate_source.connectivity_feasible:
                    continue

                current_plan = candidate_plan

                actions.append(
                    RepairAction(
                        action="relocate_customer",
                        customer_id=customer_id,
                        source_vehicle_id=(
                            source_vehicle_id
                        ),
                        destination_vehicle_id=(
                            destination_vehicle_id
                        ),
                        source_position=source_position,
                        destination_position=(
                            destination_position
                        ),
                        reason=(
                            "repair directed-road "
                            "connectivity"
                        ),
                    )
                )

                changed = True
                break

        return (
            current_plan,
            actions,
            changed,
        )

    # ------------------------------------------------------------------
    # Time-window repair
    # ------------------------------------------------------------------

    def _repair_time_windows(
        self,
        route_plan: RoutePlan,
    ) -> tuple[
        RoutePlan,
        list[RepairAction],
        bool,
    ]:
        """Relocate customers causing release/window violations."""

        actions: list[RepairAction] = []
        current_plan = route_plan
        changed = False

        evaluation = self.evaluator.evaluate(
            current_plan,
            planning_time_s=self._planning_time_s,
        )

        for vehicle_evaluation in (
            evaluation.vehicle_evaluations
        ):
            if vehicle_evaluation.time_window_feasible:
                continue

            violated_stops = [
                stop
                for stop in vehicle_evaluation.stops
                if (
                    not stop.time_window_feasible
                    or not stop.release_feasible
                )
            ]

            violated_stops.sort(
                key=lambda stop: (
                    -self._time_window_violation(
                        stop
                    ),
                    str(stop.customer_id),
                )
            )

            for stop in violated_stops:
                source_vehicle_id = (
                    vehicle_evaluation.vehicle_id
                )

                destination = (
                    self._best_destination_for_customer(
                        current_plan,
                        source_vehicle_id=(
                            source_vehicle_id
                        ),
                        customer_id=stop.customer_id,
                        reason="time_window",
                    )
                )

                if destination is None:
                    continue

                (
                    destination_vehicle_id,
                    destination_position,
                ) = destination

                candidate_plan = self._move_customer(
                    current_plan,
                    source_vehicle_id=(
                        source_vehicle_id
                    ),
                    customer_id=stop.customer_id,
                    destination_vehicle_id=(
                        destination_vehicle_id
                    ),
                    destination_position=(
                        destination_position
                    ),
                )

                candidate_evaluation = (
                    self.evaluator.evaluate(
                        candidate_plan,
                        planning_time_s=self._planning_time_s,
                    )
                )

                destination_evaluation = next(
                    result
                    for result in (
                        candidate_evaluation
                        .vehicle_evaluations
                    )
                    if result.vehicle_id
                    == destination_vehicle_id
                )

                if not (
                    destination_evaluation.time_window_feasible
                ):
                    continue

                current_plan = candidate_plan

                actions.append(
                    RepairAction(
                        action="relocate_customer",
                        customer_id=stop.customer_id,
                        source_vehicle_id=(
                            source_vehicle_id
                        ),
                        destination_vehicle_id=(
                            destination_vehicle_id
                        ),
                        source_position=None,
                        destination_position=(
                            destination_position
                        ),
                        reason=(
                            "repair service-start "
                            "time-window violation"
                        ),
                    )
                )

                changed = True
                break

        return (
            current_plan,
            actions,
            changed,
        )

    # ------------------------------------------------------------------
    # Destination search
    # ------------------------------------------------------------------

    def _best_destination_for_customer(
        self,
        route_plan: RoutePlan,
        *,
        source_vehicle_id: VehicleId,
        customer_id: CustomerId,
        reason: str,
    ) -> tuple[
        VehicleId,
        int,
    ] | None:
        """Find the best deterministic destination insertion."""

        if not self._allow_cross_vehicle_relocation:
            return None

        candidates: list[
            tuple[
                float,
                int,
                str,
                str,
                VehicleId,
                int,
            ]
        ] = []

        for route in route_plan.vehicle_routes:
            destination_vehicle_id = (
                route.vehicle_id
            )

            if destination_vehicle_id == source_vehicle_id:
                continue

            for position in range(
                len(route.customer_ids) + 1
            ):
                candidate_plan = self._move_customer(
                    route_plan,
                    source_vehicle_id=(
                        source_vehicle_id
                    ),
                    customer_id=customer_id,
                    destination_vehicle_id=(
                        destination_vehicle_id
                    ),
                    destination_position=position,
                )

                evaluation = self.evaluator.evaluate(
                    candidate_plan,
                    planning_time_s=self._planning_time_s,
                )

                destination_evaluation = next(
                    result
                    for result in (
                        evaluation.vehicle_evaluations
                    )
                    if result.vehicle_id
                    == destination_vehicle_id
                )

                source_evaluation = next(
                    result
                    for result in (
                        evaluation.vehicle_evaluations
                    )
                    if result.vehicle_id
                    == source_vehicle_id
                )

                if not destination_evaluation.connectivity_feasible:
                    continue

                if reason == "capacity":
                    if not destination_evaluation.capacity_feasible:
                        continue

                if reason == "time_window":
                    if not destination_evaluation.time_window_feasible:
                        continue

                if reason == "connectivity":
                    if not source_evaluation.connectivity_feasible:
                        continue

                score = self._destination_score(
                    destination_evaluation
                )

                candidates.append(
                    (
                        score,
                        position,
                        str(destination_vehicle_id),
                        str(customer_id),
                        destination_vehicle_id,
                        position,
                    )
                )

        if not candidates:
            return None

        selected = min(candidates)

        return (
            selected[4],
            selected[5],
        )

    def _destination_score(
        self,
        vehicle_evaluation: Any,
    ) -> float:
        return float(
            self.config.distance_weight
            * vehicle_evaluation.total_distance_m
            + self.config.travel_time_weight
            * vehicle_evaluation.total_travel_time_s
            + self.config.waiting_time_weight
            * vehicle_evaluation.total_waiting_time_s
            + self.config.lateness_weight
            * vehicle_evaluation.lateness_s
        )

    # ------------------------------------------------------------------
    # Route mutation
    # ------------------------------------------------------------------

    @staticmethod
    def _move_customer(
        route_plan: RoutePlan,
        *,
        source_vehicle_id: VehicleId,
        customer_id: CustomerId,
        destination_vehicle_id: VehicleId,
        destination_position: int,
    ) -> RoutePlan:
        """Move one customer between vehicle routes."""

        new_routes: list[VehicleRoute] = []

        for route in route_plan.vehicle_routes:
            customers = list(
                route.customer_ids
            )

            if route.vehicle_id == source_vehicle_id:
                try:
                    customers.remove(
                        customer_id
                    )
                except ValueError:
                    pass

            new_routes.append(
                VehicleRoute.from_sequence(
                    vehicle_id=route.vehicle_id,
                    customer_ids=customers,
                )
            )

        updated_routes: list[VehicleRoute] = []

        for route in new_routes:
            if (
                route.vehicle_id
                != destination_vehicle_id
            ):
                updated_routes.append(route)
                continue

            customers = list(
                route.customer_ids
            )

            position = max(
                0,
                min(
                    destination_position,
                    len(customers),
                ),
            )

            customers.insert(
                position,
                customer_id,
            )

            updated_routes.append(
                VehicleRoute.from_sequence(
                    vehicle_id=route.vehicle_id,
                    customer_ids=customers,
                )
            )

        return RoutePlan.from_routes(
            updated_routes
        )

    # ------------------------------------------------------------------
    # Constraint helpers
    # ------------------------------------------------------------------

    def _vehicle_route_is_capacity_feasible(
        self,
        route_plan: RoutePlan,
        vehicle_id: VehicleId,
    ) -> bool:
        evaluation = self.evaluator.evaluate(
            route_plan
        )

        vehicle_evaluation = next(
            result
            for result in evaluation.vehicle_evaluations
            if result.vehicle_id == vehicle_id
        )

        return (
            vehicle_evaluation.capacity_feasible
        )

    @staticmethod
    def _time_window_violation(
        stop: Any,
    ) -> float:
        late_amount = max(
            0.0,
            float(stop.service_start_time_s)
            - float(stop.latest_time_s),
        )

        release_amount = max(
            0.0,
            float(stop.release_time_s)
            - float(stop.service_start_time_s),
        )

        return max(
            late_amount,
            release_amount,
        )

    def _unresolved_violations(
        self,
        evaluation: RoutePlanEvaluation,
    ) -> tuple[
        tuple[CustomerId, ...],
        tuple[str, ...],
    ]:
        customer_ids: list[CustomerId] = []
        reasons: list[str] = []

        customer_ids.extend(
            evaluation.unserved_customer_ids
        )

        if evaluation.unserved_customer_ids:
            reasons.append(
                "unserved customers remain"
            )

        if evaluation.duplicate_customer_ids:
            customer_ids.extend(
                evaluation.duplicate_customer_ids
            )

            reasons.append(
                "duplicate customer assignments remain"
            )

        for vehicle_evaluation in (
            evaluation.vehicle_evaluations
        ):
            if not vehicle_evaluation.capacity_feasible:
                customer_ids.extend(vehicle_evaluation.customer_ids)
                reasons.append(
                    f"vehicle "
                    f"{vehicle_evaluation.vehicle_id!r} "
                    "still exceeds capacity"
                )

            if not vehicle_evaluation.connectivity_feasible:
                reasons.append(
                    f"vehicle "
                    f"{vehicle_evaluation.vehicle_id!r} "
                    "still has a connectivity violation"
                )

            if not vehicle_evaluation.time_window_feasible:
                reasons.append(
                    f"vehicle "
                    f"{vehicle_evaluation.vehicle_id!r} "
                    "still has a time-window violation"
                )

            for stop in vehicle_evaluation.stops:
                if not stop.time_window_feasible:
                    customer_ids.append(
                        stop.customer_id
                    )

        return (
            tuple(
                dict.fromkeys(customer_ids)
            ),
            tuple(
                dict.fromkeys(reasons)
            ),
        )

    # ------------------------------------------------------------------
    # Scenario accessors — V1.2 contract
    # ------------------------------------------------------------------

    def _scenario_requests(
        self,
    ) -> Iterable[Any]:
        if not hasattr(
            self.scenario,
            "requests",
        ):
            raise AttributeError(
                "Scenario does not expose requests"
            )

        return self.scenario.requests

    def _scenario_vehicles(
        self,
    ) -> Iterable[Any]:
        if not hasattr(
            self.scenario,
            "fleet",
        ):
            raise AttributeError(
                "Scenario does not expose fleet"
            )

        return self.scenario.fleet

    @staticmethod
    def _customer_id(
        request: Any,
    ) -> CustomerId:
        return request.request_id

    def _request_demand(
        self,
        customer_id: CustomerId,
    ) -> float:
        request = self._request_by_id.get(
            customer_id
        )

        if request is None:
            raise KeyError(
                f"Unknown customer ID: {customer_id!r}"
            )

        return float(request.demand)

    def _request_latest(self, customer_id: CustomerId) -> float:
        request = self._request_by_id.get(customer_id)
        if request is None:
            raise KeyError(f"Unknown customer ID: {customer_id!r}")
        return float(request.latest_service_start_s)

    @staticmethod
    def _vehicle_id(
        vehicle: Any,
    ) -> VehicleId:
        return vehicle.vehicle_id