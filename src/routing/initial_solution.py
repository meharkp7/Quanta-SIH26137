"""Constructive initial-route generation for Step 4.

The initial solution is the first routing layer before GNN, Transformer,
DRL and QPSO.

It creates a logical RoutePlan only. Physical road realization and feasibility
truth remain the responsibility of RouteEvaluator.

Supported construction orders
-----------------------------
- earliest_deadline
- earliest_release
- largest_demand
- customer_id
- nearest_feasible

The implementation is deliberately generic:
- no fixed customer limit;
- no fixed vehicle limit;
- no dependency on the Step 3 fixture;
- uses the scenario's actual fleet and requests;
- uses the same evaluator as downstream optimization;
- deterministic tie-breaking;
- can leave requests unassigned when no feasible insertion exists.

This is a constructive baseline, not an optimizer and not a claim of
optimality.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.contracts.core_types import (
    CustomerId,
    DemandUnits,
    TimeS,
    VehicleId,
)
from src.routing.route_evaluator import (
    RouteEvaluator,
    RoutePlanEvaluation,
    VehicleRouteEvaluation,
)
from src.routing.route_plan import RoutePlan, VehicleRoute


@dataclass(frozen=True)
class InitialSolutionConfig:
    """Configuration for constructive initial-route generation."""

    heuristic: str = "earliest_deadline"
    # Public alias retained for the Step 4 API/tests. When supplied, this
    # overrides ``heuristic`` for customer ordering.
    customer_ordering: str | None = None

    allow_unassigned: bool = True

    prioritize_time_windows: bool = True
    prioritize_distance: bool = True
    prioritize_travel_time: bool = True
    prioritize_waiting_time: bool = True

    require_capacity_feasibility: bool = True
    require_connectivity: bool = True
    require_time_window_feasibility: bool = True

    # None means evaluate every insertion candidate.
    max_candidates_per_insertion: int | None = None

    # Bounded scanning for large real-city networks (e.g. the ~2000-edge
    # Delhi OSM graphs). When set, one insertion round stops evaluating
    # further (customer, vehicle, position) candidates once this many
    # FEASIBLE candidates have been found. Selection still takes the
    # best-scoring candidate among those seen. None (default) keeps the
    # exhaustive scan used by the fixtures and tests.
    feasible_candidates_per_round: int | None = None

    # Hard cap on evaluated trials in one insertion round regardless of
    # feasibility (None = unbounded). Protects large graphs from long
    # scans when feasible insertions are scarce.
    max_trials_per_round: int | None = None

    # Planning snapshot. Requests whose release is later than this time are
    # not inserted unless explicitly allowed.
    planning_time_s: TimeS = 0.0

    include_future_released_requests: bool = False


@dataclass(frozen=True)
class InsertionCandidate:
    """One possible customer insertion."""

    vehicle_id: VehicleId
    customer_id: CustomerId
    position: int

    score: float
    feasible: bool

    evaluation: RoutePlanEvaluation
    vehicle_evaluation: VehicleRouteEvaluation


@dataclass(frozen=True)
class InitialSolutionResult:
    """Complete output of the constructive initial solver."""

    route_plan: RoutePlan
    evaluation: RoutePlanEvaluation

    assigned_customer_ids: tuple[CustomerId, ...]
    unassigned_customer_ids: tuple[CustomerId, ...]

    feasible: bool
    complete: bool

    construction_method: str
    construction_trace: tuple[str, ...]
    insertion_count: int = 0
    attempt_count: int = 0

    @property
    def assigned_request_ids(self) -> tuple[CustomerId, ...]:
        return self.assigned_customer_ids

    @property
    def unassigned_request_ids(self) -> tuple[CustomerId, ...]:
        return self.unassigned_customer_ids

    @property
    def errors(self) -> tuple[str, ...]:
        return self.evaluation.errors


class InitialSolutionBuilder:
    """Construct a deterministic feasible starting RoutePlan."""

    def __init__(
        self,
        scenario: Any,
        evaluator: RouteEvaluator,
        *,
        config: InitialSolutionConfig | None = None,
    ) -> None:
        self.scenario = scenario
        self.evaluator = evaluator
        self.config = config or InitialSolutionConfig()

        if self.config.max_candidates_per_insertion is not None:
            if self.config.max_candidates_per_insertion < 1:
                raise ValueError(
                    "max_candidates_per_insertion must be at least 1 or None"
                )

        for name, value in (
            ("feasible_candidates_per_round", self.config.feasible_candidates_per_round),
            ("max_trials_per_round", self.config.max_trials_per_round),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1 or None")

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        planning_time_s: TimeS | None = None,
        request_ids: Iterable[CustomerId] | None = None,
    ) -> InitialSolutionResult:
        """Build the initial route plan for a planning snapshot.

        ``planning_time_s`` overrides the configuration snapshot for this
        invocation only. The builder remains stateless across calls.
        """
        effective_planning_time_s = (
            self.config.planning_time_s
            if planning_time_s is None
            else float(planning_time_s)
        )

        routes = tuple(
            VehicleRoute.from_sequence(
                vehicle_id=self._vehicle_id(vehicle),
                customer_ids=(),
            )
            for vehicle in self._vehicles
        )

        route_plan = RoutePlan.from_routes(
            routes
        )

        eligible_ids = self._eligible_customer_ids(
            planning_time_s=effective_planning_time_s
        )
        if request_ids is None:
            target_ids = eligible_ids
        else:
            target_ids = tuple(request_ids)
            unknown = tuple(
                customer_id for customer_id in target_ids
                if customer_id not in self._request_by_id
            )
            if unknown:
                raise ValueError(
                    "Unknown request IDs: " + ", ".join(map(str, unknown))
                )
            if len(target_ids) != len(set(target_ids)):
                raise ValueError("request_ids must not contain duplicates")
            not_visible = tuple(
                customer_id for customer_id in target_ids
                if customer_id not in set(eligible_ids)
                and not self.config.include_future_released_requests
            )
            if not_visible:
                raise ValueError(
                    "Requested IDs are not released/visible at planning time: "
                    + ", ".join(map(str, not_visible))
                )
        remaining = list(target_ids)

        trace: list[str] = []
        self._last_attempt_count = 0

        while remaining:
            candidates = self._generate_candidates(
                route_plan,
                remaining,
                planning_time_s=effective_planning_time_s,
            )

            feasible_candidates = [
                candidate
                for candidate in candidates
                if candidate.feasible
            ]

            if not feasible_candidates:
                break

            selected = min(
                feasible_candidates,
                key=self._candidate_sort_key,
            )

            route_plan = self._apply_candidate(
                route_plan,
                selected,
            )

            remaining.remove(
                selected.customer_id
            )

            trace.append(
                "assigned "
                f"{selected.customer_id!r} "
                f"to vehicle "
                f"{selected.vehicle_id!r} "
                f"at position "
                f"{selected.position}"
            )

        assigned = route_plan.unique_customer_ids()
        assigned_set = set(assigned)

        eligible_set = set(target_ids)
        eligible_ids = tuple(target_ids)

        unassigned = tuple(
            customer_id
            for customer_id in eligible_ids
            if customer_id not in assigned_set
        )

        future_ids = tuple(
            customer_id
            for customer_id in self._request_by_id
            if customer_id not in assigned_set
            and customer_id not in eligible_set
        )

        evaluation = self.evaluator.evaluate(
            route_plan
        )

        complete = (
            not unassigned
            and not future_ids
            and len(assigned_set) == len(target_ids)
        )

        return InitialSolutionResult(
            route_plan=route_plan,
            evaluation=evaluation,
            assigned_customer_ids=assigned,
            unassigned_customer_ids=unassigned,
            feasible=evaluation.feasible,
            complete=complete,
            construction_method=self.config.heuristic,
            construction_trace=tuple(trace),
            insertion_count=len(trace),
            attempt_count=self._last_attempt_count,
        )

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        route_plan: RoutePlan,
        remaining: Iterable[CustomerId],
        *,
        planning_time_s: TimeS = 0.0,
    ) -> list[InsertionCandidate]:
        candidates: list[InsertionCandidate] = []
        target = self.config.feasible_candidates_per_round
        trial_cap = self.config.max_trials_per_round
        bounded = target is not None or trial_cap is not None
        feasible_seen = 0
        trials = 0

        ordered_customers = self._order_customers(
            remaining
        )

        stop = False
        for customer_id in ordered_customers:
            if stop:
                break
            for vehicle_route in route_plan.vehicle_routes:
                if stop:
                    break
                positions = range(
                    len(vehicle_route.customer_ids) + 1
                )

                for position in positions:
                    candidate_plan = self._insert_customer(
                        route_plan=route_plan,
                        vehicle_id=vehicle_route.vehicle_id,
                        customer_id=customer_id,
                        position=position,
                    )

                    self._last_attempt_count += 1
                    trials += 1
                    evaluation = self.evaluator.evaluate(
                        candidate_plan,
                        planning_time_s=planning_time_s,
                    )

                    vehicle_evaluation = (
                        self._vehicle_evaluation_for(
                            evaluation,
                            vehicle_route.vehicle_id,
                        )
                    )

                    feasible = self._candidate_is_feasible(
                        vehicle_evaluation
                    )

                    score = self._candidate_score(
                        evaluation=evaluation,
                        vehicle_evaluation=vehicle_evaluation,
                    )

                    candidates.append(
                        InsertionCandidate(
                            vehicle_id=vehicle_route.vehicle_id,
                            customer_id=customer_id,
                            position=position,
                            score=float(score),
                            feasible=feasible,
                            evaluation=evaluation,
                            vehicle_evaluation=vehicle_evaluation,
                        )
                    )

                    if bounded:
                        if feasible:
                            feasible_seen += 1
                        if (
                            target is not None
                            and feasible_seen >= target
                        ):
                            stop = True
                            break
                        if (
                            trial_cap is not None
                            and trials >= trial_cap
                        ):
                            stop = True
                            break

        if self.config.max_candidates_per_insertion is not None:
            candidates.sort(
                key=self._candidate_sort_key
            )

            return candidates[
                : self.config.max_candidates_per_insertion
            ]

        return candidates

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def _candidate_is_feasible(
        self,
        vehicle_evaluation: VehicleRouteEvaluation,
    ) -> bool:
        if self.config.require_capacity_feasibility:
            if not vehicle_evaluation.capacity_feasible:
                return False

        if self.config.require_connectivity:
            if not vehicle_evaluation.connectivity_feasible:
                return False

        if self.config.require_time_window_feasibility:
            if not vehicle_evaluation.time_window_feasible:
                return False

        return True

    # ------------------------------------------------------------------
    # Customer ordering
    # ------------------------------------------------------------------

    def _order_customers(
        self,
        customer_ids: Iterable[CustomerId],
    ) -> tuple[CustomerId, ...]:
        requests = [
            self._request_by_id[customer_id]
            for customer_id in customer_ids
        ]

        heuristic = (self.config.customer_ordering or self.config.heuristic).lower()

        if heuristic in {
            "earliest_deadline",
            "deadline",
            "vrptw",
        }:
            requests.sort(
                key=lambda request: (
                    float(
                        self._request_latest(request)
                    ),
                    float(
                        self._request_earliest(request)
                    ),
                    str(
                        self._request_id(request)
                    ),
                )
            )

        elif heuristic in {
            "earliest_release",
            "release",
        }:
            requests.sort(
                key=lambda request: (
                    float(
                        self._request_release(request)
                    ),
                    float(
                        self._request_latest(request)
                    ),
                    str(
                        self._request_id(request)
                    ),
                )
            )

        elif heuristic in {
            "largest_demand",
            "demand",
        }:
            requests.sort(
                key=lambda request: (
                    -float(
                        self._request_demand(request)
                    ),
                    float(
                        self._request_latest(request)
                    ),
                    str(
                        self._request_id(request)
                    ),
                )
            )

        elif heuristic in {
            "customer_id",
            "deterministic",
        }:
            requests.sort(
                key=lambda request: str(
                    self._request_id(request)
                )
            )

        elif heuristic in {
            "nearest_feasible",
        }:
            # The actual distance-sensitive insertion score determines the
            # final placement. Customer ordering remains deterministic.
            requests.sort(
                key=lambda request: (
                    float(
                        self._request_earliest(request)
                    ),
                    str(
                        self._request_id(request)
                    ),
                )
            )

        else:
            raise ValueError(
                "Unknown initial-solution heuristic: "
                f"{self.config.heuristic!r}"
            )

        return tuple(
            self._request_id(request)
            for request in requests
        )

    # ------------------------------------------------------------------
    # Candidate scoring
    # ------------------------------------------------------------------

    def _candidate_score(
        self,
        *,
        evaluation: RoutePlanEvaluation,
        vehicle_evaluation: VehicleRouteEvaluation,
    ) -> float:
        """Compute deterministic insertion cost."""

        score = 0.0

        if self.config.prioritize_distance:
            score += (
                vehicle_evaluation.total_distance_m
            )

        if self.config.prioritize_travel_time:
            score += (
                vehicle_evaluation.total_travel_time_s
            )

        if self.config.prioritize_waiting_time:
            score += (
                vehicle_evaluation.total_waiting_time_s
            )

        if self.config.prioritize_time_windows:
            score += (
                vehicle_evaluation.lateness_s
                * self.evaluator.config.lateness_penalty
            )

        # Small global term prevents insertion from considering only the
        # currently modified vehicle.
        score += (
            0.01
            * evaluation.objective_value
        )

        return float(score)

    @staticmethod
    def _candidate_sort_key(
        candidate: InsertionCandidate,
    ) -> tuple[
        float,
        str,
        str,
        int,
    ]:
        return (
            candidate.score,
            str(candidate.vehicle_id),
            str(candidate.customer_id),
            candidate.position,
        )

    # ------------------------------------------------------------------
    # Route manipulation
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_customer(
        *,
        route_plan: RoutePlan,
        vehicle_id: VehicleId,
        customer_id: CustomerId,
        position: int,
    ) -> RoutePlan:
        new_routes: list[VehicleRoute] = []

        for route in route_plan.vehicle_routes:
            if route.vehicle_id != vehicle_id:
                new_routes.append(route)
                continue

            customer_ids = list(
                route.customer_ids
            )

            customer_ids.insert(
                position,
                customer_id,
            )

            new_routes.append(
                VehicleRoute.from_sequence(
                    vehicle_id=vehicle_id,
                    customer_ids=customer_ids,
                )
            )

        return RoutePlan.from_routes(
            new_routes
        )

    @staticmethod
    def _apply_candidate(
        route_plan: RoutePlan,
        candidate: InsertionCandidate,
    ) -> RoutePlan:
        return InitialSolutionBuilder._insert_customer(
            route_plan=route_plan,
            vehicle_id=candidate.vehicle_id,
            customer_id=candidate.customer_id,
            position=candidate.position,
        )

    # ------------------------------------------------------------------
    # Scenario filtering
    # ------------------------------------------------------------------

    def _eligible_customer_ids(
        self,
        *,
        planning_time_s: TimeS | None = None,
    ) -> tuple[CustomerId, ...]:
        """Return requests visible/released for this planning snapshot."""

        effective_planning_time_s = (
            self.config.planning_time_s
            if planning_time_s is None
            else float(planning_time_s)
        )

        result: list[CustomerId] = []

        for request in self._requests:
            customer_id = self._request_id(
                request
            )

            if self.config.include_future_released_requests:
                result.append(customer_id)
                continue

            release_time = self._request_release(
                request
            )

            if release_time <= effective_planning_time_s:
                result.append(customer_id)

        return tuple(result)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vehicle_evaluation_for(
        evaluation: RoutePlanEvaluation,
        vehicle_id: VehicleId,
    ) -> VehicleRouteEvaluation:
        for vehicle_evaluation in (
            evaluation.vehicle_evaluations
        ):
            if (
                vehicle_evaluation.vehicle_id
                == vehicle_id
            ):
                return vehicle_evaluation

        raise KeyError(
            "Vehicle evaluation not found for "
            f"vehicle_id={vehicle_id!r}"
        )

    # ------------------------------------------------------------------
    # Scenario accessors
    # ------------------------------------------------------------------

    def _scenario_requests(
        self,
    ) -> Iterable[Any]:
        if not hasattr(self.scenario, "requests"):
            raise AttributeError(
                "Scenario does not expose requests"
            )

        return self.scenario.requests

    def _scenario_vehicles(
        self,
    ) -> Iterable[Any]:
        if not hasattr(self.scenario, "fleet"):
            raise AttributeError(
                "Scenario does not expose fleet"
            )

        return self.scenario.fleet

    # ------------------------------------------------------------------
    # V1.2 request accessors
    # ------------------------------------------------------------------

    @staticmethod
    def _request_id(
        request: Any,
    ) -> CustomerId:
        return request.request_id

    @staticmethod
    def _request_demand(
        request: Any,
    ) -> DemandUnits:
        return request.demand

    @staticmethod
    def _request_release(
        request: Any,
    ) -> TimeS:
        return request.release_s

    @staticmethod
    def _request_earliest(
        request: Any,
    ) -> TimeS:
        return request.earliest_service_start_s

    @staticmethod
    def _request_latest(
        request: Any,
    ) -> TimeS:
        return request.latest_service_start_s

    # ------------------------------------------------------------------
    # V1.2 vehicle accessors
    # ------------------------------------------------------------------

    @staticmethod
    def _vehicle_id(
        vehicle: Any,
    ) -> VehicleId:
        return vehicle.vehicle_id