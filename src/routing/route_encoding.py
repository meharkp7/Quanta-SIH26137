"""Continuous-key route encoding and bounded feasibility repair for Step 7.

The module is deliberately independent of any particle optimizer.  It defines
only the deterministic mapping between a continuous 2n-dimensional key and a
logical :class:`RoutePlan`, plus a bounded repair operator that delegates
feasibility truth to :class:`RouteEvaluator`.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from time import monotonic
from typing import Any, Iterable, Sequence

from src.routing.evaluator_state import CommitmentSnapshot, VehicleCommitment
from src.routing.route_evaluator import RouteEvaluator, RoutePlanEvaluation
from src.routing.route_plan import RoutePlan, VehicleRoute
from src.contracts.core_types import CustomerId, VehicleId


class RouteEncodingError(ValueError):
    """Raised when a continuous key vector cannot be decoded or encoded."""


@dataclass(frozen=True)
class RouteEncodingConfig:
    """Policy for the Step 7 continuous-key representation.

    Assignment keys are interpreted using equal-width fleet bins over
    ``[0, 1]``.  Values below 0 or above 1 are clipped.  The right boundary
    ``1.0`` belongs to the final vehicle.  Within a vehicle, lower order keys
    are visited first; exact ties are resolved by customer ID.
    """

    clip_keys_to_unit_interval: bool = True
    tie_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.tie_tolerance < 0 or not isfinite(self.tie_tolerance):
            raise ValueError("tie_tolerance must be finite and non-negative")


@dataclass(frozen=True)
class RouteEncoding:
    """Immutable encoded candidate and its deterministic decoded metadata."""

    keys: tuple[float, ...]
    assignment_preferences: tuple[float, ...]
    order_priorities: tuple[float, ...]
    customer_ids: tuple[CustomerId, ...]
    assignments: tuple[tuple[CustomerId, VehicleId], ...]
    route_plan: RoutePlan

    @property
    def dimension(self) -> int:
        return len(self.keys)

    def assignment_for(self, customer_id: CustomerId) -> VehicleId:
        for cid, vehicle_id in self.assignments:
            if cid == customer_id:
                return vehicle_id
        raise KeyError(customer_id)


class ContinuousRouteEncoder:
    """Encode/decode logical routes using the Step 7 ``2n`` representation."""

    def __init__(
        self,
        scenario: Any,
        *,
        config: RouteEncodingConfig | None = None,
    ) -> None:
        self.scenario = scenario
        self.config = config or RouteEncodingConfig()
        self._requests = tuple(scenario.requests)
        self._vehicles = tuple(scenario.fleet)
        self._customer_ids = tuple(r.request_id for r in self._requests)
        self._vehicle_ids = tuple(v.vehicle_id for v in self._vehicles)

        if not self._vehicles:
            raise ValueError("Step 7 requires at least one vehicle")
        if len(set(self._customer_ids)) != len(self._customer_ids):
            raise ValueError("Scenario contains duplicate request IDs")
        if len(set(self._vehicle_ids)) != len(self._vehicle_ids):
            raise ValueError("Scenario contains duplicate vehicle IDs")

        self._customer_set = set(self._customer_ids)
        self._vehicle_set = set(self._vehicle_ids)

        # Per-repair-call memoization. A candidate RoutePlan can be generated
        # more than once while exploring the bounded repair neighborhood.
        # Cache scope is intentionally limited to one repair call so no
        # evaluation can survive a planning/network-state transition.
        self._evaluation_cache: dict[tuple[Any, ...], RoutePlanEvaluation] = {}
        self._evaluation_cache_hits = 0
        self._evaluation_calls = 0
        self._customer_index = {cid: i for i, cid in enumerate(self._customer_ids)}

    @property
    def customer_ids(self) -> tuple[CustomerId, ...]:
        return self._customer_ids

    @property
    def vehicle_ids(self) -> tuple[VehicleId, ...]:
        return self._vehicle_ids

    @property
    def dimension(self) -> int:
        return 2 * len(self._customer_ids)

    def decode(
        self,
        keys: Sequence[float] | Iterable[float],
        *,
        commitments: CommitmentSnapshot | None = None,
    ) -> RouteEncoding:
        values = tuple(float(x) for x in keys)
        self._validate_keys(values)
        n = len(self._customer_ids)
        assignment_keys = tuple(self._normalise(value) for value in values[:n])
        order_keys = tuple(self._normalise(value) for value in values[n:])

        locked_by_customer = self._locked_assignments(commitments)
        assignments: dict[CustomerId, VehicleId] = {}
        for i, cid in enumerate(self._customer_ids):
            locked_vehicle = locked_by_customer.get(cid)
            if locked_vehicle is not None:
                assignments[cid] = locked_vehicle
            else:
                assignments[cid] = self._decode_vehicle(assignment_keys[i])

        route_members: dict[VehicleId, list[tuple[float, CustomerId, int]]] = {
            vehicle_id: [] for vehicle_id in self._vehicle_ids
        }
        for i, cid in enumerate(self._customer_ids):
            route_members[assignments[cid]].append((order_keys[i], cid, i))

        locked_prefixes = self._locked_prefixes(commitments)
        routes: list[VehicleRoute] = []
        for vehicle_id in self._vehicle_ids:
            members = route_members[vehicle_id]
            locked = locked_prefixes.get(vehicle_id, ())
            locked_set = set(locked)

            missing_locked = [cid for cid in locked if cid not in {x[1] for x in members}]
            if missing_locked:
                raise RouteEncodingError(
                    f"locked customers {tuple(missing_locked)!r} cannot be represented "
                    f"on vehicle {vehicle_id!r}"
                )

            mutable = [x for x in members if x[1] not in locked_set]
            mutable.sort(key=self._order_sort_key)
            ordered = tuple(locked) + tuple(x[1] for x in mutable)
            routes.append(VehicleRoute.from_sequence(vehicle_id, ordered))

        plan = RoutePlan.from_routes(routes)
        return RouteEncoding(
            keys=values,
            assignment_preferences=assignment_keys,
            order_priorities=order_keys,
            customer_ids=self._customer_ids,
            assignments=tuple((cid, assignments[cid]) for cid in self._customer_ids),
            route_plan=plan,
        )

    def _order_sort_key(
        self,
        item: tuple[float, CustomerId, int],
    ) -> tuple[int | float, str, int]:
        """Return the deterministic order key used by decoding.

        ``tie_tolerance`` is implemented as a quantisation policy rather than
        as a pairwise approximate comparison.  Pairwise ``abs(a-b) <= tol``
        comparisons are not transitive and can therefore make sorting
        implementation-dependent.  Quantisation gives us a stable, total
        ordering while still treating priorities within the configured
        tolerance bucket as ties.
        """
        priority, customer_id, original_index = item
        tolerance = self.config.tie_tolerance
        if tolerance == 0.0:
            return priority, str(customer_id), original_index
        bucket = floor(priority / tolerance + 0.5)
        return bucket, str(customer_id), original_index

    def encode(
        self,
        route_plan: RoutePlan,
        *,
        commitments: CommitmentSnapshot | None = None,
    ) -> RouteEncoding:
        """Encode a logical plan into a canonical ``2n`` key vector.

        The encoding is canonical: decoding the returned keys under the same
        commitment snapshot must reproduce ``route_plan`` exactly.  In
        particular, committed customer prefixes are validated rather than
        silently reordered during encoding.
        """
        routes = {route.vehicle_id: route for route in route_plan.vehicle_routes}
        unknown_vehicles = set(routes) - self._vehicle_set
        if unknown_vehicles:
            raise RouteEncodingError(
                "route plan contains unknown vehicles: "
                f"{sorted(unknown_vehicles, key=str)}"
            )
        if set(routes) != self._vehicle_set:
            missing = self._vehicle_set - set(routes)
            raise RouteEncodingError(
                "route plan is missing vehicles: "
                f"{sorted(missing, key=str)}"
            )

        flattened = [cid for route in route_plan.vehicle_routes for cid in route.customer_ids]
        unknown_customers = set(flattened) - self._customer_set
        if unknown_customers:
            raise RouteEncodingError(
                "route plan contains unknown customers: "
                f"{sorted(unknown_customers, key=str)}"
            )
        if len(flattened) != len(self._customer_ids) or len(set(flattened)) != len(flattened):
            raise RouteEncodingError(
                "route plan must contain every scenario customer exactly once"
            )

        locked_prefixes = self._locked_prefixes(commitments)
        locked_by_customer = self._locked_assignments(commitments)
        for vehicle_id, expected_prefix in locked_prefixes.items():
            actual = routes[vehicle_id].customer_ids
            prefix = actual[: len(expected_prefix)]
            if prefix != expected_prefix:
                raise RouteEncodingError(
                    f"vehicle {vehicle_id!r} does not preserve the committed "
                    f"customer prefix: expected={expected_prefix!r}, actual={prefix!r}"
                )

        assignment_keys: list[float] = []
        order_keys: list[float] = []
        assignments: list[tuple[CustomerId, VehicleId]] = []

        vehicle_index = {vid: i for i, vid in enumerate(self._vehicle_ids)}
        n = len(self._customer_ids)
        route_by_customer: dict[CustomerId, tuple[VehicleId, VehicleRoute]] = {}
        for route in route_plan.vehicle_routes:
            for cid in route.customer_ids:
                route_by_customer[cid] = (route.vehicle_id, route)

        for cid in self._customer_ids:
            assigned_vehicle, route = route_by_customer[cid]
            locked_vehicle = locked_by_customer.get(cid)
            if locked_vehicle is not None and assigned_vehicle != locked_vehicle:
                raise RouteEncodingError(
                    f"locked customer {cid!r} is assigned to {assigned_vehicle!r}; "
                    f"commitment requires {locked_vehicle!r}"
                )

            idx = vehicle_index[assigned_vehicle]
            assignment_keys.append((idx + 0.5) / len(self._vehicle_ids))

            pos = route.customer_ids.index(cid)
            route_length = len(route.customer_ids)
            order_keys.append((pos + 0.5) / max(1, route_length))
            assignments.append((cid, assigned_vehicle))

        keys = tuple(assignment_keys + order_keys)
        encoded = RouteEncoding(
            keys=keys,
            assignment_preferences=tuple(assignment_keys),
            order_priorities=tuple(order_keys),
            customer_ids=self._customer_ids,
            assignments=tuple(assignments),
            route_plan=route_plan,
        )

        # This is an intentional invariant check, not merely a test helper.
        # A future change to the decoder or boundary policy must never make
        # canonical particle coordinates represent a different route.
        decoded = self.decode(keys, commitments=commitments)
        if decoded.route_plan != route_plan:
            raise RouteEncodingError(
                "canonical encoding invariant failed: decode(encode(route_plan)) "
                "did not reproduce the supplied route plan"
            )
        return encoded

    def _validate_keys(self, keys: tuple[float, ...]) -> None:
        if len(keys) != self.dimension:
            raise RouteEncodingError(
                f"expected exactly {self.dimension} keys for {len(self._customer_ids)} customers; "
                f"received {len(keys)}"
            )
        for value in keys:
            if not isfinite(value):
                raise RouteEncodingError(f"all route keys must be finite; got {value!r}")

    def _normalise(self, value: float) -> float:
        if self.config.clip_keys_to_unit_interval:
            return min(1.0, max(0.0, value))
        if value < 0.0 or value > 1.0:
            raise RouteEncodingError(f"route key {value!r} is outside [0, 1]")
        return value

    def _decode_vehicle(self, value: float) -> VehicleId:
        x = self._normalise(value)
        index = min(len(self._vehicle_ids) - 1, floor(x * len(self._vehicle_ids)))
        return self._vehicle_ids[index]

    @staticmethod
    def _locked_prefixes(commitments: CommitmentSnapshot | None) -> dict[VehicleId, tuple[CustomerId, ...]]:
        if commitments is None:
            return {}
        result: dict[VehicleId, tuple[CustomerId, ...]] = {}
        for commitment in commitments.vehicles:
            prefix = tuple(commitment.onboard_request_ids) + tuple(commitment.committed_customer_ids)
            result[commitment.vehicle_id] = prefix
        return result

    def _locked_assignments(self, commitments: CommitmentSnapshot | None) -> dict[CustomerId, VehicleId]:
        result: dict[CustomerId, VehicleId] = {}
        for vehicle_id, prefix in self._locked_prefixes(commitments).items():
            if vehicle_id not in self._vehicle_set:
                raise RouteEncodingError(f"commitment references unknown vehicle {vehicle_id!r}")
            for cid in prefix:
                if cid not in self._customer_set:
                    raise RouteEncodingError(f"commitment references unknown customer {cid!r}")
                previous = result.get(cid)
                if previous is not None and previous != vehicle_id:
                    raise RouteEncodingError(f"customer {cid!r} is locked to multiple vehicles")
                result[cid] = vehicle_id
        return result


@dataclass(frozen=True)
class RepairConfig:
    """Hard bounds and deterministic policy for Step 7 repair."""

    max_attempts: int = 250
    max_wall_time_s: float | None = None
    max_moves: int = 100
    require_all_requests_served: bool = True
    objective_tie_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.max_moves < 1:
            raise ValueError("max_moves must be at least 1")
        if self.max_wall_time_s is not None and (self.max_wall_time_s <= 0 or not isfinite(self.max_wall_time_s)):
            raise ValueError("max_wall_time_s must be finite and positive when provided")
        if self.objective_tie_tolerance < 0 or not isfinite(self.objective_tie_tolerance):
            raise ValueError("objective_tie_tolerance must be finite and non-negative")


@dataclass(frozen=True)
class RepairAction:
    customer_id: CustomerId
    source_vehicle_id: VehicleId
    destination_vehicle_id: VehicleId
    source_position: int
    destination_position: int
    reason: str


@dataclass(frozen=True)
class BoundedRepairResult:
    original_route_plan: RoutePlan
    repaired_route_plan: RoutePlan
    original_evaluation: RoutePlanEvaluation
    repaired_evaluation: RoutePlanEvaluation
    actions: tuple[RepairAction, ...]
    feasible: bool
    complete: bool
    attempts_used: int
    moves_used: int
    elapsed_time_s: float
    attempt_limit_reached: bool
    deadline_reached: bool
    unresolved_customer_ids: tuple[CustomerId, ...]
    failure_reason: str | None
    evaluation_calls: int
    evaluation_cache_hits: int


class BoundedRouteRepairer:
    """Repair decoded particle candidates without violating commitments."""

    def __init__(self, scenario: Any, evaluator: RouteEvaluator, *, config: RepairConfig | None = None) -> None:
        self.scenario = scenario
        self.evaluator = evaluator
        self.config = config or RepairConfig()
        self._customer_ids = tuple(r.request_id for r in scenario.requests)
        self._customer_set = set(self._customer_ids)
        self._vehicle_ids = tuple(v.vehicle_id for v in scenario.fleet)
        self._vehicle_set = set(self._vehicle_ids)

        # Per-repair-call memoization. A candidate RoutePlan can be generated
        # more than once while exploring the bounded repair neighborhood.
        # Cache scope is intentionally limited to one repair call so no
        # evaluation can survive a planning/network-state transition.
        self._evaluation_cache: dict[tuple[Any, ...], RoutePlanEvaluation] = {}
        self._evaluation_cache_hits = 0
        self._evaluation_calls = 0

    def repair(
        self,
        route_plan: RoutePlan,
        *,
        commitments: CommitmentSnapshot | None = None,
        planning_time_s: float = 0.0,
    ) -> BoundedRepairResult:
        started = monotonic()
        self._evaluation_cache.clear()
        self._evaluation_cache_hits = 0
        self._evaluation_calls = 0
        original = self._evaluate_cached(
            route_plan, commitments=commitments, planning_time_s=planning_time_s
        )
        current = route_plan
        current_eval = original
        actions: list[RepairAction] = []
        attempts = 0
        moves = 0
        attempt_limit = False
        deadline = False
        locked = self._locked_customers(commitments)

        if self._success(current_eval):
            return self._result(original, current, current_eval, actions, attempts, moves, started, False, False, ())

        for _ in range(self.config.max_moves):
            if attempts >= self.config.max_attempts:
                attempt_limit = True
                break
            if self._deadline_reached(started):
                deadline = True
                break

            conflict_customers = self._conflicting_mutable_customers(current_eval, current, locked)
            if not conflict_customers:
                break

            best: tuple[tuple[float, int, int, str, str, int], RoutePlan, RoutePlanEvaluation, RepairAction] | None = None
            for customer_id, source_vehicle_id, source_position, reason in conflict_customers:
                if self._budget_exhausted(started, attempts):
                    attempt_limit = attempts >= self.config.max_attempts
                    deadline = self._deadline_reached(started)
                    break
                for destination_vehicle_id in self._vehicle_ids:
                    source_route = current.route_for(source_vehicle_id)
                    destination_route = current.route_for(destination_vehicle_id)
                    for position in self._eligible_positions(destination_route, destination_vehicle_id, locked, moving_customer=customer_id, source_vehicle_id=source_vehicle_id):
                        if self._budget_exhausted(started, attempts):
                            attempt_limit = attempts >= self.config.max_attempts
                            deadline = self._deadline_reached(started)
                            break
                        attempts += 1
                        candidate = self._move(current, source_vehicle_id, customer_id, destination_vehicle_id, position)
                        evaluation = self._evaluate_cached(
                            candidate, commitments=commitments, planning_time_s=planning_time_s
                        )
                        score = self._score(evaluation)
                        action = RepairAction(customer_id, source_vehicle_id, destination_vehicle_id, source_position, position, reason)
                        rank = (0.0 if evaluation.feasible else 1.0, score[0], score[1], str(destination_vehicle_id), str(customer_id), position)
                        if best is None or rank < best[0]:
                            best = (rank, candidate, evaluation, action)
                        if evaluation.feasible:
                            break
                    if best is not None and best[2].feasible:
                        break
                if best is not None and best[2].feasible:
                    break

            if best is None:
                break

            _, candidate, candidate_eval, action = best
            if not candidate_eval.feasible and self._score(candidate_eval) >= self._score(current_eval):
                break

            current = candidate
            current_eval = candidate_eval
            actions.append(action)
            moves += 1
            if current_eval.feasible:
                break

        unresolved = self._unresolved(current_eval, locked)
        failure = None if current_eval.feasible else self._failure_reason(current_eval, attempts, attempt_limit, deadline)
        return self._result(original, current, current_eval, actions, attempts, moves, started, attempt_limit, deadline, unresolved, failure)

    def _evaluate_cached(
        self,
        route_plan: RoutePlan,
        *,
        commitments: CommitmentSnapshot | None,
        planning_time_s: float,
    ) -> RoutePlanEvaluation:
        """Evaluate a candidate once within the lifetime of one repair call.

        Repair exploration is deterministic and evaluator inputs are immutable
        for a single call, so memoizing by route/commitment/time is safe. The
        cache is cleared at the beginning of every :meth:`repair` call.
        """
        key = (
            tuple(
                (route.vehicle_id, tuple(route.customer_ids))
                for route in route_plan.vehicle_routes
            ),
            self._commitment_cache_key(commitments),
            float(planning_time_s),
        )
        cached = self._evaluation_cache.get(key)
        if cached is not None:
            self._evaluation_cache_hits += 1
            return cached

        evaluation = self.evaluator.evaluate(
            route_plan,
            commitments=commitments,
            planning_time_s=planning_time_s,
        )
        self._evaluation_calls += 1
        self._evaluation_cache[key] = evaluation
        return evaluation

    @staticmethod
    def _commitment_cache_key(
        commitments: CommitmentSnapshot | None,
    ) -> tuple[Any, ...] | None:
        if commitments is None:
            return None
        return tuple(
            (
                commitment.vehicle_id,
                commitment.current_node_id,
                float(commitment.current_time_s),
                float(commitment.current_load_units),
                tuple(commitment.onboard_request_ids),
                tuple(commitment.committed_customer_ids),
                tuple(commitment.frozen_prefix_edge_ids),
            )
            for commitment in commitments.vehicles
        )

    def _success(self, evaluation: RoutePlanEvaluation) -> bool:
        return evaluation.feasible and (evaluation.all_requests_served if self.config.require_all_requests_served else True)

    def _locked_customers(self, commitments: CommitmentSnapshot | None) -> set[CustomerId]:
        if commitments is None:
            return set()
        result: set[CustomerId] = set()
        for commitment in commitments.vehicles:
            result.update(commitment.onboard_request_ids)
            result.update(commitment.committed_customer_ids)
        return result

    def _conflicting_mutable_customers(self, evaluation: RoutePlanEvaluation, plan: RoutePlan, locked: set[CustomerId]) -> list[tuple[CustomerId, VehicleId, int, str]]:
        implicated: list[tuple[CustomerId, VehicleId, int, str, float]] = []
        for vehicle_eval in evaluation.vehicle_evaluations:
            route = plan.route_for(vehicle_eval.vehicle_id)
            for violation in vehicle_eval.violations:
                if violation.customer_id in locked:
                    continue
                if violation.customer_id in route.customer_ids:
                    idx = route.customer_ids.index(violation.customer_id)
                    implicated.append((violation.customer_id, vehicle_eval.vehicle_id, idx, violation.name, float(violation.magnitude)))
            if not vehicle_eval.feasible:
                for idx, cid in enumerate(route.customer_ids):
                    if cid not in locked:
                        implicated.append((cid, vehicle_eval.vehicle_id, idx, "route_constraint", 0.0))
        unique: dict[tuple[CustomerId, VehicleId], tuple[CustomerId, VehicleId, int, str, float]] = {}
        for item in implicated:
            key = (item[0], item[1])
            old = unique.get(key)
            if old is None or (item[4], item[3], -item[2]) > (old[4], old[3], -old[2]):
                unique[key] = item
        return [(cid, vid, pos, reason) for cid, vid, pos, reason, _ in sorted(unique.values(), key=lambda x: (-x[4], str(x[3]), str(x[0]), x[2]))]

    def _eligible_positions(self, route: VehicleRoute, vehicle_id: VehicleId, locked: set[CustomerId], *, moving_customer: CustomerId, source_vehicle_id: VehicleId) -> Iterable[int]:
        customers = list(route.customer_ids)
        if source_vehicle_id == vehicle_id:
            try:
                old = customers.index(moving_customer)
                customers.pop(old)
            except ValueError:
                return ()
        prefix_len = self._locked_prefix_length(vehicle_id, customers, locked)
        return range(prefix_len, len(customers) + 1)

    def _locked_prefix_length(self, vehicle_id: VehicleId, customers: Sequence[CustomerId], locked: set[CustomerId]) -> int:
        count = 0
        for cid in customers:
            if cid in locked:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _move(plan: RoutePlan, source_vehicle_id: VehicleId, customer_id: CustomerId, destination_vehicle_id: VehicleId, destination_position: int) -> RoutePlan:
        routes: list[VehicleRoute] = []
        for route in plan.vehicle_routes:
            customers = list(route.customer_ids)
            if route.vehicle_id == source_vehicle_id:
                try:
                    customers.remove(customer_id)
                except ValueError:
                    pass
            routes.append(VehicleRoute.from_sequence(route.vehicle_id, customers))
        result: list[VehicleRoute] = []
        for route in routes:
            if route.vehicle_id == destination_vehicle_id:
                customers = list(route.customer_ids)
                position = max(0, min(destination_position, len(customers)))
                customers.insert(position, customer_id)
                result.append(VehicleRoute.from_sequence(route.vehicle_id, customers))
            else:
                result.append(route)
        return RoutePlan.from_routes(result)

    @staticmethod
    def _score(evaluation: RoutePlanEvaluation) -> tuple[float, float]:
        violation_magnitude = sum(max(0.0, float(v.magnitude)) for v in evaluation.violations)
        violation_count = float(len(evaluation.violations))
        return violation_count, violation_magnitude

    def _budget_exhausted(self, started: float, attempts: int) -> bool:
        return attempts >= self.config.max_attempts or self._deadline_reached(started)

    def _deadline_reached(self, started: float) -> bool:
        return self.config.max_wall_time_s is not None and monotonic() - started >= self.config.max_wall_time_s

    def _unresolved(self, evaluation: RoutePlanEvaluation, locked: set[CustomerId]) -> tuple[CustomerId, ...]:
        result = set(evaluation.unserved_customer_ids)
        for ve in evaluation.vehicle_evaluations:
            for v in ve.violations:
                if v.customer_id is not None and v.customer_id not in locked:
                    result.add(v.customer_id)
        return tuple(sorted(result, key=str))

    @staticmethod
    def _failure_reason(evaluation: RoutePlanEvaluation, attempts: int, attempt_limit: bool, deadline: bool) -> str:
        if attempt_limit:
            return f"repair attempt cap reached after {attempts} candidate evaluations"
        if deadline:
            return "repair wall-time deadline reached"
        if evaluation.violations:
            return "no bounded mutation reduced the remaining evaluator violations"
        return "candidate remains infeasible"

    def _result(self, original, plan, evaluation, actions, attempts, moves, started, attempt_limit, deadline, unresolved, failure=None):
        return BoundedRepairResult(
            original_route_plan=original.route_plan,
            repaired_route_plan=plan,
            original_evaluation=original,
            repaired_evaluation=evaluation,
            actions=tuple(actions),
            feasible=evaluation.feasible,
            complete=evaluation.all_requests_served and not evaluation.duplicate_customer_ids and not evaluation.unknown_request_ids,
            attempts_used=attempts,
            moves_used=moves,
            elapsed_time_s=monotonic() - started,
            attempt_limit_reached=attempt_limit,
            deadline_reached=deadline,
            unresolved_customer_ids=tuple(unresolved),
            failure_reason=failure,
            evaluation_calls=self._evaluation_calls,
            evaluation_cache_hits=self._evaluation_cache_hits,
        )

@dataclass(frozen=True)
class RouteCandidate:
    """Complete Step 7 candidate lifecycle for one continuous key vector.

    ``stored_keys`` are always canonical keys for the route actually used for
    fitness.  If repair changes the decoded plan, callers therefore never
    associate the repaired fitness with stale particle coordinates.
    """

    decoded: RouteEncoding
    repaired_plan: RoutePlan
    repaired_evaluation: RoutePlanEvaluation
    stored_keys: tuple[float, ...]
    repair_result: BoundedRepairResult | None

    @property
    def feasible(self) -> bool:
        return self.repaired_evaluation.feasible


class Step7RouteEngine:
    """Reference implementation of the complete Step 7 decode/repair path."""

    def __init__(
        self,
        scenario: Any,
        evaluator: RouteEvaluator,
        *,
        encoding_config: RouteEncodingConfig | None = None,
        repair_config: RepairConfig | None = None,
    ) -> None:
        self.encoder = ContinuousRouteEncoder(scenario, config=encoding_config)
        self.repairer = BoundedRouteRepairer(
            scenario,
            evaluator,
            config=repair_config,
        )
        self.evaluator = evaluator

    def evaluate_keys(
        self,
        keys: Sequence[float] | Iterable[float],
        *,
        commitments: CommitmentSnapshot | None = None,
        planning_time_s: float = 0.0,
        repair: bool = True,
    ) -> RouteCandidate:
        decoded = self.encoder.decode(keys, commitments=commitments)
        if not repair:
            evaluation = self.evaluator.evaluate(
                decoded.route_plan,
                commitments=commitments,
                planning_time_s=planning_time_s,
            )
            return RouteCandidate(
                decoded=decoded,
                repaired_plan=decoded.route_plan,
                repaired_evaluation=evaluation,
                stored_keys=decoded.keys,
                repair_result=None,
            )

        repair_result = self.repairer.repair(
            decoded.route_plan,
            commitments=commitments,
            planning_time_s=planning_time_s,
        )
        canonical = self.encoder.encode(
            repair_result.repaired_route_plan,
            commitments=commitments,
        )
        return RouteCandidate(
            decoded=decoded,
            repaired_plan=repair_result.repaired_route_plan,
            repaired_evaluation=repair_result.repaired_evaluation,
            stored_keys=canonical.keys,
            repair_result=repair_result,
        )