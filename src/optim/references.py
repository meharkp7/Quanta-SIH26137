"""
Step 9 — Exact and independent heuristic routing references.

This module provides two reference solvers for the routing research track:

1. Snapshot MILP reference
   - Uses SciPy/HiGHS when available.
   - Designed for small CVRP/CVRPTW-style instances.
   - Explicitly records primal objective, dual/lower bound, status,
     time limit, requested MIP gap and certification state.
   - Never labels a time-limited non-zero-gap solve as a certified optimum.

2. ALNS reference
   - Independent destroy/repair metaheuristic.
   - Uses the same external route evaluator as the production optimizer.
   - Supports fixed fleet, capacity and time-window feasibility through
     the evaluator.
   - Includes multiple removal operators and adaptive operator selection.
   - Uses a reproducible random-number generator and an explicit evaluation
     budget.

The reference layer deliberately does not implement a second route evaluator.
All candidate routes are evaluated by the supplied oracle/evaluator so that
reference methods and QPSO share the same feasibility semantics.

Research semantics
------------------
A certified optimum is only reported when the exact solver establishes the
requested optimality tolerance.

For heuristic or time-limited exact runs, consumers must distinguish:

    best_feasible_objective
    lower_bound
    certified_optimum
    best_known_reference

This distinction is important for all later optimality-gap calculations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, isfinite, log
import random
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .common import FitnessResult


class ReferenceSolverError(RuntimeError):
    """Raised when a reference solver cannot construct a valid result."""


class ExactSolverUnavailable(ReferenceSolverError):
    """Raised when the optional exact-solver dependency is unavailable."""


class ReferenceMethod(str, Enum):
    """Supported Step 9 reference methods."""

    MILP = "milp"
    ALNS = "alns"


class ExactStatus(str, Enum):
    """
    Normalized exact-solver status.

    The values are intentionally independent of SciPy's numeric status codes.
    """

    OPTIMAL = "optimal"
    TIME_LIMIT = "time_limit"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NUMERICAL = "numerical"
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ReferenceRoute:
    """
    Immutable route representation used by the reference layer.

    ``vehicles`` maps vehicle identifiers to ordered customer identifiers.
    The representation intentionally does not contain physical road edges;
    the supplied evaluator is responsible for decoding/evaluating the route
    against the road graph.
    """

    vehicles: tuple[tuple[int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        seen: set[int] = set()

        for vehicle_id, customers in self.vehicles:
            if not isinstance(vehicle_id, int):
                raise ReferenceSolverError(
                    "vehicle_id must be an integer"
                )

            if vehicle_id in seen:
                raise ReferenceSolverError(
                    f"duplicate vehicle_id: {vehicle_id}"
                )

            seen.add(vehicle_id)

            for customer_id in customers:
                if not isinstance(customer_id, int):
                    raise ReferenceSolverError(
                        "customer_id must be an integer"
                    )

    @classmethod
    def from_routes(
        cls,
        routes: Mapping[int, Sequence[int]],
    ) -> "ReferenceRoute":
        return cls(
            vehicles=tuple(
                (
                    int(vehicle_id),
                    tuple(int(customer) for customer in customers),
                )
                for vehicle_id, customers in sorted(routes.items())
            )
        )

    def as_dict(self) -> dict[int, tuple[int, ...]]:
        return {
            vehicle_id: customers
            for vehicle_id, customers in self.vehicles
        }

    def all_customers(self) -> tuple[int, ...]:
        return tuple(
            customer_id
            for _, customers in self.vehicles
            for customer_id in customers
        )


@dataclass(frozen=True)
class ExactReferenceResult:
    """
    Complete result of a snapshot exact solve.

    ``objective`` is the best feasible primal objective if one exists.

    ``lower_bound`` is the solver's valid bound when available.

    ``certified_optimum`` is true only when the solver has established
    optimality to the requested tolerance.
    """

    method: ReferenceMethod
    status: ExactStatus
    route: ReferenceRoute | None
    objective: float | None
    lower_bound: float | None
    relative_gap: float | None
    mip_gap_tolerance: float
    time_limit_s: float
    elapsed_s: float
    evaluations: int
    certified_optimum: bool
    solver_status_code: int | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.method is not ReferenceMethod.MILP:
            raise ReferenceSolverError(
                "ExactReferenceResult method must be MILP"
            )

        if self.time_limit_s <= 0.0:
            raise ReferenceSolverError(
                "time_limit_s must be positive"
            )

        if self.mip_gap_tolerance < 0.0:
            raise ReferenceSolverError(
                "mip_gap_tolerance must be non-negative"
            )

        if self.elapsed_s < 0.0:
            raise ReferenceSolverError(
                "elapsed_s cannot be negative"
            )

        if self.evaluations < 0:
            raise ReferenceSolverError(
                "evaluations cannot be negative"
            )

        if self.objective is not None and not isfinite(
            float(self.objective)
        ):
            raise ReferenceSolverError(
                "objective must be finite when present"
            )

        if self.lower_bound is not None and not isfinite(
            float(self.lower_bound)
        ):
            raise ReferenceSolverError(
                "lower_bound must be finite when present"
            )

        if self.relative_gap is not None:
            if not isfinite(float(self.relative_gap)):
                raise ReferenceSolverError(
                    "relative_gap must be finite when present"
                )
            if self.relative_gap < 0.0:
                raise ReferenceSolverError(
                    "relative_gap cannot be negative"
                )

        if self.certified_optimum:
            if self.status is not ExactStatus.OPTIMAL:
                raise ReferenceSolverError(
                    "only OPTIMAL status can be certified"
                )

            if self.objective is None:
                raise ReferenceSolverError(
                    "certified optimum requires an objective"
                )

    @property
    def best_feasible_objective(self) -> float | None:
        return self.objective

    @property
    def gap_certified(self) -> bool:
        return self.certified_optimum


@dataclass(frozen=True)
class ALNSReferenceResult:
    """
    Result of an independent ALNS run.

    A heuristic result is never described as an optimum.  ``lower_bound`` may
    optionally be supplied from an independent bound source, but ALNS itself
    does not manufacture a mathematical lower bound.
    """

    method: ReferenceMethod
    route: ReferenceRoute | None
    objective: float | None
    lower_bound: float | None
    elapsed_s: float
    evaluations: int
    iterations: int
    feasible: bool
    seed: int
    removal_statistics: Mapping[str, int]
    repair_statistics: Mapping[str, int]
    operator_weights: Mapping[str, float]
    message: str = ""

    def __post_init__(self) -> None:
        if self.method is not ReferenceMethod.ALNS:
            raise ReferenceSolverError(
                "ALNSReferenceResult method must be ALNS"
            )

        if self.elapsed_s < 0.0:
            raise ReferenceSolverError(
                "elapsed_s cannot be negative"
            )

        if self.evaluations < 0:
            raise ReferenceSolverError(
                "evaluations cannot be negative"
            )

        if self.iterations < 0:
            raise ReferenceSolverError(
                "iterations cannot be negative"
            )

        if self.objective is not None and not isfinite(
            float(self.objective)
        ):
            raise ReferenceSolverError(
                "objective must be finite when present"
            )

        if self.lower_bound is not None and not isfinite(
            float(self.lower_bound)
        ):
            raise ReferenceSolverError(
                "lower_bound must be finite when present"
            )

        if self.feasible and self.route is None:
            raise ReferenceSolverError(
                "a feasible ALNS result requires a route"
            )

        if self.feasible and self.objective is None:
            raise ReferenceSolverError(
                "a feasible ALNS result requires an objective"
            )


@dataclass(frozen=True)
class ALNSConfig:
    """
    Configuration for the independent ALNS reference.

    The evaluation budget is the primary reproducibility control.  The
    wall-clock limit is a secondary safety bound.
    """

    max_evaluations: int = 2_000
    time_limit_s: float = 5.0
    removal_fraction_min: float = 0.10
    removal_fraction_max: float = 0.40
    reaction_factor: float = 0.20
    segment_length: int = 25
    initial_temperature: float = 1.0
    cooling_rate: float = 0.995
    destroy_operators: tuple[str, ...] = (
        "random_removal",
        "worst_removal",
        "related_removal",
    )
    repair_operators: tuple[str, ...] = (
        "greedy_insertion",
        "regret_insertion",
    )

    def __post_init__(self) -> None:
        if self.max_evaluations <= 0:
            raise ReferenceSolverError(
                "max_evaluations must be positive"
            )

        if self.time_limit_s <= 0.0:
            raise ReferenceSolverError(
                "time_limit_s must be positive"
            )

        if not (
            0.0 < self.removal_fraction_min
            <= self.removal_fraction_max
            < 1.0
        ):
            raise ReferenceSolverError(
                "removal fractions must satisfy "
                "0 < min <= max < 1"
            )

        if not 0.0 < self.reaction_factor <= 1.0:
            raise ReferenceSolverError(
                "reaction_factor must lie in (0, 1]"
            )

        if self.segment_length <= 0:
            raise ReferenceSolverError(
                "segment_length must be positive"
            )

        if self.initial_temperature < 0.0:
            raise ReferenceSolverError(
                "initial_temperature cannot be negative"
            )

        if not 0.0 < self.cooling_rate <= 1.0:
            raise ReferenceSolverError(
                "cooling_rate must lie in (0, 1]"
            )

        if len(self.destroy_operators) < 2:
            raise ReferenceSolverError(
                "ALNS requires at least two destroy operators"
            )

        if not self.repair_operators:
            raise ReferenceSolverError(
                "at least one repair operator is required"
            )


@dataclass(frozen=True)
class ALNSCandidate:
    """
    Internal immutable ALNS candidate.

    The route is represented as vehicle -> ordered customers.
    """

    route: ReferenceRoute
    fitness_result: FitnessResult

    @property
    def objective(self) -> float:
        return float(self.fitness_result.fitness)

    @property
    def feasible(self) -> bool:
        return bool(self.fitness_result.feasible)


@dataclass
class _OperatorState:
    """Adaptive roulette-wheel state for one ALNS operator."""

    names: tuple[str, ...]
    weights: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    uses: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.names:
            self.weights.setdefault(name, 1.0)
            self.scores.setdefault(name, 0.0)
            self.uses.setdefault(name, 0)

    def select(self, rng: random.Random) -> str:
        total = sum(
            max(0.0, float(self.weights[name]))
            for name in self.names
        )

        if total <= 0.0:
            return self.names[rng.randrange(len(self.names))]

        target = rng.random() * total
        cumulative = 0.0

        for name in self.names:
            cumulative += max(0.0, float(self.weights[name]))
            if target <= cumulative:
                return name

        return self.names[-1]

    def record(
        self,
        name: str,
        score: float,
    ) -> None:
        if name not in self.weights:
            raise ReferenceSolverError(
                f"unknown ALNS operator: {name}"
            )

        self.scores[name] += float(score)
        self.uses[name] += 1

    def update(
        self,
        reaction_factor: float,
    ) -> None:
        for name in self.names:
            uses = self.uses[name]

            if uses <= 0:
                continue

            reward = self.scores[name] / uses

            self.weights[name] = (
                (1.0 - reaction_factor) * self.weights[name]
                + reaction_factor * max(0.01, reward)
            )

            self.scores[name] = 0.0
            self.uses[name] = 0


class ALNSReference:
    """
    Independent Adaptive Large Neighbourhood Search reference.

    Parameters
    ----------
    evaluator:
        Callable accepting a ReferenceRoute and returning FitnessResult.

        The evaluator must be the independent route evaluator already used by
        the project. ALNS therefore cannot silently invent different
        capacity/window/closure semantics.

    customer_ids:
        Complete set of customer identifiers that must be served.

    vehicle_ids:
        Fixed fleet identifiers. ALNS never creates additional vehicles.

    initial_route:
        Optional known feasible starting solution. When omitted, a deterministic
        round-robin route is constructed and then checked by the evaluator.

    seed:
        Reproducibility seed.
    """

    def __init__(
        self,
        *,
        evaluator: Callable[[ReferenceRoute], FitnessResult],
        customer_ids: Sequence[int],
        vehicle_ids: Sequence[int],
        initial_route: ReferenceRoute | Mapping[int, Sequence[int]] | None = None,
        config: ALNSConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.evaluator = evaluator
        self.customer_ids = tuple(int(x) for x in customer_ids)
        self.vehicle_ids = tuple(int(x) for x in vehicle_ids)
        self.config = config or ALNSConfig()
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        if not self.customer_ids:
            raise ReferenceSolverError(
                "customer_ids cannot be empty"
            )

        if not self.vehicle_ids:
            raise ReferenceSolverError(
                "vehicle_ids cannot be empty"
            )

        if len(set(self.customer_ids)) != len(self.customer_ids):
            raise ReferenceSolverError(
                "customer_ids must be unique"
            )

        if len(set(self.vehicle_ids)) != len(self.vehicle_ids):
            raise ReferenceSolverError(
                "vehicle_ids must be unique"
            )

        if initial_route is None:
            self.initial_route = self._construct_initial_route()
        elif isinstance(initial_route, ReferenceRoute):
            self.initial_route = initial_route
        else:
            self.initial_route = ReferenceRoute.from_routes(
                initial_route
            )

        self.evaluations = 0

        self.destroy_state = _OperatorState(
            names=self.config.destroy_operators
        )
        self.repair_state = _OperatorState(
            names=self.config.repair_operators
        )

        self.removal_statistics = {
            name: 0
            for name in self.config.destroy_operators
        }
        self.repair_statistics = {
            name: 0
            for name in self.config.repair_operators
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ALNSReferenceResult:
        started = time.perf_counter()

        current = self._evaluate(self.initial_route)

        best = current if current.feasible else None

        iterations = 0
        temperature = self.config.initial_temperature

        while self._within_budget(started):
            iterations += 1

            destroy_name = self.destroy_state.select(self.rng)
            repair_name = self.repair_state.select(self.rng)

            removed, partial = self._destroy(
                current.route,
                destroy_name,
            )

            candidate_route = self._repair(
                partial,
                removed,
                repair_name,
            )

            candidate = self._evaluate(candidate_route)

            if candidate is None:
                break

            accepted = self._accept(
                current=current,
                candidate=candidate,
                temperature=temperature,
            )

            reward = 0.0

            if candidate.feasible:
                if best is None:
                    best = candidate
                    reward = 5.0
                elif candidate.objective < best.objective - 1e-12:
                    best = candidate
                    reward = 5.0
                elif candidate.objective < current.objective - 1e-12:
                    reward = 2.0

            if accepted:
                current = candidate

                if reward == 0.0:
                    reward = 1.0

            self.destroy_state.record(
                destroy_name,
                reward,
            )
            self.repair_state.record(
                repair_name,
                reward,
            )

            self.removal_statistics[destroy_name] += 1
            self.repair_statistics[repair_name] += 1

            if (
                iterations % self.config.segment_length == 0
            ):
                self.destroy_state.update(
                    self.config.reaction_factor
                )
                self.repair_state.update(
                    self.config.reaction_factor
                )

            if temperature > 0.0:
                temperature *= self.config.cooling_rate

        elapsed = time.perf_counter() - started

        if best is None:
            return ALNSReferenceResult(
                method=ReferenceMethod.ALNS,
                route=None,
                objective=None,
                lower_bound=None,
                elapsed_s=elapsed,
                evaluations=self.evaluations,
                iterations=iterations,
                feasible=False,
                seed=self.seed,
                removal_statistics=dict(
                    self.removal_statistics
                ),
                repair_statistics=dict(
                    self.repair_statistics
                ),
                operator_weights=dict(
                    self.destroy_state.weights
                )
                | {
                    f"repair:{name}": weight
                    for name, weight
                    in self.repair_state.weights.items()
                },
                message="ALNS did not produce a feasible solution",
            )

        return ALNSReferenceResult(
            method=ReferenceMethod.ALNS,
            route=best.route,
            objective=best.objective,
            lower_bound=None,
            elapsed_s=elapsed,
            evaluations=self.evaluations,
            iterations=iterations,
            feasible=True,
            seed=self.seed,
            removal_statistics=dict(
                self.removal_statistics
            ),
            repair_statistics=dict(
                self.repair_statistics
            ),
            operator_weights=dict(
                self.destroy_state.weights
            )
            | {
                f"repair:{name}": weight
                for name, weight
                in self.repair_state.weights.items()
            },
            message="ALNS completed within the declared budget",
        )

    # ------------------------------------------------------------------
    # Initial construction
    # ------------------------------------------------------------------

    def _construct_initial_route(self) -> ReferenceRoute:
        routes = {
            vehicle_id: []
            for vehicle_id in self.vehicle_ids
        }

        for index, customer_id in enumerate(self.customer_ids):
            vehicle_id = self.vehicle_ids[
                index % len(self.vehicle_ids)
            ]
            routes[vehicle_id].append(customer_id)

        return ReferenceRoute.from_routes(routes)

    # ------------------------------------------------------------------
    # Evaluation and budget
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        route: ReferenceRoute,
    ) -> ALNSCandidate | None:
        if self.evaluations >= self.config.max_evaluations:
            return None

        result = self.evaluator(route)

        if not isinstance(result, FitnessResult):
            raise ReferenceSolverError(
                "reference evaluator must return FitnessResult"
            )

        if not isfinite(float(result.fitness)):
            raise ReferenceSolverError(
                "reference evaluator returned non-finite fitness"
            )

        self.evaluations += 1

        return ALNSCandidate(
            route=route,
            fitness_result=result,
        )

    def _within_budget(
        self,
        started: float,
    ) -> bool:
        if self.evaluations >= self.config.max_evaluations:
            return False

        return (
            time.perf_counter() - started
            < self.config.time_limit_s
        )

    # ------------------------------------------------------------------
    # Destroy operators
    # ------------------------------------------------------------------

    def _destroy(
        self,
        route: ReferenceRoute,
        operator: str,
    ) -> tuple[tuple[int, ...], ReferenceRoute]:
        if operator == "random_removal":
            return self._random_removal(route)

        if operator == "worst_removal":
            return self._worst_removal(route)

        if operator == "related_removal":
            return self._related_removal(route)

        raise ReferenceSolverError(
            f"unknown destroy operator: {operator}"
        )

    def _random_removal(
        self,
        route: ReferenceRoute,
    ) -> tuple[tuple[int, ...], ReferenceRoute]:
        customers = list(route.all_customers())

        if not customers:
            return (), route

        count = self._removal_count(len(customers))

        removed = tuple(
            self.rng.sample(customers, count)
        )

        return removed, self._remove_customers(
            route,
            removed,
        )

    def _worst_removal(
        self,
        route: ReferenceRoute,
    ) -> tuple[tuple[int, ...], ReferenceRoute]:
        customers = list(route.all_customers())

        if not customers:
            return (), route

        count = self._removal_count(len(customers))

        # The reference layer cannot assume an objective decomposition.
        # Therefore "worst" is estimated through local singleton removal
        # effects using the supplied evaluator. Each trial is bounded by the
        # same global evaluation budget.
        baseline = self._evaluate(route)

        if baseline is None:
            return (), route

        impacts: list[tuple[float, int]] = []

        for customer_id in customers:
            if self.evaluations >= self.config.max_evaluations:
                break

            reduced = self._remove_customers(
                route,
                (customer_id,),
            )

            candidate = self._evaluate(reduced)

            if candidate is None:
                break

            # A larger improvement after removal means the customer was
            # relatively expensive under the current route.
            impact = baseline.objective - candidate.objective

            impacts.append(
                (float(impact), customer_id)
            )

        impacts.sort(reverse=True)

        removed = tuple(
            customer_id
            for _, customer_id in impacts[:count]
        )

        if not removed:
            return self._random_removal(route)

        return removed, self._remove_customers(
            route,
            removed,
        )

    def _related_removal(
        self,
        route: ReferenceRoute,
    ) -> tuple[tuple[int, ...], ReferenceRoute]:
        customers = list(route.all_customers())

        if not customers:
            return (), route

        count = self._removal_count(len(customers))

        seed_customer = self.rng.choice(customers)

        # Without requiring a geometry-specific contract, relatedness is
        # approximated structurally: customers close in the route sequence
        # to a sampled seed are considered related.
        positions = self._customer_positions(route)

        ranked = sorted(
            customers,
            key=lambda customer_id: abs(
                positions[customer_id]
                - positions[seed_customer]
            ),
        )

        removed = tuple(ranked[:count])

        return removed, self._remove_customers(
            route,
            removed,
        )

    # ------------------------------------------------------------------
    # Repair operators
    # ------------------------------------------------------------------

    def _repair(
        self,
        partial: ReferenceRoute,
        removed: Sequence[int],
        operator: str,
    ) -> ReferenceRoute:
        if not removed:
            return partial

        if operator == "greedy_insertion":
            return self._greedy_insertion(
                partial,
                removed,
            )

        if operator == "regret_insertion":
            return self._regret_insertion(
                partial,
                removed,
            )

        raise ReferenceSolverError(
            f"unknown repair operator: {operator}"
        )

    def _greedy_insertion(
        self,
        route: ReferenceRoute,
        removed: Sequence[int],
    ) -> ReferenceRoute:
        current = route

        for customer_id in removed:
            best_route: ReferenceRoute | None = None
            best_candidate: ALNSCandidate | None = None

            for vehicle_id in self.vehicle_ids:
                customers = list(
                    current.as_dict().get(vehicle_id, ())
                )

                for position in range(
                    len(customers) + 1
                ):
                    trial = list(customers)
                    trial.insert(position, customer_id)

                    trial_routes = current.as_dict()
                    trial_routes[vehicle_id] = tuple(trial)

                    candidate_route = (
                        ReferenceRoute.from_routes(
                            trial_routes
                        )
                    )

                    candidate = self._evaluate(
                        candidate_route
                    )

                    if candidate is None:
                        return current

                    if (
                        best_candidate is None
                        or self._candidate_better(
                            candidate,
                            best_candidate,
                        )
                    ):
                        best_candidate = candidate
                        best_route = candidate_route

            if best_route is None:
                return current

            current = best_route

        return current

    def _regret_insertion(
        self,
        route: ReferenceRoute,
        removed: Sequence[int],
    ) -> ReferenceRoute:
        current = route
        pending = list(removed)

        while pending:
            selected_customer: int | None = None
            selected_route: ReferenceRoute | None = None
            selected_regret = float("-inf")

            for customer_id in pending:
                alternatives: list[ALNSCandidate] = []

                for vehicle_id in self.vehicle_ids:
                    customers = list(
                        current.as_dict().get(vehicle_id, ())
                    )

                    for position in range(
                        len(customers) + 1
                    ):
                        trial = list(customers)
                        trial.insert(position, customer_id)

                        trial_routes = current.as_dict()
                        trial_routes[vehicle_id] = tuple(trial)

                        candidate_route = (
                            ReferenceRoute.from_routes(
                                trial_routes
                            )
                        )

                        candidate = self._evaluate(
                            candidate_route
                        )

                        if candidate is None:
                            return current

                        alternatives.append(candidate)

                if not alternatives:
                    continue

                alternatives.sort(
                    key=self._candidate_sort_key
                )

                best = alternatives[0].objective

                if len(alternatives) >= 2:
                    second = alternatives[1].objective
                else:
                    second = best

                regret = second - best

                if regret > selected_regret:
                    selected_regret = regret
                    selected_customer = customer_id
                    selected_route = alternatives[0].route

            if selected_customer is None or selected_route is None:
                return current

            current = selected_route
            pending.remove(selected_customer)

        return current

    # ------------------------------------------------------------------
    # Acceptance
    # ------------------------------------------------------------------

    def _accept(
        self,
        *,
        current: ALNSCandidate,
        candidate: ALNSCandidate,
        temperature: float,
    ) -> bool:
        if self._candidate_better(
            candidate,
            current,
        ):
            return True

        if (
            current.feasible
            and not candidate.feasible
        ):
            return False

        if (
            not current.feasible
            and candidate.feasible
        ):
            return True

        if temperature <= 1e-12:
            return False

        delta = candidate.objective - current.objective

        probability = min(
            1.0,
            max(
                0.0,
                pow(
                    2.718281828459045,
                    -delta / temperature,
                ),
            ),
        )

        return self.rng.random() < probability

    @staticmethod
    def _candidate_better(
        candidate: ALNSCandidate,
        incumbent: ALNSCandidate,
    ) -> bool:
        if candidate.feasible != incumbent.feasible:
            return candidate.feasible

        return (
            candidate.objective
            < incumbent.objective - 1e-12
        )

    @staticmethod
    def _candidate_sort_key(
        candidate: ALNSCandidate,
    ) -> tuple[int, float]:
        return (
            0 if candidate.feasible else 1,
            candidate.objective,
        )

    # ------------------------------------------------------------------
    # Route utilities
    # ------------------------------------------------------------------

    def _removal_count(
        self,
        customer_count: int,
    ) -> int:
        lower = max(
            1,
            ceil(
                customer_count
                * self.config.removal_fraction_min
            ),
        )

        upper = max(
            lower,
            ceil(
                customer_count
                * self.config.removal_fraction_max
            ),
        )

        upper = min(
            upper,
            customer_count,
        )

        return self.rng.randint(
            lower,
            upper,
        )

    @staticmethod
    def _customer_positions(
        route: ReferenceRoute,
    ) -> dict[int, int]:
        positions: dict[int, int] = {}

        position = 0

        for _, customers in route.vehicles:
            for customer_id in customers:
                positions[customer_id] = position
                position += 1

        return positions

    @staticmethod
    def _remove_customers(
        route: ReferenceRoute,
        removed: Iterable[int],
    ) -> ReferenceRoute:
        removed_set = set(int(x) for x in removed)

        routes = {
            vehicle_id: tuple(
                customer_id
                for customer_id in customers
                if customer_id not in removed_set
            )
            for vehicle_id, customers
            in route.vehicles
        }

        return ReferenceRoute.from_routes(routes)


# ----------------------------------------------------------------------
# Exact MILP reference
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MILPNode:
    """Snapshot routing node used by the generic exact builder."""

    node_id: int
    demand: float = 0.0
    service_time: float = 0.0
    earliest: float = 0.0
    latest: float = float("inf")


@dataclass(frozen=True)
class MILPVehicle:
    """Fixed-fleet vehicle specification."""

    vehicle_id: int
    capacity: float


@dataclass(frozen=True)
class MILPArc:
    """
    Allowed directed customer/depot arc.

    ``travel_time`` and ``distance`` must describe the same physical movement.
    """

    origin: int
    destination: int
    travel_time: float
    distance: float = 0.0

    def __post_init__(self) -> None:
        if self.origin == self.destination:
            raise ReferenceSolverError(
                "self-arcs are not supported"
            )

        if self.travel_time < 0.0:
            raise ReferenceSolverError(
                "travel_time cannot be negative"
            )

        if self.distance < 0.0:
            raise ReferenceSolverError(
                "distance cannot be negative"
            )


@dataclass(frozen=True)
class SnapshotCVRPTW:
    """
    Small fixed-fleet snapshot problem for the exact reference.

    Node IDs must include:
        depot_start
        customers
        depot_end

    Customer service starts are constrained by [earliest, latest].
    """

    depot_start: int
    depot_end: int
    customers: tuple[MILPNode, ...]
    vehicles: tuple[MILPVehicle, ...]
    arcs: tuple[MILPArc, ...]
    departure_time: float = 0.0
    return_deadline: float = float("inf")
    objective_time_weight: float = 1.0
    objective_distance_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.customers:
            raise ReferenceSolverError(
                "snapshot problem must contain customers"
            )

        if not self.vehicles:
            raise ReferenceSolverError(
                "snapshot problem must contain vehicles"
            )

        if self.departure_time < 0.0:
            raise ReferenceSolverError(
                "departure_time cannot be negative"
            )

        if (
            self.return_deadline != float("inf")
            and self.return_deadline < self.departure_time
        ):
            raise ReferenceSolverError(
                "return_deadline must not precede departure_time"
            )

        if self.objective_time_weight < 0.0:
            raise ReferenceSolverError(
                "objective_time_weight cannot be negative"
            )

        if self.objective_distance_weight < 0.0:
            raise ReferenceSolverError(
                "objective_distance_weight cannot be negative"
            )

        if (
            self.objective_time_weight
            + self.objective_distance_weight
            <= 0.0
        ):
            raise ReferenceSolverError(
                "at least one objective weight must be positive"
            )

        node_ids = {
            self.depot_start,
            self.depot_end,
        }

        for customer in self.customers:
            if customer.node_id in node_ids:
                raise ReferenceSolverError(
                    f"duplicate node id: {customer.node_id}"
                )
            node_ids.add(customer.node_id)

        known = node_ids

        for arc in self.arcs:
            if (
                arc.origin not in known
                or arc.destination not in known
            ):
                raise ReferenceSolverError(
                    "MILP arc references an unknown node"
                )


class SnapshotMILPReference:
    """
    Exact snapshot CVRPTW reference using scipy.optimize.milp.

    The formulation is deliberately explicit rather than delegating to a
    black-box vehicle-routing solver.

    The objective is:

        w_T * total route operating time
        + w_D * total route distance

    The model contains:
        - fixed fleet activation
        - customer assignment
        - directed flow conservation
        - vehicle capacity
        - service time windows
        - route start/end constraints
        - MTZ visit-order constraints
        - finite big-M time propagation

    This class is intended for small exact-reference instances.
    """

    def __init__(
        self,
        *,
        problem: SnapshotCVRPTW,
        time_limit_s: float = 10.0,
        mip_gap: float = 1e-6,
    ) -> None:
        if time_limit_s <= 0.0:
            raise ReferenceSolverError(
                "time_limit_s must be positive"
            )

        if mip_gap < 0.0:
            raise ReferenceSolverError(
                "mip_gap must be non-negative"
            )

        self.problem = problem
        self.time_limit_s = float(time_limit_s)
        self.mip_gap = float(mip_gap)

    def solve(self) -> ExactReferenceResult:
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import lil_matrix
        except ImportError as exc:
            raise ExactSolverUnavailable(
                "SciPy with optimize.milp is required for the "
                "snapshot MILP reference"
            ) from exc

        problem = self.problem

        customer_ids = tuple(
            customer.node_id
            for customer in problem.customers
        )

        vehicle_ids = tuple(
            vehicle.vehicle_id
            for vehicle in problem.vehicles
        )

        arcs = tuple(problem.arcs)

        if not arcs:
            raise ReferenceSolverError(
                "snapshot problem contains no allowed arcs"
            )

        arc_lookup = {
            (arc.origin, arc.destination): arc
            for arc in arcs
        }

        customer_index = {
            customer_id: index
            for index, customer_id
            in enumerate(customer_ids)
        }

        vehicle_index = {
            vehicle_id: index
            for index, vehicle_id
            in enumerate(vehicle_ids)
        }

        # --------------------------------------------------------------
        # Variable indexing
        # --------------------------------------------------------------

        # x[a, k] = vehicle k uses stop-level arc a
        x_index: dict[tuple[int, int], int] = {}

        # y[i, k] = customer i assigned to vehicle k
        y_index: dict[tuple[int, int], int] = {}

        # z[k] = vehicle k activated
        z_index: dict[int, int] = {}

        # t[i, k] = service start at customer i
        t_index: dict[tuple[int, int], int] = {}

        # u[i, k] = MTZ visit order
        u_index: dict[tuple[int, int], int] = {}

        variable_count = 0

        for arc_id, _ in enumerate(arcs):
            for vehicle_id in vehicle_ids:
                x_index[(arc_id, vehicle_id)] = variable_count
                variable_count += 1

        for customer_id in customer_ids:
            for vehicle_id in vehicle_ids:
                y_index[(customer_id, vehicle_id)] = variable_count
                variable_count += 1

        for vehicle_id in vehicle_ids:
            z_index[vehicle_id] = variable_count
            variable_count += 1

        horizon = self._finite_horizon()

        for customer_id in customer_ids:
            customer = problem.customers[
                customer_index[customer_id]
            ]

            for vehicle_id in vehicle_ids:
                t_index[(customer_id, vehicle_id)] = variable_count
                variable_count += 1

                u_index[(customer_id, vehicle_id)] = variable_count
                variable_count += 1

                # Ensure the object is used, avoiding accidental removal
                # by future refactoring.
                _ = customer

        objective = np.zeros(variable_count, dtype=float)

        for arc_id, arc in enumerate(arcs):
            coefficient = (
                problem.objective_time_weight
                * arc.travel_time
                + problem.objective_distance_weight
                * arc.distance
            )

            for vehicle_id in vehicle_ids:
                objective[
                    x_index[(arc_id, vehicle_id)]
                ] = coefficient

        integrality = np.zeros(
            variable_count,
            dtype=int,
        )

        lower_bounds = np.full(
            variable_count,
            -np.inf,
            dtype=float,
        )

        upper_bounds = np.full(
            variable_count,
            np.inf,
            dtype=float,
        )

        # Binary x, y, z.
        for index in (
            list(x_index.values())
            + list(y_index.values())
            + list(z_index.values())
        ):
            integrality[index] = 1
            lower_bounds[index] = 0.0
            upper_bounds[index] = 1.0

        # Time variables.
        for index in t_index.values():
            lower_bounds[index] = (
                problem.departure_time
            )
            upper_bounds[index] = horizon

        # MTZ variables.
        n_customers = len(customer_ids)

        for index in u_index.values():
            lower_bounds[index] = 0.0
            upper_bounds[index] = float(n_customers)

        # --------------------------------------------------------------
        # Linear constraints
        # --------------------------------------------------------------

        rows: list[dict[int, float]] = []
        row_lower: list[float] = []
        row_upper: list[float] = []

        def add_constraint(
            coefficients: Mapping[int, float],
            lower: float,
            upper: float,
        ) -> None:
            rows.append(dict(coefficients))
            row_lower.append(lower)
            row_upper.append(upper)

        # --------------------------------------------------------------
        # Every customer served exactly once.
        # --------------------------------------------------------------

        for customer_id in customer_ids:
            coefficients = {
                y_index[(customer_id, vehicle_id)]: 1.0
                for vehicle_id in vehicle_ids
            }

            add_constraint(
                coefficients,
                1.0,
                1.0,
            )

        # --------------------------------------------------------------
        # Flow conservation at every customer for every vehicle.
        # --------------------------------------------------------------

        for customer_id in customer_ids:
            for vehicle_id in vehicle_ids:
                incoming = {
                    x_index[(arc_id, vehicle_id)]: 1.0
                    for arc_id, arc in enumerate(arcs)
                    if arc.destination == customer_id
                }

                outgoing = {
                    x_index[(arc_id, vehicle_id)]: 1.0
                    for arc_id, arc in enumerate(arcs)
                    if arc.origin == customer_id
                }

                coefficients: dict[int, float] = {}

                for index, value in incoming.items():
                    coefficients[index] = (
                        coefficients.get(index, 0.0)
                        + value
                    )

                for index, value in outgoing.items():
                    coefficients[index] = (
                        coefficients.get(index, 0.0)
                        - value
                    )

                coefficients[
                    y_index[(customer_id, vehicle_id)]
                ] = -0.0

                add_constraint(
                    coefficients,
                    0.0,
                    0.0,
                )

                # Incoming = assignment.
                add_constraint(
                    {
                        **incoming,
                        y_index[
                            (customer_id, vehicle_id)
                        ]: -1.0,
                    },
                    0.0,
                    0.0,
                )

        # --------------------------------------------------------------
        # Vehicle depot start/end and activation.
        # --------------------------------------------------------------

        for vehicle_id in vehicle_ids:
            outgoing_depot = {
                x_index[(arc_id, vehicle_id)]: 1.0
                for arc_id, arc in enumerate(arcs)
                if arc.origin == problem.depot_start
            }

            incoming_end = {
                x_index[(arc_id, vehicle_id)]: 1.0
                for arc_id, arc in enumerate(arcs)
                if arc.destination == problem.depot_end
            }

            add_constraint(
                {
                    **outgoing_depot,
                    z_index[vehicle_id]: -1.0,
                },
                0.0,
                0.0,
            )

            add_constraint(
                {
                    **incoming_end,
                    z_index[vehicle_id]: -1.0,
                },
                0.0,
                0.0,
            )

            # No customer may be assigned to an inactive vehicle.
            for customer_id in customer_ids:
                add_constraint(
                    {
                        y_index[
                            (customer_id, vehicle_id)
                        ]: 1.0,
                        z_index[vehicle_id]: -1.0,
                    },
                    -np.inf,
                    0.0,
                )

        # --------------------------------------------------------------
        # Capacity.
        # --------------------------------------------------------------

        for vehicle in problem.vehicles:
            add_constraint(
                {
                    y_index[
                        (customer.node_id, vehicle.vehicle_id)
                    ]: customer.demand
                    for customer in problem.customers
                },
                -np.inf,
                vehicle.capacity,
            )

        # --------------------------------------------------------------
        # Time windows.
        # --------------------------------------------------------------

        for customer in problem.customers:
            for vehicle_id in vehicle_ids:
                y = y_index[
                    (customer.node_id, vehicle_id)
                ]
                t = t_index[
                    (customer.node_id, vehicle_id)
                ]

                add_constraint(
                    {
                        t: 1.0,
                        y: -customer.earliest,
                    },
                    0.0,
                    np.inf,
                )

                add_constraint(
                    {
                        t: 1.0,
                        y: -customer.latest,
                    },
                    -np.inf,
                    0.0,
                )

                # If not assigned, service time is fixed at zero.
                # Since t has lower bound departure_time, we use a
                # tighter conditional formulation around the horizon.
                add_constraint(
                    {
                        t: 1.0,
                        y: -horizon,
                    },
                    -np.inf,
                    0.0,
                )

        # --------------------------------------------------------------
        # Time propagation on customer-to-customer arcs.
        # --------------------------------------------------------------

        for arc_id, arc in enumerate(arcs):
            if (
                arc.origin not in customer_index
                or arc.destination not in customer_index
            ):
                continue

            origin = arc.origin
            destination = arc.destination

            origin_node = problem.customers[
                customer_index[origin]
            ]

            big_m = self._time_big_m(
                origin_node,
                arc,
                horizon,
            )

            for vehicle_id in vehicle_ids:
                x = x_index[(arc_id, vehicle_id)]
                t_origin = t_index[(origin, vehicle_id)]
                t_destination = t_index[
                    (destination, vehicle_id)
                ]

                add_constraint(
                    {
                        t_destination: 1.0,
                        t_origin: -1.0,
                        x: big_m,
                    },
                    origin_node.service_time
                    + arc.travel_time
                    - big_m,
                    np.inf,
                )

        # --------------------------------------------------------------
        # Depot departure propagation.
        # --------------------------------------------------------------

        for arc_id, arc in enumerate(arcs):
            if arc.origin != problem.depot_start:
                continue

            if arc.destination not in customer_index:
                continue

            destination = arc.destination

            for vehicle_id in vehicle_ids:
                x = x_index[(arc_id, vehicle_id)]
                t_destination = t_index[
                    (destination, vehicle_id)
                ]

                big_m = horizon + arc.travel_time + 1.0

                add_constraint(
                    {
                        t_destination: 1.0,
                        x: big_m,
                    },
                    problem.departure_time
                    + arc.travel_time,
                    np.inf,
                )

        # --------------------------------------------------------------
        # MTZ subtour elimination.
        # --------------------------------------------------------------

        for vehicle_id in vehicle_ids:
            for origin in customer_ids:
                for destination in customer_ids:
                    if origin == destination:
                        continue

                    arc = arc_lookup.get(
                        (origin, destination)
                    )

                    if arc is None:
                        continue

                    # Find its index.
                    arc_id = next(
                        index
                        for index, candidate_arc
                        in enumerate(arcs)
                        if candidate_arc == arc
                    )

                    x = x_index[(arc_id, vehicle_id)]
                    u_origin = u_index[
                        (origin, vehicle_id)
                    ]
                    u_destination = u_index[
                        (destination, vehicle_id)
                    ]

                    M = float(n_customers + 1)

                    add_constraint(
                        {
                            u_destination: 1.0,
                            u_origin: -1.0,
                            x: M,
                        },
                        1.0 - M,
                        np.inf,
                    )

        # --------------------------------------------------------------
        # Build sparse constraint matrix.
        # --------------------------------------------------------------

        matrix = lil_matrix(
            (
                len(rows),
                variable_count,
            ),
            dtype=float,
        )

        for row_index, coefficients in enumerate(rows):
            for column_index, coefficient in coefficients.items():
                matrix[row_index, column_index] = coefficient

        constraints = LinearConstraint(
            matrix.tocsr(),
            np.asarray(row_lower, dtype=float),
            np.asarray(row_upper, dtype=float),
        )

        bounds = Bounds(
            lower_bounds,
            upper_bounds,
        )

        options = {
            "time_limit": self.time_limit_s,
            "mip_rel_gap": self.mip_gap,
        }

        started = time.perf_counter()

        result = milp(
            c=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        elapsed = time.perf_counter() - started

        status = self._normalize_scipy_status(
            int(result.status)
        )

        objective_value = (
            float(result.fun)
            if result.fun is not None
            and isfinite(float(result.fun))
            else None
        )

        lower_bound = (
            float(result.mip_node_count * 0.0)
            if False
            else self._extract_lower_bound(result)
        )

        relative_gap = self._extract_gap(
            result,
            objective_value,
            lower_bound,
        )

        route = None

        if result.x is not None:
            route = self._decode_milp_route(
                result.x,
                x_index,
                arcs,
                vehicle_ids,
            )

        certified = (
            status is ExactStatus.OPTIMAL
            and objective_value is not None
            and (
                relative_gap is None
                or relative_gap <= self.mip_gap + 1e-12
            )
        )

        return ExactReferenceResult(
            method=ReferenceMethod.MILP,
            status=status,
            route=route,
            objective=objective_value,
            lower_bound=lower_bound,
            relative_gap=relative_gap,
            mip_gap_tolerance=self.mip_gap,
            time_limit_s=self.time_limit_s,
            elapsed_s=elapsed,
            evaluations=0,
            certified_optimum=certified,
            solver_status_code=int(result.status),
            message=str(result.message),
        )

    # ------------------------------------------------------------------
    # Exact-solver helpers
    # ------------------------------------------------------------------

    def _finite_horizon(self) -> float:
        problem = self.problem

        if problem.return_deadline != float("inf"):
            return float(problem.return_deadline)

        max_latest = max(
            customer.latest
            for customer in problem.customers
        )

        max_arc = max(
            arc.travel_time
            for arc in problem.arcs
        )

        total_service = sum(
            customer.service_time
            for customer in problem.customers
        )

        return max(
            max_latest + max_arc + total_service,
            problem.departure_time + 1.0,
        )

    @staticmethod
    def _time_big_m(
        origin: MILPNode,
        arc: MILPArc,
        horizon: float,
    ) -> float:
        return max(
            1.0,
            horizon
            + origin.service_time
            + arc.travel_time
            + 1.0,
        )

    @staticmethod
    def _normalize_scipy_status(
        status: int,
    ) -> ExactStatus:
        return {
            0: ExactStatus.OPTIMAL,
            1: ExactStatus.TIME_LIMIT,
            2: ExactStatus.INFEASIBLE,
            3: ExactStatus.UNBOUNDED,
            4: ExactStatus.NUMERICAL,
        }.get(
            status,
            ExactStatus.UNKNOWN,
        )

    @staticmethod
    def _extract_lower_bound(
        result: Any,
    ) -> float | None:
        # SciPy/HiGHS exposes mip dual information differently across
        # versions. Prefer an explicit attribute when available.
        for name in (
            "mip_dual_bound",
            "dual_bound",
            "lower_bound",
        ):
            value = getattr(result, name, None)

            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if isfinite(value):
                return value

        return None

    @staticmethod
    def _extract_gap(
        result: Any,
        objective: float | None,
        lower_bound: float | None,
    ) -> float | None:
        for name in (
            "mip_gap",
            "relative_gap",
        ):
            value = getattr(result, name, None)

            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if isfinite(value) and value >= 0.0:
                return value

        if (
            objective is None
            or lower_bound is None
        ):
            return None

        denominator = max(
            abs(objective),
            1e-12,
        )

        return max(
            0.0,
            (objective - lower_bound) / denominator,
        )

    @staticmethod
    def _decode_milp_route(
        solution: Sequence[float],
        x_index: Mapping[tuple[int, int], int],
        arcs: Sequence[MILPArc],
        vehicle_ids: Sequence[int],
    ) -> ReferenceRoute:
        routes: dict[int, list[int]] = {
            vehicle_id: []
            for vehicle_id in vehicle_ids
        }

        selected: dict[int, list[MILPArc]] = {
            vehicle_id: []
            for vehicle_id in vehicle_ids
        }

        for arc_id, arc in enumerate(arcs):
            for vehicle_id in vehicle_ids:
                index = x_index[
                    (arc_id, vehicle_id)
                ]

                if float(solution[index]) > 0.5:
                    selected[vehicle_id].append(arc)

        for vehicle_id in vehicle_ids:
            arcs_for_vehicle = selected[vehicle_id]

            if not arcs_for_vehicle:
                continue

            outgoing: dict[int, MILPArc] = {
                arc.origin: arc
                for arc in arcs_for_vehicle
            }

            current = arcs_for_vehicle[0].origin
            visited: set[int] = set()

            while current in outgoing:
                if current in visited:
                    break

                visited.add(current)

                arc = outgoing[current]
                current = arc.destination

                if current == self.problem.depot_end:
                    break

                if current in {
                    customer.node_id
                    for customer in self.problem.customers
                }:
                    routes[vehicle_id].append(current)

        return ReferenceRoute.from_routes(routes)


# ----------------------------------------------------------------------
# Public comparison helpers
# ----------------------------------------------------------------------


def relative_optimality_gap(
    objective: float,
    reference_objective: float,
) -> float:
    """
    Compute a certified/reference gap.

    This function does not decide whether the reference is certified.
    Callers must establish that separately.
    """

    objective = float(objective)
    reference_objective = float(reference_objective)

    if not isfinite(objective):
        raise ReferenceSolverError(
            "objective must be finite"
        )

    if not isfinite(reference_objective):
        raise ReferenceSolverError(
            "reference_objective must be finite"
        )

    if reference_objective <= 0.0:
        raise ReferenceSolverError(
            "relative gap requires a positive reference objective"
        )

    return (
        objective - reference_objective
    ) / reference_objective


def absolute_difference(
    objective: float,
    reference_objective: float,
) -> float:
    """Return absolute objective difference."""

    objective = float(objective)
    reference_objective = float(reference_objective)

    if not isfinite(objective) or not isfinite(
        reference_objective
    ):
        raise ReferenceSolverError(
            "objectives must be finite"
        )

    return abs(
        objective - reference_objective
    )


def certified_gap_or_none(
    objective: float,
    exact_result: ExactReferenceResult,
) -> float | None:
    """
    Return a relative gap only when the exact reference is certified and
    strictly positive.

    This prevents accidental reporting of a time-limited MILP bound as an
    optimum.
    """

    if not exact_result.certified_optimum:
        return None

    if exact_result.objective is None:
        return None

    if exact_result.objective <= 0.0:
        return None

    return relative_optimality_gap(
        objective,
        exact_result.objective,
    )


def best_known_gap(
    objective: float,
    reference_objective: float,
) -> float:
    """
    Relative gap to a best-known feasible reference.

    This is deliberately named differently from ``certified_gap``.
    """

    return relative_optimality_gap(
        objective,
        reference_objective,
    )


__all__ = [
    "ALNSCandidate",
    "ALNSConfig",
    "ALNSReference",
    "ALNSReferenceResult",
    "ExactReferenceResult",
    "ExactSolverUnavailable",
    "ExactStatus",
    "MILPArc",
    "MILPNode",
    "MILPVehicle",
    "ReferenceMethod",
    "ReferenceRoute",
    "ReferenceSolverError",
    "SnapshotCVRPTW",
    "SnapshotMILPReference",
    "absolute_difference",
    "best_known_gap",
    "certified_gap_or_none",
    "relative_optimality_gap",
]