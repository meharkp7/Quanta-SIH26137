"""
Representation-aware diversity metrics for routing optimization.

Step 8B-1
---------

This module measures diversity at multiple semantic layers of the Step 7
continuous-key routing representation.

The central distinction is:

    continuous particle
        -> decoded logical RoutePlan
        -> repaired logical RoutePlan

The module intentionally does NOT:
    * evaluate routes,
    * repair routes,
    * mutate RoutePlan objects,
    * invoke randomness,
    * depend on an optimizer.

All functions are deterministic and side-effect free.

Metric layers
-------------

1. Genotype:
       continuous-coordinate diversity

2. Decoded phenotype:
       assignment diversity
       precedence/order diversity
       structural route diversity

3. Repaired phenotype:
       assignment diversity
       precedence/order diversity
       structural route diversity

The module does not collapse these measurements into one composite diversity
score. Any weighting belongs to a later experimental analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite, sqrt
from typing import Iterable, Sequence

from src.contracts.core_types import CustomerId, VehicleId
from src.routing.route_plan import RoutePlan


class DiversityMetricError(ValueError):
    """Raised when a diversity metric receives invalid input."""


@dataclass(frozen=True)
class RouteDistance:
    """
    Decomposed distance between two logical route plans.

    All components are normalized to [0, 1].

    assignment:
        Fraction of customers assigned to different vehicles.

    precedence:
        Fraction of unordered customer pairs whose relative precedence
        relationship differs.

    structure:
        Average of assignment and precedence distance.

    exact:
        1.0 when the two plans differ, otherwise 0.0.
    """

    assignment: float
    precedence: float
    structure: float
    exact: float

    def __post_init__(self) -> None:
        for name, value in (
            ("assignment", self.assignment),
            ("precedence", self.precedence),
            ("structure", self.structure),
            ("exact", self.exact),
        ):
            if not isfinite(float(value)):
                raise DiversityMetricError(
                    f"{name} distance must be finite"
                )

            if not 0.0 <= float(value) <= 1.0:
                raise DiversityMetricError(
                    f"{name} distance must lie in [0, 1]"
                )


@dataclass(frozen=True)
class PopulationDiversity:
    """
    Aggregate diversity statistics for one population.

    The fields intentionally remain separate. This prevents an arbitrary
    composite metric from being silently introduced into later experiments.
    """

    genotype_coordinate: float

    decoded_assignment: float
    decoded_precedence: float
    decoded_structure: float

    repaired_assignment: float
    repaired_precedence: float
    repaired_structure: float

    def __post_init__(self) -> None:
        fields = (
            "genotype_coordinate",
            "decoded_assignment",
            "decoded_precedence",
            "decoded_structure",
            "repaired_assignment",
            "repaired_precedence",
            "repaired_structure",
        )

        for field_name in fields:
            value = float(getattr(self, field_name))

            if not isfinite(value):
                raise DiversityMetricError(
                    f"{field_name} must be finite"
                )

            if not 0.0 <= value <= 1.0:
                raise DiversityMetricError(
                    f"{field_name} must lie in [0, 1]"
                )


def _validate_vector_population(
    population: Sequence[Sequence[float]],
) -> None:
    if not population:
        return

    dimensions = len(population[0])

    if dimensions == 0:
        raise DiversityMetricError(
            "population vectors cannot have zero dimensions"
        )

    for index, vector in enumerate(population):
        if len(vector) != dimensions:
            raise DiversityMetricError(
                "population vectors have inconsistent dimensions: "
                f"index 0 has {dimensions}, index {index} has {len(vector)}"
            )

        for coordinate in vector:
            if not isfinite(float(coordinate)):
                raise DiversityMetricError(
                    "population coordinates must be finite"
                )


def coordinate_population_diversity(
    population: Sequence[Sequence[float]],
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> float:
    """
    Compute normalized mean Euclidean distance from the population centroid.

    The normalization corresponds to the maximum centroid distance scale
    for a bounded d-dimensional hypercube:

        sqrt(d) / 2

    for the full [lower_bound, upper_bound]^d domain.

    The result is clipped only for numerical robustness and therefore lies
    in [0, 1].
    """

    _validate_vector_population(population)

    if not isfinite(float(lower_bound)):
        raise DiversityMetricError("lower_bound must be finite")

    if not isfinite(float(upper_bound)):
        raise DiversityMetricError("upper_bound must be finite")

    if lower_bound >= upper_bound:
        raise DiversityMetricError(
            "lower_bound must be strictly smaller than upper_bound"
        )

    if not population:
        return 0.0

    dimensions = len(population[0])

    centroid = tuple(
        sum(float(vector[d]) for vector in population)
        / len(population)
        for d in range(dimensions)
    )

    mean_distance = sum(
        sqrt(
            sum(
                (float(vector[d]) - centroid[d]) ** 2
                for d in range(dimensions)
            )
        )
        for vector in population
    ) / len(population)

    half_diagonal = (
        (upper_bound - lower_bound)
        * sqrt(dimensions)
        / 2.0
    )

    if half_diagonal <= 0.0:
        return 0.0

    return min(
        1.0,
        max(0.0, mean_distance / half_diagonal),
    )


def _canonical_routes(
    route_plan: RoutePlan,
) -> tuple[tuple[VehicleId, tuple[CustomerId, ...]], ...]:
    """
    Return a deterministic vehicle-indexed representation.

    RoutePlan already preserves route ordering, but sorting vehicle IDs here
    ensures that metric calculations do not accidentally depend on the order
    in which a caller constructed a semantically equivalent plan.
    """

    return tuple(
        (
            route.vehicle_id,
            tuple(route.customer_ids),
        )
        for route in sorted(
            route_plan.vehicle_routes,
            key=lambda route: str(route.vehicle_id),
        )
    )


def _customer_set(
    route_plan: RoutePlan,
) -> frozenset[CustomerId]:
    return frozenset(route_plan.all_customer_ids())


def _assignment_map(
    route_plan: RoutePlan,
) -> dict[CustomerId, VehicleId]:
    """
    Map every unique customer to its assigned vehicle.

    Duplicate customers are intentionally rejected.

    Diversity is a structural property of a route plan. Silently choosing the
    first occurrence of a duplicate would make the metric depend on an
    invalid representation.
    """

    assignments: dict[CustomerId, VehicleId] = {}

    for vehicle_route in route_plan.vehicle_routes:
        for customer_id in vehicle_route.customer_ids:
            if customer_id in assignments:
                raise DiversityMetricError(
                    "route plan contains duplicate customer "
                    f"{customer_id!r}"
                )

            assignments[customer_id] = vehicle_route.vehicle_id

    return assignments


def _precedence_map(
    route_plan: RoutePlan,
) -> dict[tuple[CustomerId, CustomerId], int]:
    """
    Encode pairwise customer precedence.

    For each unordered pair (a, b):

        +1 -> a occurs before b
        -1 -> b occurs before a
         0 -> they are assigned to different vehicles

    Pairs assigned to different vehicles deliberately contribute through
    assignment distance rather than pretending that a global precedence
    relation exists between separate vehicle routes.
    """

    result: dict[tuple[CustomerId, CustomerId], int] = {}

    for vehicle_route in route_plan.vehicle_routes:
        customers = tuple(vehicle_route.customer_ids)

        if len(customers) != len(set(customers)):
            raise DiversityMetricError(
                f"vehicle {vehicle_route.vehicle_id!r} contains duplicate "
                "customers"
            )

        for left_index in range(len(customers)):
            for right_index in range(
                left_index + 1,
                len(customers),
            ):
                left = customers[left_index]
                right = customers[right_index]

                pair = _ordered_customer_pair(left, right)

                # Store orientation relative to the canonical pair.
                if left == pair[0]:
                    result[pair] = 1
                else:
                    result[pair] = -1

    return result


def _ordered_customer_pair(
    first: CustomerId,
    second: CustomerId,
) -> tuple[CustomerId, CustomerId]:
    """
    Produce a deterministic unordered-pair representation.

    Customer IDs may be integers, strings, UUID-like values, etc., so we
    compare their string representations rather than assuming a numeric type.
    """

    if str(first) <= str(second):
        return first, second

    return second, first


def assignment_distance(
    first: RoutePlan,
    second: RoutePlan,
) -> float:
    """
    Fraction of customers assigned to different vehicles.

    The plans must contain the same unique customer set.
    """

    first_assignments = _assignment_map(first)
    second_assignments = _assignment_map(second)

    first_customers = _customer_set(first)
    second_customers = _customer_set(second)

    if first_customers != second_customers:
        raise DiversityMetricError(
            "route plans must contain the same customer set"
        )

    if not first_customers:
        return 0.0

    differing = sum(
        first_assignments[cid] != second_assignments[cid]
        for cid in first_customers
    )

    return differing / len(first_customers)


def precedence_distance(
    first: RoutePlan,
    second: RoutePlan,
) -> float:
    """
    Fraction of comparable customer pairs whose ordering differs.

    A pair is comparable only when both customers are assigned to the same
    vehicle in both plans.

    This prevents the metric from double-counting assignment changes as
    ordering changes.
    """

    first_assignments = _assignment_map(first)
    second_assignments = _assignment_map(second)

    customers = _customer_set(first)

    if customers != _customer_set(second):
        raise DiversityMetricError(
            "route plans must contain the same customer set"
        )

    first_precedence = _precedence_map(first)
    second_precedence = _precedence_map(second)

    disagreements = 0
    comparable_pairs = 0

    for first_customer, second_customer in combinations(
        sorted(customers, key=str),
        2,
    ):
        same_vehicle = (
            first_assignments[first_customer]
            == first_assignments[second_customer]
            and second_assignments[first_customer]
            == second_assignments[second_customer]
        )

        if not same_vehicle:
            continue

        comparable_pairs += 1

        pair = _ordered_customer_pair(
            first_customer,
            second_customer,
        )

        if first_precedence.get(pair) != second_precedence.get(pair):
            disagreements += 1

    if comparable_pairs == 0:
        return 0.0

    return disagreements / comparable_pairs


def route_distance(
    first: RoutePlan,
    second: RoutePlan,
) -> RouteDistance:
    """
    Compute the complete decomposed logical route distance.
    """

    assignment = assignment_distance(first, second)
    precedence = precedence_distance(first, second)

    return RouteDistance(
        assignment=assignment,
        precedence=precedence,
        structure=(assignment + precedence) / 2.0,
        exact=0.0 if _canonical_routes(first) == _canonical_routes(second) else 1.0,
    )


def mean_pairwise_route_distance(
    route_plans: Sequence[RoutePlan],
) -> RouteDistance:
    """
    Mean pairwise logical-route distance.

    Empty and singleton populations have zero diversity.

    All route plans must contain the same customer set.
    """

    if len(route_plans) < 2:
        return RouteDistance(
            assignment=0.0,
            precedence=0.0,
            structure=0.0,
            exact=0.0,
        )

    distances = [
        route_distance(first, second)
        for first, second in combinations(route_plans, 2)
    ]

    count = len(distances)

    return RouteDistance(
        assignment=sum(d.assignment for d in distances) / count,
        precedence=sum(d.precedence for d in distances) / count,
        structure=sum(d.structure for d in distances) / count,
        exact=sum(d.exact for d in distances) / count,
    )


def repair_displacement(
    original: RoutePlan,
    repaired: RoutePlan,
) -> RouteDistance:
    """
    Measure how much the repair operator changed a decoded plan.

    The full decomposed distance is returned so later experiments can
    determine whether repair changes assignments, order, or both.
    """

    return route_distance(original, repaired)


def population_repair_pressure(
    decoded_plans: Sequence[RoutePlan],
    repaired_plans: Sequence[RoutePlan],
) -> RouteDistance:
    """
    Aggregate decoded->repaired displacement across a population.

    The two sequences must be aligned one-to-one.
    """

    if len(decoded_plans) != len(repaired_plans):
        raise DiversityMetricError(
            "decoded and repaired populations must have equal size"
        )

    if not decoded_plans:
        return RouteDistance(
            assignment=0.0,
            precedence=0.0,
            structure=0.0,
            exact=0.0,
        )

    distances = [
        repair_displacement(decoded, repaired)
        for decoded, repaired in zip(
            decoded_plans,
            repaired_plans,
        )
    ]

    count = len(distances)

    return RouteDistance(
        assignment=sum(d.assignment for d in distances) / count,
        precedence=sum(d.precedence for d in distances) / count,
        structure=sum(d.structure for d in distances) / count,
        exact=sum(d.exact for d in distances) / count,
    )


def population_diversity(
    positions: Sequence[Sequence[float]],
    decoded_plans: Sequence[RoutePlan],
    repaired_plans: Sequence[RoutePlan],
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> PopulationDiversity:
    """
    Compute all Step 8B-1 diversity observables for one aligned population.

    ``positions[i]``, ``decoded_plans[i]`` and ``repaired_plans[i]`` must refer
    to the same candidate.

    This function performs no routing evaluation and no repair.
    """

    if len(positions) != len(decoded_plans):
        raise DiversityMetricError(
            "positions and decoded_plans must have equal size"
        )

    if len(decoded_plans) != len(repaired_plans):
        raise DiversityMetricError(
            "decoded_plans and repaired_plans must have equal size"
        )

    coordinate = coordinate_population_diversity(
        positions,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    decoded = mean_pairwise_route_distance(
        decoded_plans,
    )

    repaired = mean_pairwise_route_distance(
        repaired_plans,
    )

    return PopulationDiversity(
        genotype_coordinate=coordinate,
        decoded_assignment=decoded.assignment,
        decoded_precedence=decoded.precedence,
        decoded_structure=decoded.structure,
        repaired_assignment=repaired.assignment,
        repaired_precedence=repaired.precedence,
        repaired_structure=repaired.structure,
    )