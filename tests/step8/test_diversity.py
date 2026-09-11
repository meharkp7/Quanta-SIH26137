from __future__ import annotations

import pytest

from src.optim.diversity import (
    DiversityMetricError,
    assignment_distance,
    coordinate_population_diversity,
    mean_pairwise_route_distance,
    population_diversity,
    precedence_distance,
    repair_displacement,
    route_distance,
)
from src.routing.route_plan import RoutePlan, VehicleRoute


def plan(*routes):
    return RoutePlan.from_routes(
        VehicleRoute.from_sequence(vehicle_id, customers)
        for vehicle_id, customers in routes
    )


def test_identical_plans_have_zero_distance():
    first = plan(
        (1, (101, 102, 103)),
        (2, (104, 105)),
    )

    second = plan(
        (1, (101, 102, 103)),
        (2, (104, 105)),
    )

    distance = route_distance(first, second)

    assert distance.assignment == 0.0
    assert distance.precedence == 0.0
    assert distance.structure == 0.0
    assert distance.exact == 0.0


def test_vehicle_assignment_change_is_detected():
    first = plan(
        (1, (101, 102)),
        (2, (103, 104)),
    )

    second = plan(
        (1, (101,)),
        (2, (102, 103, 104)),
    )

    distance = route_distance(first, second)

    assert distance.assignment == pytest.approx(0.25)
    assert distance.precedence == 0.0
    assert distance.structure == pytest.approx(0.125)
    assert distance.exact == 1.0


def test_order_change_is_detected_without_assignment_change():
    first = plan(
        (1, (101, 102, 103, 104)),
        (2, (105,)),
    )

    second = plan(
        (1, (101, 103, 102, 104)),
        (2, (105,)),
    )

    assert assignment_distance(first, second) == 0.0

    precedence = precedence_distance(first, second)

    # Of the six customer pairs on vehicle 1, only (102, 103) changes.
    assert precedence == pytest.approx(1.0 / 6.0)


def test_assignment_changes_are_not_double_counted_as_precedence():
    first = plan(
        (1, (101, 102)),
        (2, (103, 104)),
    )

    second = plan(
        (1, (101, 103)),
        (2, (102, 104)),
    )

    assert assignment_distance(first, second) == pytest.approx(0.5)

    # No pair is on the same vehicle in both plans.
    assert precedence_distance(first, second) == 0.0


def test_empty_routes_are_handled():
    first = plan(
        (1, ()),
        (2, (101, 102)),
    )

    second = plan(
        (1, ()),
        (2, (101, 102)),
    )

    assert route_distance(first, second).structure == 0.0


def test_singleton_population_has_zero_diversity():
    route = plan(
        (1, (101, 102)),
        (2, (103,)),
    )

    distance = mean_pairwise_route_distance((route,))

    assert distance.assignment == 0.0
    assert distance.precedence == 0.0
    assert distance.structure == 0.0
    assert distance.exact == 0.0


def test_coordinate_diversity_is_zero_for_identical_population():
    population = (
        (0.25, 0.50),
        (0.25, 0.50),
        (0.25, 0.50),
    )

    assert coordinate_population_diversity(population) == 0.0


def test_coordinate_diversity_increases_with_spread():
    compact = (
        (0.49, 0.49),
        (0.51, 0.51),
    )

    spread = (
        (0.0, 0.0),
        (1.0, 1.0),
    )

    assert (
        coordinate_population_diversity(spread)
        > coordinate_population_diversity(compact)
    )


def test_coordinate_diversity_is_normalized():
    population = (
        (0.0, 0.0),
        (1.0, 1.0),
    )

    value = coordinate_population_diversity(population)

    assert 0.0 <= value <= 1.0


def test_coordinate_diversity_rejects_mismatched_dimensions():
    with pytest.raises(DiversityMetricError):
        coordinate_population_diversity(
            (
                (0.1, 0.2),
                (0.3,),
            )
        )


def test_route_distance_rejects_different_customer_sets():
    first = plan(
        (1, (101, 102)),
        (2, (103,)),
    )

    second = plan(
        (1, (101, 102)),
        (2, (104,)),
    )

    with pytest.raises(DiversityMetricError):
        route_distance(first, second)


def test_duplicate_customer_is_rejected():
    first = plan(
        (1, (101, 102)),
        (2, (102, 103)),
    )

    second = plan(
        (1, (101, 102)),
        (2, (102, 103)),
    )

    with pytest.raises(DiversityMetricError):
        route_distance(first, second)


def test_repair_displacement_is_zero_when_repair_does_nothing():
    route = plan(
        (1, (101, 102)),
        (2, (103,)),
    )

    displacement = repair_displacement(route, route)

    assert displacement.assignment == 0.0
    assert displacement.precedence == 0.0
    assert displacement.structure == 0.0
    assert displacement.exact == 0.0


def test_population_diversity_keeps_representation_layers_separate():
    decoded_a = plan(
        (1, (101, 102)),
        (2, (103, 104)),
    )

    decoded_b = plan(
        (1, (101, 103)),
        (2, (102, 104)),
    )

    repaired_a = plan(
        (1, (101, 102)),
        (2, (103, 104)),
    )

    repaired_b = plan(
        (1, (101, 102)),
        (2, (103, 104)),
    )

    metrics = population_diversity(
        positions=(
            (0.10, 0.20),
            (0.90, 0.80),
        ),
        decoded_plans=(decoded_a, decoded_b),
        repaired_plans=(repaired_a, repaired_b),
    )

    assert metrics.genotype_coordinate > 0.0
    assert metrics.decoded_assignment > 0.0
    assert metrics.repaired_assignment == 0.0