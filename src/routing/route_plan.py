"""Logical route-plan representations for Step 4.

This module represents the logical solution produced by an initial routing
algorithm. It intentionally does not construct physical road paths; that is
handled by the path-builder layer.

Design goals:
- generic for arbitrary numbers of vehicles and customers
- immutable route objects for reproducibility
- deterministic ordering
- no dependency on a particular optimization algorithm
- suitable for initial routing, re-routing, and later QPSO warm starts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.contracts.core_types import CustomerId, RoadNodeId, VehicleId


@dataclass(frozen=True)
class VehicleRoute:
    """Logical route assigned to a single vehicle.

    ``customer_ids`` contains only the ordered customer sequence.
    Depot departure and depot return are implicit and are resolved from the
    scenario/fleet definition.

    Example:
        V1 -> (C17, C42, C91)
    """

    vehicle_id: VehicleId
    customer_ids: tuple[CustomerId, ...]

    @classmethod
    def from_sequence(
        cls,
        vehicle_id: VehicleId,
        customer_ids: Sequence[CustomerId] | Iterable[CustomerId],
    ) -> "VehicleRoute":
        return cls(
            vehicle_id=vehicle_id,
            customer_ids=tuple(customer_ids),
        )

    @property
    def stop_count(self) -> int:
        """Number of customer stops assigned to the vehicle."""
        return len(self.customer_ids)

    @property
    def is_empty(self) -> bool:
        """Whether the vehicle has no assigned customers."""
        return not self.customer_ids

    def contains(self, customer_id: CustomerId) -> bool:
        """Return whether this route contains a customer."""
        return customer_id in self.customer_ids


@dataclass(frozen=True)
class RoutePlan:
    """Complete logical multi-vehicle routing solution.

    The plan contains assignments and visit order, but not physical road
    edges, travel times, or simulation results.

    This separation allows the same logical plan to be evaluated against
    different dynamic road-network states.
    """

    vehicle_routes: tuple[VehicleRoute, ...]

    @classmethod
    def from_routes(
        cls,
        vehicle_routes: Sequence[VehicleRoute] | Iterable[VehicleRoute],
    ) -> "RoutePlan":
        routes = tuple(vehicle_routes)

        vehicle_ids = [route.vehicle_id for route in routes]
        if len(vehicle_ids) != len(set(vehicle_ids)):
            raise ValueError("RoutePlan contains duplicate vehicle IDs")

        return cls(vehicle_routes=routes)

    @property
    def vehicle_count(self) -> int:
        """Number of vehicle routes represented by the plan."""
        return len(self.vehicle_routes)

    @property
    def non_empty_vehicle_count(self) -> int:
        """Number of vehicles serving at least one customer."""
        return sum(not route.is_empty for route in self.vehicle_routes)

    @property
    def total_stop_count(self) -> int:
        """Total number of customer visits across all vehicles."""
        return sum(route.stop_count for route in self.vehicle_routes)

    def route_for(self, vehicle_id: VehicleId) -> VehicleRoute:
        """Return the route assigned to ``vehicle_id``."""
        for route in self.vehicle_routes:
            if route.vehicle_id == vehicle_id:
                return route

        raise KeyError(f"No route found for vehicle_id={vehicle_id!r}")

    def all_customer_ids(self) -> tuple[CustomerId, ...]:
        """Return all customer IDs in vehicle/visit order."""
        return tuple(
            customer_id
            for route in self.vehicle_routes
            for customer_id in route.customer_ids
        )

    def unique_customer_ids(self) -> tuple[CustomerId, ...]:
        """Return customers in first-seen order without duplicates."""
        seen: set[CustomerId] = set()
        result: list[CustomerId] = []

        for customer_id in self.all_customer_ids():
            if customer_id not in seen:
                seen.add(customer_id)
                result.append(customer_id)

        return tuple(result)

    def duplicate_customer_ids(self) -> tuple[CustomerId, ...]:
        """Return customer IDs appearing more than once."""
        counts: dict[CustomerId, int] = {}

        for customer_id in self.all_customer_ids():
            counts[customer_id] = counts.get(customer_id, 0) + 1

        return tuple(
            customer_id
            for customer_id in self.all_customer_ids()
            if counts[customer_id] > 1
        )

    def is_customer_unique(self) -> bool:
        """Return whether every customer occurs at most once."""
        return len(self.all_customer_ids()) == len(self.unique_customer_ids())


@dataclass(frozen=True)
class StopLocation:
    """Mapping from a customer request to its road-network node."""

    customer_id: CustomerId
    node_id: RoadNodeId