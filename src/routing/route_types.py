"""Physical route representations for Step 4.

These types represent the road-network realization of a logical RoutePlan.
They are deliberately separate from optimization decisions so that the same
logical stop sequence can be evaluated against different network states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.contracts.core_types import (
    DistanceM,
    RoadEdgeId,
    RoadNodeId,
    TimeS,
    VehicleId,
)


@dataclass(frozen=True)
class RouteLeg:
    """Physical directed road path between two consecutive route nodes."""

    from_node: RoadNodeId
    to_node: RoadNodeId
    edge_ids: tuple[RoadEdgeId, ...]
    distance_m: DistanceM
    travel_time_s: TimeS

    @classmethod
    def from_sequence(
        cls,
        from_node: RoadNodeId,
        to_node: RoadNodeId,
        edge_ids: Sequence[RoadEdgeId] | Iterable[RoadEdgeId],
        distance_m: DistanceM,
        travel_time_s: TimeS,
    ) -> "RouteLeg":
        return cls(
            from_node=from_node,
            to_node=to_node,
            edge_ids=tuple(edge_ids),
            distance_m=distance_m,
            travel_time_s=travel_time_s,
        )

    @property
    def edge_count(self) -> int:
        return len(self.edge_ids)

    @property
    def is_empty(self) -> bool:
        return not self.edge_ids


@dataclass(frozen=True)
class PhysicalRoute:
    """Complete road-network realization for one vehicle."""

    vehicle_id: VehicleId
    start_node: RoadNodeId
    end_node: RoadNodeId
    legs: tuple[RouteLeg, ...]

    @classmethod
    def from_legs(
        cls,
        vehicle_id: VehicleId,
        start_node: RoadNodeId,
        end_node: RoadNodeId,
        legs: Sequence[RouteLeg] | Iterable[RouteLeg],
    ) -> "PhysicalRoute":
        return cls(
            vehicle_id=vehicle_id,
            start_node=start_node,
            end_node=end_node,
            legs=tuple(legs),
        )

    @property
    def total_distance_m(self) -> DistanceM:
        return sum(leg.distance_m for leg in self.legs)

    @property
    def total_travel_time_s(self) -> TimeS:
        return sum(leg.travel_time_s for leg in self.legs)

    @property
    def total_edge_count(self) -> int:
        return sum(leg.edge_count for leg in self.legs)

    @property
    def edge_ids(self) -> tuple[RoadEdgeId, ...]:
        """Flatten all physical directed edges in traversal order."""
        return tuple(
            edge_id
            for leg in self.legs
            for edge_id in leg.edge_ids
        )


@dataclass(frozen=True)
class PhysicalRoutePlan:
    """Physical road-network realization of a complete RoutePlan."""

    routes: tuple[PhysicalRoute, ...]

    @classmethod
    def from_routes(
        cls,
        routes: Sequence[PhysicalRoute] | Iterable[PhysicalRoute],
    ) -> "PhysicalRoutePlan":
        return cls(routes=tuple(routes))

    @property
    def route_count(self) -> int:
        return len(self.routes)

    @property
    def total_distance_m(self) -> DistanceM:
        return sum(route.total_distance_m for route in self.routes)

    @property
    def total_travel_time_s(self) -> TimeS:
        return sum(route.total_travel_time_s for route in self.routes)

    @property
    def total_edge_count(self) -> int:
        return sum(route.total_edge_count for route in self.routes)

    def route_for(self, vehicle_id: VehicleId) -> PhysicalRoute:
        for route in self.routes:
            if route.vehicle_id == vehicle_id:
                return route

        raise KeyError(
            f"No physical route found for vehicle_id={vehicle_id!r}"
        )