"""Physical route representations for Step 5.

These types represent the road-network realization of a logical RoutePlan.

Logical optimization decisions remain separate from physical road paths.
The physical representation preserves the complete cost decomposition of
each selected path.
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
    free_flow_time_s: TimeS = 0.0
    congestion_delay_s: TimeS = 0.0

    @classmethod
    def from_sequence(
        cls,
        from_node: RoadNodeId,
        to_node: RoadNodeId,
        edge_ids: Sequence[RoadEdgeId] | Iterable[RoadEdgeId],
        distance_m: DistanceM,
        travel_time_s: TimeS,
        *,
        free_flow_time_s: TimeS | None = None,
        congestion_delay_s: TimeS | None = None,
    ) -> "RouteLeg":
        travel = float(travel_time_s)

        if free_flow_time_s is None:
            free_flow = travel
        else:
            free_flow = float(free_flow_time_s)

        if congestion_delay_s is None:
            congestion = max(
                0.0,
                travel - free_flow,
            )
        else:
            congestion = float(congestion_delay_s)

        return cls(
            from_node=from_node,
            to_node=to_node,
            edge_ids=tuple(edge_ids),
            distance_m=float(distance_m),
            travel_time_s=travel,
            free_flow_time_s=free_flow,
            congestion_delay_s=congestion,
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
        return sum(
            leg.distance_m
            for leg in self.legs
        )

    @property
    def total_travel_time_s(self) -> TimeS:
        return sum(
            leg.travel_time_s
            for leg in self.legs
        )

    @property
    def total_free_flow_time_s(self) -> TimeS:
        return sum(
            leg.free_flow_time_s
            for leg in self.legs
        )

    @property
    def total_congestion_delay_s(self) -> TimeS:
        return sum(
            leg.congestion_delay_s
            for leg in self.legs
        )

    @property
    def total_edge_count(self) -> int:
        return sum(
            leg.edge_count
            for leg in self.legs
        )

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
        return cls(
            routes=tuple(routes)
        )

    @property
    def route_count(self) -> int:
        return len(self.routes)

    @property
    def total_distance_m(self) -> DistanceM:
        return sum(
            route.total_distance_m
            for route in self.routes
        )

    @property
    def total_travel_time_s(self) -> TimeS:
        return sum(
            route.total_travel_time_s
            for route in self.routes
        )

    @property
    def total_free_flow_time_s(self) -> TimeS:
        return sum(
            route.total_free_flow_time_s
            for route in self.routes
        )

    @property
    def total_congestion_delay_s(self) -> TimeS:
        return sum(
            route.total_congestion_delay_s
            for route in self.routes
        )

    @property
    def total_edge_count(self) -> int:
        return sum(
            route.total_edge_count
            for route in self.routes
        )

    def route_for(
        self,
        vehicle_id: VehicleId,
    ) -> PhysicalRoute:
        for route in self.routes:
            if route.vehicle_id == vehicle_id:
                return route

        raise KeyError(
            f"No physical route found for "
            f"vehicle_id={vehicle_id!r}"
        )