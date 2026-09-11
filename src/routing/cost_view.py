"""Coherent road-cost views for Step 5.

A CostView is the single source of truth for physical edge costs used by
shortest-path construction and route evaluation.

For every edge traversal, the view exposes distance and travel-time
components together. This prevents distance from one network/cost state
being combined with travel time from another.

The default view is free-flow:

    free_flow_time = length_m / speed_limit_mps
    congestion_delay = 0
    travel_time = free_flow_time
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable

from src.contracts.core_types import TimeS
from src.contracts.scenario import RoadEdge


EdgeTravelTimeProvider = Callable[[RoadEdge, TimeS], TimeS]


class InvalidEdgeCostError(ValueError):
    """Raised when an edge cost is invalid."""


@dataclass(frozen=True)
class EdgeCost:
    """Complete cost decomposition for one directed road edge."""

    distance_m: float
    free_flow_time_s: float
    travel_time_s: float
    congestion_delay_s: float

    def __post_init__(self) -> None:
        values = (
            ("distance_m", self.distance_m),
            ("free_flow_time_s", self.free_flow_time_s),
            ("travel_time_s", self.travel_time_s),
            ("congestion_delay_s", self.congestion_delay_s),
        )

        for name, value in values:
            numeric = float(value)

            if not isfinite(numeric):
                raise InvalidEdgeCostError(
                    f"{name} must be finite, got {value!r}"
                )

            if numeric < 0.0:
                raise InvalidEdgeCostError(
                    f"{name} must be non-negative, got {value!r}"
                )

        expected_delay = (
            float(self.travel_time_s)
            - float(self.free_flow_time_s)
        )

        if abs(
            float(self.congestion_delay_s)
            - expected_delay
        ) > 1e-9:
            raise InvalidEdgeCostError(
                "congestion_delay_s must equal "
                "travel_time_s - free_flow_time_s"
            )


@dataclass(frozen=True)
class PathCost:
    """Aggregated cost decomposition for a physical directed path."""

    distance_m: float
    free_flow_time_s: float
    travel_time_s: float
    congestion_delay_s: float

    def __post_init__(self) -> None:
        values = (
            ("distance_m", self.distance_m),
            ("free_flow_time_s", self.free_flow_time_s),
            ("travel_time_s", self.travel_time_s),
            ("congestion_delay_s", self.congestion_delay_s),
        )

        for name, value in values:
            numeric = float(value)

            if not isfinite(numeric):
                raise InvalidEdgeCostError(
                    f"{name} must be finite, got {value!r}"
                )

            if numeric < 0.0:
                raise InvalidEdgeCostError(
                    f"{name} must be non-negative, got {value!r}"
                )

        expected_delay = (
            float(self.travel_time_s)
            - float(self.free_flow_time_s)
        )

        if abs(
            float(self.congestion_delay_s)
            - expected_delay
        ) > 1e-9:
            raise InvalidEdgeCostError(
                "congestion_delay_s must equal "
                "travel_time_s - free_flow_time_s"
            )


class CostView:
    """Immutable configuration of the road-cost model.

    The view owns the travel-time interpretation used for path selection.
    Every edge query returns the complete distance/time decomposition.
    """

    def __init__(
        self,
        *,
        graph_version: str = "graph-v1",
        cost_version: str = "cost-free-flow-v1",
        forecast_version: str = "forecast-none",
        travel_time_provider: EdgeTravelTimeProvider | None = None,
    ) -> None:
        if not graph_version:
            raise ValueError("graph_version must be non-empty")

        if not cost_version:
            raise ValueError("cost_version must be non-empty")

        if not forecast_version:
            raise ValueError("forecast_version must be non-empty")

        self.graph_version = str(graph_version)
        self.cost_version = str(cost_version)
        self.forecast_version = str(forecast_version)

        self._travel_time_provider = (
            travel_time_provider
            or self._free_flow_travel_time
        )

    @property
    def travel_time_provider(
        self,
    ) -> EdgeTravelTimeProvider:
        """Return the provider used by this cost view."""

        return self._travel_time_provider

    def edge_cost(
        self,
        edge: RoadEdge,
        *,
        departure_time_s: TimeS = 0.0,
    ) -> EdgeCost:
        """Return the complete cost of one edge at a departure time."""

        departure_time = float(departure_time_s)

        if not isfinite(departure_time):
            raise InvalidEdgeCostError(
                f"departure_time_s must be finite, "
                f"got {departure_time_s!r}"
            )

        distance = float(edge.length_m)

        if not isfinite(distance) or distance < 0.0:
            raise InvalidEdgeCostError(
                f"Edge {edge.edge_id!r} has invalid "
                f"length_m={edge.length_m!r}"
            )

        free_flow = self._free_flow_travel_time(
            edge,
            departure_time,
        )

        travel_time = self._validate_travel_time(
            self._travel_time_provider(
                edge,
                departure_time,
            ),
            edge=edge,
            departure_time_s=departure_time,
        )

        congestion_delay = (
            travel_time - free_flow
        )

        if congestion_delay < -1e-9:
            raise InvalidEdgeCostError(
                f"Travel time for edge {edge.edge_id!r} "
                f"cannot be below free-flow time: "
                f"{travel_time} < {free_flow}"
            )

        congestion_delay = max(
            0.0,
            congestion_delay,
        )

        return EdgeCost(
            distance_m=distance,
            free_flow_time_s=free_flow,
            travel_time_s=travel_time,
            congestion_delay_s=congestion_delay,
        )

    def path_cost(
        self,
        edges: tuple[RoadEdge, ...],
        *,
        departure_time_s: TimeS = 0.0,
    ) -> PathCost:
        """Aggregate coherent costs across a physical path.

        Edge departure times are propagated sequentially, so dynamic traffic
        conditions are evaluated at the time the vehicle reaches each edge.
        """

        current_time = float(departure_time_s)

        if not isfinite(current_time):
            raise InvalidEdgeCostError(
                f"departure_time_s must be finite, "
                f"got {departure_time_s!r}"
            )

        distance = 0.0
        free_flow_time = 0.0
        travel_time = 0.0
        congestion_delay = 0.0

        for edge in edges:
            cost = self.edge_cost(
                edge,
                departure_time_s=current_time,
            )

            distance += cost.distance_m
            free_flow_time += cost.free_flow_time_s
            travel_time += cost.travel_time_s
            congestion_delay += cost.congestion_delay_s

            current_time += cost.travel_time_s

        return PathCost(
            distance_m=distance,
            free_flow_time_s=free_flow_time,
            travel_time_s=travel_time,
            congestion_delay_s=congestion_delay,
        )

    @staticmethod
    def _free_flow_travel_time(
        edge: RoadEdge,
        departure_time_s: TimeS,
    ) -> TimeS:
        del departure_time_s

        length = float(edge.length_m)
        speed = float(edge.speed_limit_mps)

        if not isfinite(length) or length < 0.0:
            raise InvalidEdgeCostError(
                f"Edge {edge.edge_id!r} has invalid "
                f"length_m={length!r}"
            )

        if not isfinite(speed) or speed <= 0.0:
            raise InvalidEdgeCostError(
                f"Edge {edge.edge_id!r} has invalid "
                f"speed_limit_mps={speed!r}"
            )

        return length / speed

    @staticmethod
    def _validate_travel_time(
        travel_time_s: TimeS,
        *,
        edge: RoadEdge,
        departure_time_s: TimeS,
    ) -> float:
        value = float(travel_time_s)

        if not isfinite(value):
            raise InvalidEdgeCostError(
                f"Travel-time provider returned non-finite "
                f"value {value!r} for edge={edge.edge_id!r} "
                f"at t={departure_time_s}"
            )

        if value < 0.0:
            raise InvalidEdgeCostError(
                f"Travel-time provider returned negative "
                f"value {value!r} for edge={edge.edge_id!r} "
                f"at t={departure_time_s}"
            )

        return value