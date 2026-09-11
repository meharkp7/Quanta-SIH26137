"""Coherent, versioned, cached road-cost views.

This module implements Step 5 (road costs and the independent route
evaluator) together with the caching requirement from the same step:

    "For each selected physical path retain its distance, time and
    congestion components together. Never combine distance from one
    path with time from another."

    "Cache paths by graph/cost/forecast version and departure bucket;
    invalidate affected entries after a change."

``edge_cost``/``path_cost`` give the coherent distance/time/congestion
decomposition. ``shortest_path``/``invalidate`` give the versioned,
bounded cache used by callers that only need the winning path (route
construction, ALNS/QPSO evaluation, SUMO integration in Step 6).

Both surfaces share one ``DirectedPathBuilder`` and one travel-time
provider, so a cached ``PathResult`` and a freshly computed
``PathCost`` for the same edges are always consistent with each other.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import isfinite

#from typing import Callable
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .path_builder import DirectedPathBuilder

from src.contracts.core_types import TimeS
from src.contracts.scenario import RoadEdge

#from src.routing.path_builder import DirectedPathBuilder, PathResult
if TYPE_CHECKING:
    from src.routing.path_builder import DirectedPathBuilder, PathResult

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
            float(self.travel_time_s) - float(self.free_flow_time_s)
        )

        if abs(float(self.congestion_delay_s) - expected_delay) > 1e-9:
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
            float(self.travel_time_s) - float(self.free_flow_time_s)
        )

        if abs(float(self.congestion_delay_s) - expected_delay) > 1e-9:
            raise InvalidEdgeCostError(
                "congestion_delay_s must equal "
                "travel_time_s - free_flow_time_s"
            )


class CostView:
    """Immutable-per-version, cached, coherent road-cost model.

    ``edge_cost``/``path_cost`` return the full distance/free-flow/
    travel-time/congestion decomposition for a single edge or a
    physical path, validating internal consistency on every call.

    ``shortest_path`` returns the winning ``PathResult`` between two
    nodes, cached by (graph_version, cost_version, forecast_version,
    source, target, departure bucket). ``invalidate`` clears the cache
    when cost or forecast versions change; an edge decrease can improve
    paths that were not previously using it, so entries are cleared in
    full rather than selectively.

    Both surfaces share the same travel-time provider, so a cached
    ``PathResult`` and a freshly computed ``PathCost`` for its edges are
    always consistent with each other.
    """

    def __init__(
        self,
        edges=(),
        *,
        graph_version: str = "deafult",
        cost_version: str = "free-flow",
        forecast_version: str | None = None,
        closed_edge_ids=(),
        travel_time_provider: EdgeTravelTimeProvider | None = None,
        max_entries: int = 4096,
    ) -> None:
        if not graph_version:
            raise ValueError("graph_version must be non-empty")

        if not cost_version:
            raise ValueError("cost_version must be non-empty")

        if max_entries < 1:
            raise ValueError("max_entries must be positive")

        self.graph_version = graph_version

        self.cost_version = cost_version
        self.forecast_version = forecast_version

        self.versions = (
            graph_version,
            cost_version,
            forecast_version,
        )

        self.max_entries = max_entries

        self._travel_time_provider = (
            travel_time_provider or self._free_flow_travel_time
        )

        # self.builder = DirectedPathBuilder(
        #     edges,
        #     closed_edge_ids=closed_edge_ids,
        #     travel_time_provider=self._travel_time_provider,
        # )

        from src.routing.path_builder import DirectedPathBuilder

        self.builder = DirectedPathBuilder(
            edges,
            closed_edge_ids=closed_edge_ids,
            cost_view=self,
        )

        self._cache: "OrderedDict[tuple, PathResult]" = OrderedDict()

    @property
    def travel_time_provider(self) -> EdgeTravelTimeProvider:
        """Return the provider used by this cost view."""

        return self._travel_time_provider

    # ------------------------------------------------------------------
    # Coherent per-edge / per-path cost decomposition (Step 5)
    # ------------------------------------------------------------------

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
                f"departure_time_s must be finite, got {departure_time_s!r}"
            )

        distance = float(edge.length_m)

        if not isfinite(distance) or distance < 0.0:
            raise InvalidEdgeCostError(
                f"Edge {edge.edge_id!r} has invalid length_m={edge.length_m!r}"
            )

        free_flow = self._free_flow_travel_time(edge, departure_time)

        travel_time = self._validate_travel_time(
            self._travel_time_provider(edge, departure_time),
            edge=edge,
            departure_time_s=departure_time,
        )

        if travel_time < free_flow - 1e-9:
            raise InvalidEdgeCostError(
                f"Travel time for edge {edge.edge_id!r} "
                f"cannot be below free-flow time: "
                f"{travel_time} < {free_flow}"
            )

        congestion_delay = max(0.0, travel_time - free_flow)

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
        """Aggregate coherent costs across a physical directed path.

        Edge departure times are propagated sequentially, so dynamic
        traffic conditions are evaluated at the time the vehicle
        reaches each edge.
        """

        current_time = float(departure_time_s)

        if not isfinite(current_time):
            raise InvalidEdgeCostError(
                f"departure_time_s must be finite, got {departure_time_s!r}"
            )

        distance = 0.0
        free_flow_time = 0.0
        travel_time = 0.0
        congestion_delay = 0.0

        for edge in edges:
            cost = self.edge_cost(edge, departure_time_s=current_time)

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

    # ------------------------------------------------------------------
    # Versioned, bounded shortest-path cache (Step 6)
    # ------------------------------------------------------------------

    def shortest_path(
        self,
        source,
        target,
        *,
        departure_time_s: TimeS = 0.0,
    ) -> PathResult:
        # Exact departure is retained within its one-second bucket: no
        # time approximation.
        key = (
            *self.versions,
            source,
            target,
            int(departure_time_s),
            departure_time_s,
        )

        if key not in self._cache:
            self._cache[key] = self.builder.shortest_path(
                source, target, departure_time_s=departure_time_s
            )

            if len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)

        return self._cache[key]

    def invalidate(
        self,
        *,
        cost_version: str,
        forecast_version: str | None = None,
    ) -> None:
        # Clear all entries: an edge decrease can improve paths not
        # previously using it.
        self.versions = (self.versions[0], cost_version, forecast_version)
        self._cache.clear()

    # ------------------------------------------------------------------
    # Travel-time model
    # ------------------------------------------------------------------

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
                f"Edge {edge.edge_id!r} has invalid length_m={length!r}"
            )

        if not isfinite(speed) or speed <= 0.0:
            raise InvalidEdgeCostError(
                f"Edge {edge.edge_id!r} has invalid speed_limit_mps={speed!r}"
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