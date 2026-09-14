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

        # Free-flow travel time is a pure function of an edge's static
        # length/speed-limit and does not depend on departure_time_s (see
        # `_free_flow_travel_time`, which discards it). Profiling showed
        # it being recomputed — with full isfinite validation — 8M+ times
        # in a single 40-customer fitness evaluation because Dijkstra
        # revisits the same edges from many different search states. It
        # is safe to memoize per edge_id for the lifetime of this
        # CostView, since a new CostView is constructed whenever
        # graph_version changes (edge topology/geometry is immutable
        # within one version).
        self._free_flow_cache: dict[str, float] = {}

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

    def _edge_cost_components(
        self,
        edge: RoadEdge,
        departure_time: float,
    ) -> tuple[float, float, float, float]:
        """Validated (distance, free_flow, travel_time, congestion) tuple.

        Shared implementation behind both `edge_cost()` and `path_cost()`.
        `edge_cost()` wraps this in an `EdgeCost` dataclass for callers
        that want a coherent, named object; `path_cost()` sums these
        scalars directly across a path's edges without allocating (and
        re-validating via `EdgeCost.__post_init__`) an object per edge.
        Profiling a real QPSO run showed `path_cost()` — invoked once per
        `shortest_path()` call, including cache *hits*, to re-price a
        cached physical path at the actual departure time — was
        responsible for ~10.6M redundant `EdgeCost` constructions.
        """

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

        return distance, free_flow, travel_time, congestion_delay

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

        distance, free_flow, travel_time, congestion_delay = (
            self._edge_cost_components(edge, departure_time)
        )

        return EdgeCost(
            distance_m=distance,
            free_flow_time_s=free_flow,
            travel_time_s=travel_time,
            congestion_delay_s=congestion_delay,
        )

    def edge_travel_time(
        self,
        edge: RoadEdge,
        *,
        departure_time_s: TimeS = 0.0,
    ) -> float:
        """Return only the validated travel time for one edge.

        This is the hot-path counterpart to ``edge_cost()``. Dijkstra-style
        shortest-path search (``DirectedPathBuilder.shortest_path``) calls
        this once per edge relaxation and only ever uses
        ``EdgeCost.travel_time_s`` from the result; profiling a single
        40-customer fitness evaluation showed ~4.1M ``edge_cost()`` calls
        consuming ~89% of total evaluation time, almost entirely in
        validation and ``EdgeCost`` allocation whose other fields
        (distance_m, free_flow_time_s, congestion_delay_s) were discarded.

        This method performs exactly the same *correctness* checks as
        ``edge_cost()`` for the parts that matter to path search (finite
        non-negative length/speed/travel-time, travel time never below
        free-flow), but skips constructing/re-validating an ``EdgeCost``
        object and skips computing the congestion decomposition, which
        relaxation never uses. The full, coherent cost breakdown
        (distance + time + congestion together) remains available via
        ``edge_cost()`` / ``path_cost()`` and is what
        ``_reconstruct_path`` uses to report the winning path — so the
        reported route costs are unaffected by this fast path.
        """

        departure_time = float(departure_time_s)

        if not isfinite(departure_time):
            raise InvalidEdgeCostError(
                f"departure_time_s must be finite, got {departure_time_s!r}"
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

        return travel_time

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
            (
                edge_distance,
                edge_free_flow,
                edge_travel_time_s,
                edge_congestion,
            ) = self._edge_cost_components(edge, current_time)

            distance += edge_distance
            free_flow_time += edge_free_flow
            travel_time += edge_travel_time_s
            congestion_delay += edge_congestion

            current_time += edge_travel_time_s

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

    def _free_flow_travel_time(
        self,
        edge: RoadEdge,
        departure_time_s: TimeS,
    ) -> TimeS:
        del departure_time_s

        cached = self._free_flow_cache.get(edge.edge_id)

        if cached is not None:
            return cached

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

        result = length / speed

        self._free_flow_cache[edge.edge_id] = result

        return result

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