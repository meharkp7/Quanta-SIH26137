"""Directed and time-dependent road-path construction for Step 5.

The path builder converts logical node-to-node movements into physical
directed road-edge paths.

It is independent of optimization and can therefore be used by constructive
heuristics, ALNS, QPSO, local search, DRL action evaluation, or closed-loop
re-routing.

All physical edge costs are obtained through CostView. This guarantees that
distance, free-flow time, actual travel time, and congestion delay belong to
the same network/cost state.

Step 5B adds a versioned path cache. Cached entries retain physical path
structure; time-dependent costs are reconstructed for the actual departure
time when a cached path is reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
import heapq
from typing import Callable, Iterable, Mapping, Sequence

from src.contracts.scenario import RoadEdge
from src.routing.cost_view import (
    CostView,
    InvalidEdgeCostError,
)
from src.routing.path_cache import (
    CachedPath,
    PathCache,
)
from src.routing.route_types import RouteLeg


TimeS = float
RoadNodeId = str
RoadEdgeId = str


TravelTimeProvider = Callable[
    [RoadEdge, TimeS],
    TimeS,
]


class PathNotFoundError(RuntimeError):
    """Raised when no legal directed path exists between two nodes."""


class InvalidTravelTimeError(ValueError):
    """Raised when a legacy travel-time provider returns an invalid value."""


@dataclass(frozen=True)
class PathResult:
    """Shortest directed path between two road-network nodes."""

    node_ids: tuple[RoadNodeId, ...]
    edge_ids: tuple[RoadEdgeId, ...]
    distance_m: float
    travel_time_s: TimeS
    free_flow_time_s: TimeS
    congestion_delay_s: TimeS


class DirectedRoadGraph:
    """Lightweight directed graph over active RoadEdge objects."""

    def __init__(
        self,
        edges: Iterable[RoadEdge],
        *,
        closed_edge_ids: Iterable[RoadEdgeId] = (),
    ) -> None:
        closed = set(closed_edge_ids)

        self._edges: dict[
            RoadEdgeId,
            RoadEdge,
        ] = {}

        self._outgoing: dict[
            RoadNodeId,
            list[RoadEdge],
        ] = {}

        for edge in edges:
            if (
                edge.edge_id in closed
                or not edge.open_by_default
            ):
                continue

            if edge.edge_id in self._edges:
                raise ValueError(
                    f"Duplicate road edge ID: "
                    f"{edge.edge_id!r}"
                )

            self._edges[edge.edge_id] = edge

            self._outgoing.setdefault(
                edge.from_node,
                [],
            ).append(edge)

        for outgoing in self._outgoing.values():
            outgoing.sort(
                key=lambda edge: str(
                    edge.edge_id
                )
            )

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def outgoing_edges(
        self,
        node_id: RoadNodeId,
    ) -> tuple[RoadEdge, ...]:
        return tuple(
            self._outgoing.get(
                node_id,
                (),
            )
        )

    def edge_for(
        self,
        edge_id: RoadEdgeId,
    ) -> RoadEdge:
        return self._edges[edge_id]

    def contains_node(
        self,
        node_id: RoadNodeId,
    ) -> bool:
        return (
            node_id in self._outgoing
            or any(
                edge.to_node == node_id
                for edge in self._edges.values()
            )
        )


class DirectedPathBuilder:
    """Construct shortest legal directed paths."""

    def __init__(
            self,
            edges: Iterable[RoadEdge],
            *,
            closed_edge_ids: Iterable[RoadEdgeId] = (),
            travel_time_provider: (
                TravelTimeProvider | None
            ) = None,
            cost_view: CostView | None = None,
            path_cache: PathCache | None = None,
            graph_version: str = "default",
        ) -> None:
        if (
            cost_view is not None
            and travel_time_provider is not None
        ):
          raise ValueError(
            "Provide either cost_view or "
            "travel_time_provider, not both"
        )

        self._legacy_travel_time_provider = (
        travel_time_provider is not None
        )

        if cost_view is not None:
           self.cost_view = cost_view
        elif travel_time_provider is not None:
         self.cost_view = CostView(
            edges,
            graph_version=graph_version,
            closed_edge_ids=closed_edge_ids,
            travel_time_provider=travel_time_provider,
        )
        else:
         self.cost_view = CostView(
            edges,
            graph_version=graph_version,
            closed_edge_ids=closed_edge_ids,
        )

        self.graph = DirectedRoadGraph(
            edges,
            closed_edge_ids=closed_edge_ids,
        )

        self.path_cache = path_cache

    @property
    def travel_time_provider(
        self,
    ) -> TravelTimeProvider | None:
        """Compatibility access to the CostView provider."""

        return self.cost_view.travel_time_provider

    def invalidate_cache(
        self,
        *,
        graph_version: str | None = None,
        cost_version: str | None = None,
        forecast_version: str | None = None,
    ) -> int:
        """Invalidate matching cached routing entries."""

        if self.path_cache is None:
            return 0

        return self.path_cache.invalidate(
            graph_version=graph_version,
            cost_version=cost_version,
            forecast_version=forecast_version,
        )

    def shortest_path(
        self,
        from_node: RoadNodeId,
        to_node: RoadNodeId,
        *,
        departure_time_s: TimeS = 0.0,
        travel_time_provider: (
            TravelTimeProvider | None
        ) = None,
    ) -> PathResult:
        """Find the minimum-arrival-time directed path."""

        self._validate_time(
            departure_time_s,
            name="departure_time_s",
        )

        if from_node == to_node:
            return PathResult(
                node_ids=(from_node,),
                edge_ids=(),
                distance_m=0.0,
                travel_time_s=0.0,
                free_flow_time_s=0.0,
                congestion_delay_s=0.0,
            )

        if travel_time_provider is not None:
            if (
                travel_time_provider
                is not self.cost_view.travel_time_provider
            ):
                cost_view = CostView(
                    graph_version=(
                        self.cost_view.graph_version
                    ),
                    cost_version=(
                        self.cost_view.cost_version
                    ),
                    forecast_version=(
                        self.cost_view.forecast_version
                    ),
                    travel_time_provider=(
                        travel_time_provider
                    ),
                )
                legacy_provider_for_call = True
            else:
                cost_view = self.cost_view
                legacy_provider_for_call = (
                    self._legacy_travel_time_provider
                )
        else:
            cost_view = self.cost_view
            legacy_provider_for_call = (
                self._legacy_travel_time_provider
            )

        # A per-call legacy provider is not safe to reuse through the shared
        # cache because the provider itself is part of the cost state.
        cache = (
            self.path_cache
            if (
                self.path_cache is not None
                and not (
                    travel_time_provider is not None
                    and travel_time_provider
                    is not self.cost_view.travel_time_provider
                )
            )
            else None
        )

        cache_key = None

        if cache is not None:
            cache_key = cache.make_key(
                from_node=from_node,
                to_node=to_node,
                graph_version=(
                    cost_view.graph_version
                ),
                cost_version=(
                    cost_view.cost_version
                ),
                forecast_version=(
                    cost_view.forecast_version
                ),
                departure_time_s=(
                    departure_time_s
                ),
            )

            cached = cache.get(cache_key)

            if cached is not None:
                return self._result_from_cached_path(
                    cached,
                    departure_time_s=(
                        departure_time_s
                    ),
                    cost_view=cost_view,
                )

        arrival_times: dict[
            RoadNodeId,
            float,
        ] = {
            from_node: float(
                departure_time_s
            )
        }

        previous: dict[
            RoadNodeId,
            tuple[
                RoadNodeId,
                RoadEdgeId,
            ],
        ] = {}

        queue: list[
            tuple[
                float,
                str,
                RoadNodeId,
            ]
        ] = [
            (
                float(departure_time_s),
                str(from_node),
                from_node,
            )
        ]

        while queue:
            (
                current_arrival,
                _,
                current_node,
            ) = heapq.heappop(queue)

            known_arrival = arrival_times.get(
                current_node,
                inf,
            )

            if current_arrival > known_arrival:
                continue

            if current_node == to_node:
                result = self._reconstruct_path(
                    from_node=from_node,
                    to_node=to_node,
                    departure_time_s=float(
                        departure_time_s
                    ),
                    arrival_time_s=current_arrival,
                    previous=previous,
                    cost_view=cost_view,
                )

                if (
                    cache is not None
                    and cache_key is not None
                ):
                    cache.put_result(
                        cache_key,
                        result,
                    )

                return result

            for edge in self.graph.outgoing_edges(
                current_node
            ):
                try:
                    edge_cost = cost_view.edge_cost(
                        edge,
                        departure_time_s=(
                            current_arrival
                        ),
                    )
                except InvalidEdgeCostError as exc:
                    if legacy_provider_for_call:
                        raise InvalidTravelTimeError(
                            str(exc)
                        ) from exc
                    raise

                candidate_arrival = (
                    current_arrival
                    + edge_cost.travel_time_s
                )

                old_arrival = arrival_times.get(
                    edge.to_node,
                    inf,
                )

                if candidate_arrival < old_arrival:
                    arrival_times[
                        edge.to_node
                    ] = candidate_arrival

                    previous[
                        edge.to_node
                    ] = (
                        current_node,
                        edge.edge_id,
                    )

                    heapq.heappush(
                        queue,
                        (
                            candidate_arrival,
                            str(edge.to_node),
                            edge.to_node,
                        ),
                    )

        raise PathNotFoundError(
            f"No directed path exists from "
            f"{from_node!r} to {to_node!r}"
        )

    def build_leg(
        self,
        from_node: RoadNodeId,
        to_node: RoadNodeId,
        *,
        departure_time_s: TimeS = 0.0,
        travel_time_provider: (
            TravelTimeProvider | None
        ) = None,
    ) -> RouteLeg:
        """Build one physical directed route leg."""

        result = self.shortest_path(
            from_node=from_node,
            to_node=to_node,
            departure_time_s=departure_time_s,
            travel_time_provider=(
                travel_time_provider
            ),
        )

        return RouteLeg.from_sequence(
            from_node=from_node,
            to_node=to_node,
            edge_ids=result.edge_ids,
            distance_m=result.distance_m,
            travel_time_s=result.travel_time_s,
            free_flow_time_s=(
                result.free_flow_time_s
            ),
            congestion_delay_s=(
                result.congestion_delay_s
            ),
        )

    def build_node_sequence(
        self,
        node_ids: Sequence[RoadNodeId],
        *,
        departure_time_s: TimeS = 0.0,
        travel_time_provider: (
            TravelTimeProvider | None
        ) = None,
    ) -> tuple[RouteLeg, ...]:
        """Build physical legs for a complete node sequence."""

        if len(node_ids) < 2:
            return ()

        self._validate_time(
            departure_time_s,
            name="departure_time_s",
        )

        current_time = float(
            departure_time_s
        )

        legs: list[RouteLeg] = []

        for from_node, to_node in zip(
            node_ids,
            node_ids[1:],
        ):
            leg = self.build_leg(
                from_node=from_node,
                to_node=to_node,
                departure_time_s=current_time,
                travel_time_provider=(
                    travel_time_provider
                ),
            )

            legs.append(leg)

            current_time += float(
                leg.travel_time_s
            )

        return tuple(legs)

    def _result_from_cached_path(
        self,
        cached: CachedPath,
        *,
        departure_time_s: TimeS,
        cost_view: CostView,
    ) -> PathResult:
        """Re-evaluate cached physical path at actual departure time."""

        path_edges = tuple(
            self.graph.edge_for(edge_id)
            for edge_id in cached.edge_ids
        )

        path_cost = cost_view.path_cost(
            path_edges,
            departure_time_s=departure_time_s,
        )

        return PathResult(
            node_ids=cached.node_ids,
            edge_ids=cached.edge_ids,
            distance_m=float(
                path_cost.distance_m
            ),
            travel_time_s=float(
                path_cost.travel_time_s
            ),
            free_flow_time_s=float(
                path_cost.free_flow_time_s
            ),
            congestion_delay_s=float(
                path_cost.congestion_delay_s
            ),
        )

    def _reconstruct_path(
        self,
        *,
        from_node: RoadNodeId,
        to_node: RoadNodeId,
        departure_time_s: TimeS,
        arrival_time_s: TimeS,
        previous: Mapping[
            RoadNodeId,
            tuple[
                RoadNodeId,
                RoadEdgeId,
            ],
        ],
        cost_view: CostView,
    ) -> PathResult:
        nodes: list[RoadNodeId] = [to_node]
        edges: list[RoadEdgeId] = []

        current = to_node

        while current != from_node:
            if current not in previous:
                raise PathNotFoundError(
                    f"Unable to reconstruct directed "
                    f"path from {from_node!r} to "
                    f"{to_node!r}"
                )

            previous_node, edge_id = (
                previous[current]
            )

            edges.append(edge_id)
            nodes.append(previous_node)
            current = previous_node

        nodes.reverse()
        edges.reverse()

        path_edges = tuple(
            self.graph.edge_for(edge_id)
            for edge_id in edges
        )

        path_cost = cost_view.path_cost(
            path_edges,
            departure_time_s=departure_time_s,
        )

        calculated_travel_time = (
            path_cost.travel_time_s
        )

        if abs(
            calculated_travel_time
            - (
                float(arrival_time_s)
                - float(departure_time_s)
            )
        ) > 1e-9:
            raise RuntimeError(
                "Path reconstruction produced an "
                "inconsistent time-dependent "
                "arrival time"
            )

        return PathResult(
            node_ids=tuple(nodes),
            edge_ids=tuple(edges),
            distance_m=float(
                path_cost.distance_m
            ),
            travel_time_s=float(
                path_cost.travel_time_s
            ),
            free_flow_time_s=float(
                path_cost.free_flow_time_s
            ),
            congestion_delay_s=float(
                path_cost.congestion_delay_s
            ),
        )

    @staticmethod
    def _validate_time(
        value: TimeS,
        *,
        name: str,
    ) -> None:
        numeric = float(value)

        if not isfinite(numeric):
            raise ValueError(
                f"{name} must be finite, "
                f"got {value!r}"
            )

        if numeric < 0.0:
            raise ValueError(
                f"{name} must be non-negative, "
                f"got {value!r}"
            )