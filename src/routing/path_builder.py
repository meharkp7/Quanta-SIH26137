"""Directed and time-dependent road-path construction for Step 4.

This module converts logical node-to-node movements into physical directed
road-edge paths.

The path builder is intentionally independent of optimization. It can be used
by constructive heuristics, ALNS, QPSO, local search, DRL action evaluation,
or closed-loop re-routing.

The default travel-time model is free-flow:

    travel_time = length_m / speed_limit_mps

A dynamic environment can provide a custom travel-time function:

    travel_time_provider(edge, departure_time_s) -> travel_time_s

This allows the same routing layer to operate on static and dynamic traffic
without changing its public architecture.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import inf, isfinite
from typing import Callable, Iterable, Mapping, Sequence

from src.contracts.core_types import (
    RoadEdgeId,
    RoadNodeId,
    TimeS,
)
from src.contracts.scenario import RoadEdge
from src.routing.route_types import RouteLeg


TravelTimeProvider = Callable[[RoadEdge, TimeS], TimeS]


class PathNotFoundError(RuntimeError):
    """Raised when no legal directed path exists between two nodes."""


class InvalidTravelTimeError(ValueError):
    """Raised when a travel-time provider returns an invalid value."""


@dataclass(frozen=True)
class PathResult:
    """Shortest directed path between two road-network nodes."""

    node_ids: tuple[RoadNodeId, ...]
    edge_ids: tuple[RoadEdgeId, ...]
    distance_m: float
    travel_time_s: TimeS


class DirectedRoadGraph:
    """Lightweight directed graph over active RoadEdge objects."""

    def __init__(
        self,
        edges: Iterable[RoadEdge],
        *,
        closed_edge_ids: Iterable[RoadEdgeId] = (),
    ) -> None:
        closed = set(closed_edge_ids)

        self._edges: dict[RoadEdgeId, RoadEdge] = {}
        self._outgoing: dict[RoadNodeId, list[RoadEdge]] = {}

        for edge in edges:
            if edge.edge_id in closed or not edge.open_by_default:
                continue

            if edge.edge_id in self._edges:
                raise ValueError(
                    f"Duplicate road edge ID: {edge.edge_id!r}"
                )

            self._edges[edge.edge_id] = edge
            self._outgoing.setdefault(
                edge.from_node,
                [],
            ).append(edge)

        # Deterministic traversal order.
        for outgoing in self._outgoing.values():
            outgoing.sort(
                key=lambda edge: str(edge.edge_id)
            )

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def outgoing_edges(
        self,
        node_id: RoadNodeId,
    ) -> tuple[RoadEdge, ...]:
        return tuple(
            self._outgoing.get(node_id, ())
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
    """Construct shortest legal directed paths on an active road graph.

    The shortest-path algorithm treats the current arrival time as the label
    of a node. Therefore, when a dynamic travel-time provider is supplied,
    every outgoing edge is evaluated at the time the vehicle reaches its
    tail node.

    This is appropriate for FIFO time-dependent road networks, where leaving
    later cannot result in an earlier arrival on the same edge.
    """

    def __init__(
        self,
        edges: Iterable[RoadEdge],
        *,
        closed_edge_ids: Iterable[RoadEdgeId] = (),
        travel_time_provider: TravelTimeProvider | None = None,
    ) -> None:
        self.graph = DirectedRoadGraph(
            edges,
            closed_edge_ids=closed_edge_ids,
        )

        self._travel_time_provider = (
            travel_time_provider
            or self._free_flow_travel_time
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shortest_path(
        self,
        from_node: RoadNodeId,
        to_node: RoadNodeId,
        *,
        departure_time_s: TimeS = 0.0,
        travel_time_provider: TravelTimeProvider | None = None,
    ) -> PathResult:
        """Find the minimum-arrival-time directed path.

        Parameters
        ----------
        from_node:
            Starting road node.

        to_node:
            Destination road node.

        departure_time_s:
            Absolute simulation/planning time at which traversal starts.

        travel_time_provider:
            Optional provider overriding the builder's default provider for
            this query.

        Notes
        -----
        With a static provider this is ordinary Dijkstra over travel time.

        With a dynamic provider, each edge cost is evaluated at the arrival
        time at that edge's tail node.
        """
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
            )

        provider = (
            travel_time_provider
            or self._travel_time_provider
        )

        arrival_times: dict[RoadNodeId, float] = {
            from_node: float(departure_time_s)
        }

        previous: dict[
            RoadNodeId,
            tuple[RoadNodeId, RoadEdgeId],
        ] = {}

        # Heap entries are:
        #
        #   arrival time
        #   deterministic node string
        #   node
        #
        # The second field makes equal-cost traversal deterministic.
        queue: list[
            tuple[float, str, RoadNodeId]
        ] = [
            (
                float(departure_time_s),
                str(from_node),
                from_node,
            )
        ]

        while queue:
            current_arrival, _, current_node = (
                heapq.heappop(queue)
            )

            known_arrival = arrival_times.get(
                current_node,
                inf,
            )

            if current_arrival > known_arrival:
                continue

            if current_node == to_node:
                return self._reconstruct_path(
                    from_node=from_node,
                    to_node=to_node,
                    departure_time_s=float(
                        departure_time_s
                    ),
                    arrival_time_s=current_arrival,
                    previous=previous,
                    provider=provider,
                )

            for edge in self.graph.outgoing_edges(
                current_node
            ):
                travel_time_s = self._validated_travel_time(
                    provider(
                        edge,
                        current_arrival,
                    ),
                    edge=edge,
                    departure_time_s=current_arrival,
                )

                candidate_arrival = (
                    current_arrival
                    + travel_time_s
                )

                old_arrival = arrival_times.get(
                    edge.to_node,
                    inf,
                )

                if candidate_arrival < old_arrival:
                    arrival_times[edge.to_node] = (
                        candidate_arrival
                    )

                    previous[edge.to_node] = (
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
        travel_time_provider: TravelTimeProvider | None = None,
    ) -> RouteLeg:
        """Build one physical directed route leg."""
        result = self.shortest_path(
            from_node=from_node,
            to_node=to_node,
            departure_time_s=departure_time_s,
            travel_time_provider=travel_time_provider,
        )

        return RouteLeg.from_sequence(
            from_node=from_node,
            to_node=to_node,
            edge_ids=result.edge_ids,
            distance_m=result.distance_m,
            travel_time_s=result.travel_time_s,
        )

    def build_node_sequence(
        self,
        node_ids: Sequence[RoadNodeId],
        *,
        departure_time_s: TimeS = 0.0,
        travel_time_provider: TravelTimeProvider | None = None,
    ) -> tuple[RouteLeg, ...]:
        """Build physical legs for a complete node sequence.

        For dynamic routing, each leg begins when the previous leg ends.
        Therefore traffic conditions can change across a single vehicle route.
        """
        if len(node_ids) < 2:
            return ()

        current_time = float(departure_time_s)
        legs: list[RouteLeg] = []

        for from_node, to_node in zip(
            node_ids,
            node_ids[1:],
        ):
            leg = self.build_leg(
                from_node=from_node,
                to_node=to_node,
                departure_time_s=current_time,
                travel_time_provider=travel_time_provider,
            )

            legs.append(leg)
            current_time += float(
                leg.travel_time_s
            )

        return tuple(legs)

    # ------------------------------------------------------------------
    # Travel-time model
    # ------------------------------------------------------------------

    @staticmethod
    def _free_flow_travel_time(
        edge: RoadEdge,
        departure_time_s: TimeS,
    ) -> TimeS:
        """Calculate free-flow travel time for one road edge."""
        del departure_time_s

        speed = float(edge.speed_limit_mps)

        if speed <= 0.0:
            raise InvalidTravelTimeError(
                f"Road edge {edge.edge_id!r} has non-positive "
                f"speed_limit_mps={speed!r}"
            )

        length = float(edge.length_m)

        if length < 0.0:
            raise InvalidTravelTimeError(
                f"Road edge {edge.edge_id!r} has negative "
                f"length_m={length!r}"
            )

        return length / speed

    @staticmethod
    def _validated_travel_time(
        travel_time_s: TimeS,
        *,
        edge: RoadEdge,
        departure_time_s: TimeS,
    ) -> float:
        """Validate a dynamic travel-time provider result."""
        value = float(travel_time_s)

        if not isfinite(value):
            raise InvalidTravelTimeError(
                f"Travel-time provider returned non-finite value "
                f"{value!r} for edge={edge.edge_id!r} "
                f"at t={departure_time_s}"
            )

        if value < 0.0:
            raise InvalidTravelTimeError(
                f"Travel-time provider returned negative value "
                f"{value!r} for edge={edge.edge_id!r} "
                f"at t={departure_time_s}"
            )

        return value

    # ------------------------------------------------------------------
    # Path reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_path(
        self,
        *,
        from_node: RoadNodeId,
        to_node: RoadNodeId,
        departure_time_s: TimeS,
        arrival_time_s: TimeS,
        previous: Mapping[
            RoadNodeId,
            tuple[RoadNodeId, RoadEdgeId],
        ],
        provider: TravelTimeProvider,
    ) -> PathResult:
        nodes: list[RoadNodeId] = [to_node]
        edges: list[RoadEdgeId] = []

        current = to_node

        while current != from_node:
            if current not in previous:
                raise PathNotFoundError(
                    f"Unable to reconstruct directed path "
                    f"from {from_node!r} to {to_node!r}"
                )

            previous_node, edge_id = previous[
                current
            ]

            edges.append(edge_id)
            nodes.append(previous_node)
            current = previous_node

        nodes.reverse()
        edges.reverse()

        distance_m = sum(
            float(
                self.graph.edge_for(edge_id).length_m
            )
            for edge_id in edges
        )

        # Recompute the edge traversal timeline so the reported travel time
        # exactly corresponds to the selected dynamic path.
        current_time = float(departure_time_s)

        for edge_id in edges:
            edge = self.graph.edge_for(edge_id)

            travel_time_s = self._validated_travel_time(
                provider(
                    edge,
                    current_time,
                ),
                edge=edge,
                departure_time_s=current_time,
            )

            current_time += travel_time_s

        calculated_travel_time = (
            current_time
            - float(departure_time_s)
        )

        # Keep the Dijkstra result authoritative while protecting against
        # numerical noise during reconstruction.
        if abs(
            calculated_travel_time
            - (
                float(arrival_time_s)
                - float(departure_time_s)
            )
        ) > 1e-9:
            raise RuntimeError(
                "Path reconstruction produced an inconsistent "
                "time-dependent arrival time"
            )

        return PathResult(
            node_ids=tuple(nodes),
            edge_ids=tuple(edges),
            distance_m=float(distance_m),
            travel_time_s=float(
                calculated_travel_time
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
                f"{name} must be finite"
            )

        if numeric < 0.0:
            raise ValueError(
                f"{name} must be non-negative"
            )