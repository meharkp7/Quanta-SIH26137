"""Step 18: quantum-inspired shortest-path search and exact reference.

This module is deliberately independent from the CVRPTW route evaluator.  It
solves one snapshot source/destination path problem on a directed road graph.
The QPSO representation is a random-key vector over the graph's node IDs;
decoding is a bounded, loop-free backtracking search that only traverses legal
outgoing edges.  Dijkstra uses the exact same immutable edge-weight function.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, log
import heapq
import random
import time
from typing import Callable, Iterable, Mapping, Sequence

from src.contracts.scenario import RoadEdge

NodeId = str
EdgeId = str
WeightFn = Callable[[RoadEdge], float]


class ShortestPathError(ValueError):
    """Invalid shortest-path configuration or input."""


class ShortestPathNotFound(RuntimeError):
    """No legal source-to-destination path exists."""


@dataclass(frozen=True)
class ShortestPathResult:
    method: str
    source: NodeId
    destination: NodeId
    node_ids: tuple[NodeId, ...]
    edge_ids: tuple[EdgeId, ...]
    cost: float
    valid: bool
    runtime_s: float
    evaluations: int = 0
    backtracks: int = 0
    failure_reason: str | None = None


@dataclass(frozen=True)
class ShortestPathComparison:
    qpso: ShortestPathResult
    dijkstra: ShortestPathResult
    absolute_gap: float | None
    relative_gap: float | None


@dataclass(frozen=True)
class ShortestPathQPSOConfig:
    particles: int = 24
    iterations: int = 80
    alpha_max: float = 0.90
    alpha_min: float = 0.45
    max_backtracks: int = 64
    seed: int = 7
    time_limit_s: float | None = 2.0

    def __post_init__(self) -> None:
        if self.particles < 1 or self.iterations < 1:
            raise ShortestPathError("particles and iterations must be positive")
        if not (0.0 < self.alpha_min <= self.alpha_max < 1.5):
            raise ShortestPathError("invalid alpha bounds")
        if self.max_backtracks < 0:
            raise ShortestPathError("max_backtracks must be non-negative")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ShortestPathError("time_limit_s must be positive or None")


class DirectedWeightedGraph:
    """Deterministic directed graph with a fixed non-negative edge weight."""

    def __init__(self, edges: Iterable[RoadEdge], weight_fn: WeightFn) -> None:
        self.edges = tuple(edges)
        self._weight: dict[EdgeId, float] = {}
        self._outgoing: dict[NodeId, tuple[RoadEdge, ...]] = {}
        seen: set[EdgeId] = set()
        buckets: dict[NodeId, list[RoadEdge]] = {}
        for edge in self.edges:
            if edge.edge_id in seen:
                raise ShortestPathError(f"duplicate edge id: {edge.edge_id}")
            seen.add(edge.edge_id)
            w = float(weight_fn(edge))
            if not isfinite(w) or w < 0.0:
                raise ShortestPathError("all edge weights must be finite and non-negative")
            self._weight[edge.edge_id] = w
            buckets.setdefault(edge.from_node, []).append(edge)
        self._outgoing = {
            node: tuple(sorted(es, key=lambda e: e.edge_id))
            for node, es in buckets.items()
        }

    def outgoing(self, node: NodeId) -> tuple[RoadEdge, ...]:
        return self._outgoing.get(node, ())

    def weight(self, edge_id: EdgeId) -> float:
        return self._weight[edge_id]

    def nodes(self) -> tuple[NodeId, ...]:
        nodes = {e.from_node for e in self.edges} | {e.to_node for e in self.edges}
        return tuple(sorted(nodes))

    def path_cost(self, edge_ids: Sequence[EdgeId]) -> float:
        return sum(self._weight[eid] for eid in edge_ids)


def _validate_path(graph: DirectedWeightedGraph, source: NodeId, destination: NodeId,
                   node_ids: Sequence[NodeId], edge_ids: Sequence[EdgeId]) -> bool:
    if not node_ids or node_ids[0] != source or node_ids[-1] != destination:
        return False
    if len(edge_ids) != len(node_ids) - 1 or len(set(node_ids)) != len(node_ids):
        return False
    for i, eid in enumerate(edge_ids):
        edge = next((e for e in graph.outgoing(node_ids[i]) if e.edge_id == eid), None)
        if edge is None or edge.to_node != node_ids[i + 1]:
            return False
    return True


def dijkstra_shortest_path(graph: DirectedWeightedGraph, source: NodeId,
                           destination: NodeId) -> ShortestPathResult:
    start = time.perf_counter()
    if source == destination:
        return ShortestPathResult("dijkstra", source, destination, (source,), (), 0.0, True,
                                  time.perf_counter() - start)
    dist: dict[NodeId, float] = {source: 0.0}
    prev: dict[NodeId, tuple[NodeId, EdgeId]] = {}
    queue: list[tuple[float, str, NodeId]] = [(0.0, source, source)]
    while queue:
        cost, _, node = heapq.heappop(queue)
        if cost != dist.get(node, inf):
            continue
        if node == destination:
            break
        for edge in graph.outgoing(node):
            new = cost + graph.weight(edge.edge_id)
            if new < dist.get(edge.to_node, inf):
                dist[edge.to_node] = new
                prev[edge.to_node] = (node, edge.edge_id)
                heapq.heappush(queue, (new, edge.to_node, edge.to_node))
    if destination not in dist:
        return ShortestPathResult("dijkstra", source, destination, (), (), inf, False,
                                  time.perf_counter() - start, failure_reason="dead_end")
    nodes = [destination]
    edges: list[EdgeId] = []
    cur = destination
    while cur != source:
        parent, eid = prev[cur]
        edges.append(eid)
        nodes.append(parent)
        cur = parent
    nodes.reverse(); edges.reverse()
    return ShortestPathResult("dijkstra", source, destination, tuple(nodes), tuple(edges),
                              dist[destination], True, time.perf_counter() - start)


def _decode_random_keys(graph: DirectedWeightedGraph, source: NodeId, destination: NodeId,
                        keys: Mapping[NodeId, float], max_backtracks: int) -> tuple[tuple[NodeId, ...], tuple[EdgeId, ...], int]:
    if source == destination:
        return (source,), (), 0
    backtracks = 0

    def search(node: NodeId, visited: set[NodeId], nodes: list[NodeId], edges: list[EdgeId]):
        nonlocal backtracks
        if node == destination:
            return tuple(nodes), tuple(edges)
        candidates = [e for e in graph.outgoing(node) if e.to_node not in visited]
        candidates.sort(key=lambda e: (keys.get(e.to_node, 0.5), e.edge_id))
        for edge in candidates:
            if backtracks > max_backtracks:
                return None
            visited.add(edge.to_node); nodes.append(edge.to_node); edges.append(edge.edge_id)
            found = search(edge.to_node, visited, nodes, edges)
            if found is not None:
                return found
            visited.remove(edge.to_node); nodes.pop(); edges.pop()
            backtracks += 1
        return None

    found = search(source, {source}, [source], [])
    if found is None:
        return (), (), backtracks
    return found[0], found[1], backtracks


class ShortestPathQPSO:
    """Random-key QPSO for a single directed snapshot shortest-path query."""

    def __init__(self, graph: DirectedWeightedGraph, config: ShortestPathQPSOConfig | None = None) -> None:
        self.graph = graph
        self.config = config or ShortestPathQPSOConfig()
        self._rng = random.Random(self.config.seed)
        self._nodes = graph.nodes()

    def solve(self, source: NodeId, destination: NodeId) -> ShortestPathResult:
        start = time.perf_counter()
        if source not in self._nodes or destination not in self._nodes:
            return ShortestPathResult("qpso", source, destination, (), (), inf, False,
                                      time.perf_counter() - start, failure_reason="unknown_node")
        if source == destination:
            return ShortestPathResult("qpso", source, destination, (source,), (), 0.0, True,
                                      time.perf_counter() - start)
        dimension_nodes = tuple(n for n in self._nodes if n not in (source, destination))
        dim = len(dimension_nodes)
        if dim == 0:
            return ShortestPathResult("qpso", source, destination, (), (), inf, False,
                                      time.perf_counter() - start, failure_reason="dead_end")

        positions = [[self._rng.random() for _ in range(dim)] for _ in range(self.config.particles)]
        pbest = [p[:] for p in positions]
        pbest_score = [inf] * self.config.particles
        best_pos: list[float] | None = None
        best_score = inf
        best_path: tuple[NodeId, ...] = ()
        best_edges: tuple[EdgeId, ...] = ()
        evaluations = 0
        total_backtracks = 0

        def evaluate(pos: Sequence[float]) -> tuple[float, tuple[NodeId, ...], tuple[EdgeId, ...], int]:
            nonlocal evaluations, total_backtracks
            keys = {node: pos[i] for i, node in enumerate(dimension_nodes)}
            nodes, edges, backs = _decode_random_keys(self.graph, source, destination, keys,
                                                       self.config.max_backtracks)
            evaluations += 1; total_backtracks += backs
            if not nodes:
                return inf, (), (), backs
            return self.graph.path_cost(edges), nodes, edges, backs

        for i, pos in enumerate(positions):
            score, nodes, edges, _ = evaluate(pos)
            pbest_score[i] = score
            if score < best_score:
                best_score, best_pos = score, pos[:]
                best_path, best_edges = nodes, edges

        if best_pos is None:
            return ShortestPathResult("qpso", source, destination, (), (), inf, False,
                                      time.perf_counter() - start, evaluations, total_backtracks,
                                      "dead_end")

        for iteration in range(self.config.iterations):
            if self.config.time_limit_s is not None and time.perf_counter() - start >= self.config.time_limit_s:
                break
            mbest = [sum(p[d] for p in pbest) / len(pbest) for d in range(dim)]
            progress = iteration / max(1, self.config.iterations - 1)
            alpha = self.config.alpha_max - (self.config.alpha_max - self.config.alpha_min) * progress
            for i, pos in enumerate(positions):
                for d in range(dim):
                    phi = self._rng.random()
                    attractor = phi * pbest[i][d] + (1.0 - phi) * best_pos[d]
                    u = max(self._rng.random(), 1e-12)
                    sign = -1.0 if self._rng.random() < 0.5 else 1.0
                    pos[d] = max(0.0, min(1.0, attractor + sign * alpha * abs(mbest[d] - pos[d]) * log(1.0 / u)))
                score, nodes, edges, _ = evaluate(pos)
                if score < pbest_score[i]:
                    pbest[i] = pos[:]; pbest_score[i] = score
                if score < best_score:
                    best_score, best_pos = score, pos[:]
                    best_path, best_edges = nodes, edges

        runtime = time.perf_counter() - start
        valid = bool(best_path) and _validate_path(self.graph, source, destination, best_path, best_edges)
        return ShortestPathResult("qpso", source, destination, best_path, best_edges,
                                  best_score, valid, runtime, evaluations, total_backtracks,
                                  None if valid else "dead_end")


def compare_shortest_paths(graph: DirectedWeightedGraph, source: NodeId, destination: NodeId,
                            config: ShortestPathQPSOConfig | None = None) -> ShortestPathComparison:
    qpso = ShortestPathQPSO(graph, config).solve(source, destination)
    dijkstra = dijkstra_shortest_path(graph, source, destination)
    if qpso.valid and dijkstra.valid:
        gap = qpso.cost - dijkstra.cost
        rel = gap / dijkstra.cost if dijkstra.cost > 0 else (0.0 if gap == 0 else inf)
    else:
        gap = rel = None
    return ShortestPathComparison(qpso, dijkstra, gap, rel)
