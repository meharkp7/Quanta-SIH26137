"""Synthetic road-network construction for Step 4.

The implementation follows the plan's two-pass design:
1. a small Delaunay debugging graph;
2. a larger grid/jittered-grid or irregular synthetic street skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

from src.contracts.core_types import NodeKind, RoadClass

from .vrp_parser import VrpInstance


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class RoadSegment:
    a: int
    b: int
    parent_road_id: str


@dataclass(frozen=True)
class RoadBuildConfig:
    """Static road-generation settings from the Step 4 plan."""

    junction_count: int = 196
    topology_family: str = "grid"
    grid_rows: int = 14
    grid_cols: int = 14
    jitter_fraction: float = 0.18
    protected_cycle_fraction: float = 0.18
    prune_fraction: float = 0.22
    one_way_fraction: float = 0.15
    max_customer_snap_m: float = 500.0
    min_segment_length_m: float = 30.0
    arterial_count: int = 6
    signal_fraction: float = 0.08
    lane_saturation_per_lane: float = 900.0
    seed: int = 26137

    def __post_init__(self) -> None:
        if self.topology_family not in {"grid", "irregular"}:
            raise ValueError("topology_family must be 'grid' or 'irregular'")
        if self.junction_count < 5:
            raise ValueError("junction_count must be at least 5")
        if not 0 <= self.one_way_fraction <= 1:
            raise ValueError("one_way_fraction must be between 0 and 1")
        if not 0 <= self.prune_fraction < 1:
            raise ValueError("prune_fraction must be in [0, 1)")
        if self.max_customer_snap_m <= 0:
            raise ValueError("max_customer_snap_m must be positive")
        if self.min_segment_length_m <= 0:
            raise ValueError("min_segment_length_m must be positive")


def scale_customer_points(
    instance: VrpInstance, extent_m: float
) -> tuple[dict[str, Point], dict[str, float | str]]:
    """Apply one isotropic transform to the complete benchmark layout."""

    if extent_m <= 0:
        raise ValueError("extent_m must be positive")

    xs = [record.x for record in instance.customers]
    ys = [record.y for record in instance.customers]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min)
    if span <= 0:
        raise ValueError("CVRP coordinates must span a positive area")

    scale = extent_m / span
    points = {
        record.customer_id: Point(
            (record.x - x_min) * scale,
            (record.y - y_min) * scale,
        )
        for record in instance.customers
    }
    transform = {
        "scale": scale,
        "translation_x_m": x_min,
        "translation_y_m": y_min,
    }
    return points, transform


def build_debug_delaunay(
    points: dict[str, Point], seed: int
) -> tuple[list[dict], list[dict]]:
    """Build the small Pass-1 Delaunay fixture without external dependencies."""

    ordered = list(points.items())
    unique: list[Point] = []
    for _, point in ordered:
        if point not in unique:
            unique.append(point)
    if len(unique) < 3 or _all_collinear(unique):
        segments = _fallback_connected_segments(unique)
    else:
        triangles = _bowyer_watson(unique)
        edge_pairs = {
            tuple(sorted(pair))
            for triangle in triangles
            for pair in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            )
        }
        segments = [
            RoadSegment(a, b, f"D{index:05d}")
            for index, (a, b) in enumerate(sorted(edge_pairs))
        ]

    nodes = [
        {
            "node_id": f"D{index:04d}",
            "x_m": point.x,
            "y_m": point.y,
            "kind": NodeKind.JUNCTION,
            "zone_id": f"Z{index // 16:03d}",
            "signalized": False,
        }
        for index, point in enumerate(unique)
    ]
    edges = _segments_to_edges(nodes, segments, seed)
    return nodes, edges


def build_synthetic_roads(
    customer_points: dict[str, Point],
    config: RoadBuildConfig,
) -> tuple[list[dict], list[dict]]:
    """Build the larger Pass-2 synthetic road skeleton."""

    rng = random.Random(config.seed)
    points = _synthetic_junction_points(customer_points, config, rng)
    segments = _candidate_segments(points, config)
    segments = _connected_prune(segments, points, config, rng)
    segments, arterial_parents = _add_arterial_corridors(segments, points, config, rng)

    nodes = [
        {
            "node_id": f"N{index:04d}",
            "x_m": point.x,
            "y_m": point.y,
            "kind": NodeKind.JUNCTION,
            "zone_id": _zone(point, points),
            "signalized": False,
        }
        for index, point in enumerate(points)
    ]
    edges = _segments_to_edges(nodes, segments, config.seed, arterial_parents)
    _assign_signals(nodes, edges, config.signal_fraction, config.seed)
    _apply_safe_one_way(nodes, edges, config.one_way_fraction, config.seed)
    return nodes, edges


def attach_customers(
    nodes: list[dict],
    edges: list[dict],
    customer_points: dict[str, Point],
    max_snap_m: float,
    min_segment_length_m: float = 30.0,
) -> dict[str, tuple[str, float]]:
    """Project customers to roads and split all directions of the parent road."""

    if not edges:
        raise ValueError("Cannot attach customers to an empty road graph")

    access: dict[str, tuple[str, float]] = {}
    next_node = _next_node_number(nodes)
    next_edge = _next_edge_number(edges)

    for customer_id, point in customer_points.items():
        best = None
        for edge in list(edges):
            start = _node_point(nodes, edge["from_node"])
            end = _node_point(nodes, edge["to_node"])
            projection, distance, fraction = _project(point, start, end)
            candidate = (distance, edge, projection, fraction)
            if best is None or candidate[0] < best[0]:
                best = candidate

        assert best is not None
        snap_distance, edge, projection, fraction = best
        if snap_distance > max_snap_m:
            raise ValueError(
                f"Customer {customer_id!r} snap distance {snap_distance:.2f} m "
                f"exceeds configured maximum {max_snap_m:.2f} m"
            )

        start_id = edge["from_node"]
        end_id = edge["to_node"]
        edge_length = edge["length_m"]
        # Avoid pathological micro-edges when a customer projects extremely
        # close to an existing junction. Attach to the nearer junction instead.
        if fraction <= 1e-9 or edge_length * fraction < min_segment_length_m:
            access_node = start_id
            snap_distance = _distance(point, _node_point(nodes, start_id))
        elif fraction >= 1 - 1e-9 or edge_length * (1 - fraction) < min_segment_length_m:
            access_node = end_id
            snap_distance = _distance(point, _node_point(nodes, end_id))
        else:
            access_node = f"N{next_node:04d}"
            next_node += 1
            all_points = [Point(n["x_m"], n["y_m"]) for n in nodes]
            nodes.append(
                {
                    "node_id": access_node,
                    "x_m": projection.x,
                    "y_m": projection.y,
                    "kind": NodeKind.CUSTOMER_ACCESS,
                    "zone_id": _zone(projection, all_points),
                    "signalized": False,
                }
            )

            # Split the selected directed edge and its exact physical
            # counterpart. Using only source_edge_id is fragile after a road
            # has already been split by another customer; parent road ID plus
            # reversed endpoints keeps both directions geometrically paired.
            counterparts = [edge]
            reverse_edge = next(
                (candidate for candidate in edges
                 if candidate is not edge
                 and candidate["parent_road_id"] == edge["parent_road_id"]
                 and candidate["from_node"] == end_id
                 and candidate["to_node"] == start_id),
                None,
            )
            if reverse_edge is not None:
                counterparts.append(reverse_edge)

            for candidate in counterparts:
                candidate_start = _node_point(nodes, candidate["from_node"])
                candidate_end = _node_point(nodes, candidate["to_node"])
                _, _, candidate_fraction = _project(point, candidate_start, candidate_end)
                first_length = candidate["length_m"] * candidate_fraction
                second_length = candidate["length_m"] * (1.0 - candidate_fraction)

                if (candidate_fraction <= 1e-9 or candidate_fraction >= 1.0 - 1e-9
                        or first_length < min_segment_length_m
                        or second_length < min_segment_length_m):
                    continue

                edges.remove(candidate)
                first, second = _split_edge(candidate, access_node, candidate_fraction, next_edge)
                next_edge += 2
                edges.extend((first, second))

        access[customer_id] = (access_node, snap_distance)

    return access

def _synthetic_junction_points(
    customer_points: dict[str, Point], config: RoadBuildConfig, rng: random.Random
) -> list[Point]:
    max_x = max(point.x for point in customer_points.values())
    max_y = max(point.y for point in customer_points.values())
    count = config.junction_count

    if config.topology_family == "grid":
        rows, cols = _grid_shape(count, config.grid_rows, config.grid_cols)
        points: list[Point] = []
        for row in range(rows):
            for col in range(cols):
                x = max_x * col / max(cols - 1, 1)
                y = max_y * row / max(rows - 1, 1)
                jitter_x = (max_x / max(cols - 1, 1)) * config.jitter_fraction
                jitter_y = (max_y / max(rows - 1, 1)) * config.jitter_fraction
                if 0 < col < cols - 1:
                    x += rng.uniform(-jitter_x, jitter_x)
                if 0 < row < rows - 1:
                    y += rng.uniform(-jitter_y, jitter_y)
                points.append(Point(max(0, min(max_x, x)), max(0, min(max_y, y))))
        return points[:count]

    return [Point(rng.uniform(0, max_x), rng.uniform(0, max_y)) for _ in range(count)]


def _grid_shape(count: int, rows: int, cols: int) -> tuple[int, int]:
    if rows * cols == count:
        return rows, cols
    cols = max(3, round(math.sqrt(count)))
    rows = max(3, math.ceil(count / cols))
    return rows, cols


def _candidate_segments(points: list[Point], config: RoadBuildConfig) -> list[RoadSegment]:
    if config.topology_family == "grid":
        rows, cols = _grid_shape(len(points), config.grid_rows, config.grid_cols)
        segments: list[RoadSegment] = []
        for row in range(rows):
            for col in range(cols):
                index = row * cols + col
                if index >= len(points):
                    continue
                if col + 1 < cols and index + 1 < len(points):
                    segments.append(RoadSegment(index, index + 1, f"R{index:05d}H"))
                if row + 1 < rows and index + cols < len(points):
                    segments.append(RoadSegment(index, index + cols, f"R{index:05d}V"))
        return segments

    return [
        RoadSegment(a, b, f"R{index:05d}")
        for index, (a, b) in enumerate(_nearest_neighbor_edges(points, 4))
    ]


def _connected_prune(
    segments: list[RoadSegment], points: list[Point], config: RoadBuildConfig, rng: random.Random
) -> list[RoadSegment]:
    """Keep an MST backbone and add cycles before pruning redundant candidates."""

    unique = {(min(s.a, s.b), max(s.a, s.b)): s for s in segments}
    candidates = list(unique.values())
    backbone = _mst(candidates, points)
    backbone_keys = {(min(s.a, s.b), max(s.a, s.b)) for s in backbone}
    extras = [s for s in candidates if (min(s.a, s.b), max(s.a, s.b)) not in backbone_keys]
    extras.sort(key=lambda s: _stable_score(config.seed, s.parent_road_id))

    cycle_count = max(1, int(len(backbone) * config.protected_cycle_fraction))
    keep = backbone + extras[:cycle_count]
    remaining = extras[cycle_count:]
    keep_target = max(len(backbone) + cycle_count, int(len(candidates) * (1 - config.prune_fraction)))
    keep.extend(remaining[: max(0, keep_target - len(keep))])
    return keep


def _add_arterial_corridors(
    segments: list[RoadSegment], points: list[Point], config: RoadBuildConfig, rng: random.Random
) -> tuple[list[RoadSegment], set[str]]:
    """Mark connected shortest-path corridors as arterial roads."""

    if not segments:
        return segments, set()

    adjacency: dict[int, list[int]] = {i: [] for i in range(len(points))}
    for segment in segments:
        adjacency[segment.a].append(segment.b)
        adjacency[segment.b].append(segment.a)

    min_x, max_x = min(p.x for p in points), max(p.x for p in points)
    min_y, max_y = min(p.y for p in points), max(p.y for p in points)
    boundary = [
        i for i, p in enumerate(points)
        if p.x in {min_x, max_x} or p.y in {min_y, max_y}
    ]
    center = min(range(len(points)), key=lambda i: _distance(points[i], Point((min_x + max_x) / 2, (min_y + max_y) / 2)))
    gateways = sorted(boundary, key=lambda i: _distance(points[i], points[center]))[: max(2, min(4, len(boundary)))]

    segment_by_pair = {(min(s.a, s.b), max(s.a, s.b)): s for s in segments}
    arterial_parents: set[str] = set()
    for corridor_index, gateway in enumerate(gateways[: max(1, config.arterial_count)]):
        path = _shortest_unweighted_path(gateway, center, adjacency)
        for a, b in zip(path, path[1:]):
            segment = segment_by_pair.get((min(a, b), max(a, b)))
            if segment is not None:
                arterial_parents.add(segment.parent_road_id)
    return segments, arterial_parents


def _shortest_unweighted_path(source: int, target: int, adjacency: dict[int, list[int]]) -> list[int]:
    queue = [source]
    previous: dict[int, int | None] = {source: None}
    for current in queue:
        if current == target:
            break
        for neighbour in sorted(adjacency[current]):
            if neighbour not in previous:
                previous[neighbour] = current
                queue.append(neighbour)
    if target not in previous:
        return [source]
    path = []
    current: int | None = target
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))


def _segments_to_edges(
    nodes: list[dict], segments: list[RoadSegment], seed: int, arterial_parents: set[str] | None = None
) -> list[dict]:
    rng = random.Random(seed + 17)
    arterial_parents = arterial_parents or set()
    degree = {node["node_id"]: 0 for node in nodes}
    for segment in segments:
        a_id = nodes[segment.a]["node_id"]
        b_id = nodes[segment.b]["node_id"]
        degree[a_id] += 1
        degree[b_id] += 1

    edges: list[dict] = []
    for segment in segments:
        start = nodes[segment.a]
        end = nodes[segment.b]
        length = math.hypot(start["x_m"] - end["x_m"], start["y_m"] - end["y_m"])
        a_id = nodes[segment.a]["node_id"]
        b_id = nodes[segment.b]["node_id"]
        road_class = (
            RoadClass.ARTERIAL
            if segment.parent_road_id in arterial_parents
            else _class_from_corridor_degree(max(degree[a_id], degree[b_id]))
        )
        speed_kmh, lanes = _road_attributes(road_class)
        capacity = lanes * 900.0
        for source, target in ((segment.a, segment.b), (segment.b, segment.a)):
            edges.append(
                {
                    "edge_id": f"E{len(edges):06d}",
                    "parent_road_id": segment.parent_road_id,
                    "from_node": nodes[source]["node_id"],
                    "to_node": nodes[target]["node_id"],
                    "length_m": length,
                    "road_class": road_class,
                    "speed_limit_mps": speed_kmh / 3.6,
                    "lane_count": lanes,
                    "capacity_veh_per_hour": capacity,
                    "source_edge_id": None,
                    "provenance": "synthetic_step4",
                    "open_by_default": True,
                }
            )
    return edges


def _class_from_corridor_degree(degree: int) -> RoadClass:
    if degree >= 4:
        return RoadClass.ARTERIAL
    if degree == 3:
        return RoadClass.COLLECTOR
    return RoadClass.LOCAL


def _road_attributes(road_class: RoadClass) -> tuple[int, int]:
    if road_class == RoadClass.LOCAL:
        return 30, 1
    if road_class == RoadClass.COLLECTOR:
        return 40, 1
    return 50, 2


def _assign_signals(nodes: list[dict], edges: list[dict], fraction: float, seed: int) -> None:
    degree: dict[str, set[str]] = {node["node_id"]: set() for node in nodes}
    for edge in edges:
        degree[edge["from_node"]].add(edge["to_node"])
    candidates = [node for node in nodes if len(degree[node["node_id"]]) in {3, 4}]
    candidates.sort(key=lambda node: _stable_score(seed, node["node_id"]))
    for node in candidates[: int(len(candidates) * fraction)]:
        node["signalized"] = True


def _apply_safe_one_way(nodes: list[dict], edges: list[dict], fraction: float, seed: int) -> None:
    """Remove reverse directions only when the directed graph stays strongly connected."""

    by_parent: dict[str, list[dict]] = {}
    for edge in edges:
        by_parent.setdefault(edge["parent_road_id"], []).append(edge)
    candidates = sorted(by_parent.items(), key=lambda item: _stable_score(seed, item[0]))
    target = int(len(candidates) * fraction)
    changed = 0

    for _, pair in candidates:
        if changed >= target or len(pair) != 2:
            continue
        remove = pair[_stable_score(seed, pair[0]["edge_id"]) % 2]
        edges.remove(remove)
        if _strongly_connected(nodes, edges):
            changed += 1
        else:
            edges.append(remove)
            edges.sort(key=lambda edge: int(edge["edge_id"][1:]))


def _stable_score(seed: int, value: str) -> int:
    """Small deterministic integer score used instead of random ordering."""
    score = (seed ^ 0x9E3779B9) & 0xFFFFFFFF
    for char in value:
        score = ((score * 16777619) ^ ord(char)) & 0xFFFFFFFF
    return score


def _strongly_connected(nodes: list[dict], edges: list[dict]) -> bool:
    if not nodes:
        return False
    adjacency = {node["node_id"]: set() for node in nodes}
    reverse = {node["node_id"]: set() for node in nodes}
    for edge in edges:
        adjacency[edge["from_node"]].add(edge["to_node"])
        reverse[edge["to_node"]].add(edge["from_node"])
    start = nodes[0]["node_id"]
    return len(_reachable(start, adjacency)) == len(nodes) and len(_reachable(start, reverse)) == len(nodes)


def _reachable(start: str, adjacency: dict[str, set[str]]) -> set[str]:
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbour in adjacency[current]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


def _split_edge(edge: dict, access_node: str, fraction: float, start_id: int) -> tuple[dict, dict]:
    first = dict(edge)
    second = dict(edge)
    first["edge_id"] = f"E{start_id:06d}"
    second["edge_id"] = f"E{start_id + 1:06d}"
    first["to_node"] = access_node
    second["from_node"] = access_node
    first["length_m"] = edge["length_m"] * fraction
    second["length_m"] = edge["length_m"] * (1 - fraction)
    first["source_edge_id"] = edge["edge_id"]
    second["source_edge_id"] = edge["edge_id"]
    return first, second


def _next_node_number(nodes: list[dict]) -> int:
    return max((int(node["node_id"][1:]) for node in nodes), default=-1) + 1


def _next_edge_number(edges: list[dict]) -> int:
    return max((int(edge["edge_id"][1:]) for edge in edges), default=-1) + 1


def _node_point(nodes: list[dict], node_id: str) -> Point:
    node = next(node for node in nodes if node["node_id"] == node_id)
    return Point(node["x_m"], node["y_m"])


def _project(point: Point, start: Point, end: Point) -> tuple[Point, float, float]:
    dx, dy = end.x - start.x, end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return start, math.hypot(point.x - start.x, point.y - start.y), 0.0
    fraction = ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_sq
    fraction = max(0.0, min(1.0, fraction))
    projection = Point(start.x + fraction * dx, start.y + fraction * dy)
    return projection, math.hypot(point.x - projection.x, point.y - projection.y), fraction


def _zone(point: Point, points: list[Point]) -> str:
    if not points:
        return "Z000"
    width = max(p.x for p in points) - min(p.x for p in points)
    height = max(p.y for p in points) - min(p.y for p in points)
    cols = 4
    rows = 4
    col = min(cols - 1, int(cols * point.x / max(width, 1e-9)))
    row = min(rows - 1, int(rows * point.y / max(height, 1e-9)))
    return f"Z{row * cols + col:03d}"


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _nearest_neighbor_edges(points: list[Point], k: int) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for index, point in enumerate(points):
        neighbours = sorted(
            ((j, _distance(point, other)) for j, other in enumerate(points) if j != index),
            key=lambda item: item[1],
        )[:k]
        for j, _ in neighbours:
            result.add((min(index, j), max(index, j)))
    return result


def _mst(segments: list[RoadSegment], points: list[Point]) -> list[RoadSegment]:
    parent = list(range(len(points)))
    rank = [0] * len(points)
    chosen: list[RoadSegment] = []
    for segment in sorted(segments, key=lambda s: _distance(points[s.a], points[s.b])):
        a, b = _find(parent, segment.a), _find(parent, segment.b)
        if a == b:
            continue
        if rank[a] < rank[b]:
            parent[a] = b
        elif rank[a] > rank[b]:
            parent[b] = a
        else:
            parent[b] = a
            rank[a] += 1
        chosen.append(segment)
    if len(chosen) != max(0, len(points) - 1):
        raise ValueError("Candidate road skeleton cannot produce a connected backbone")
    return chosen


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _fallback_connected_segments(points: list[Point]) -> list[RoadSegment]:
    order = sorted(range(len(points)), key=lambda i: (points[i].x, points[i].y))
    return [RoadSegment(order[i], order[i + 1], f"F{i:05d}") for i in range(len(order) - 1)]


def _all_collinear(points: list[Point]) -> bool:
    a, b = points[0], points[1]
    for point in points[2:]:
        if abs((b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)) > 1e-9:
            return False
    return True


def _bowyer_watson(points: list[Point]) -> set[tuple[int, int, int]]:
    """Deterministic incremental Delaunay triangulation for the debug fixture."""

    min_x = min(p.x for p in points)
    max_x = max(p.x for p in points)
    min_y = min(p.y for p in points)
    max_y = max(p.y for p in points)
    span = max(max_x - min_x, max_y - min_y) or 1.0
    mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    super_points = points + [
        Point(mid_x - 20 * span, mid_y - span),
        Point(mid_x, mid_y + 20 * span),
        Point(mid_x + 20 * span, mid_y - span),
    ]
    super_ids = (len(points), len(points) + 1, len(points) + 2)
    triangles: set[tuple[int, int, int]] = {super_ids}

    for point_index in range(len(points)):
        bad = {
            triangle
            for triangle in triangles
            if _inside_circumcircle(super_points[triangle[0]], super_points[triangle[1]], super_points[triangle[2]], super_points[point_index])
        }
        boundary: dict[tuple[int, int], int] = {}
        for triangle in bad:
            for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                key = (min(a, b), max(a, b))
                boundary[key] = boundary.get(key, 0) + 1
        triangles.difference_update(bad)
        for a, b in sorted(key for key, count in boundary.items() if count == 1):
            triangles.add(_ordered_triangle(a, b, point_index))

    return {triangle for triangle in triangles if all(vertex < len(points) for vertex in triangle)}


def _ordered_triangle(a: int, b: int, c: int) -> tuple[int, int, int]:
    triangle = (a, b, c)
    area = (b - a) if False else 0
    return tuple(sorted(triangle))


def _inside_circumcircle(a: Point, b: Point, c: Point, p: Point) -> bool:
    ax, ay = a.x - p.x, a.y - p.y
    bx, by = b.x - p.x, b.y - p.y
    cx, cy = c.x - p.x, c.y - p.y
    determinant = (
        (ax * ax + ay * ay) * (bx * cy - by * cx)
        - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx)
    )
    orientation = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
    if orientation > 0:
        return determinant > 1e-9
    return determinant < -1e-9
