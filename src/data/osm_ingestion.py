"""Production-oriented OpenStreetMap road-network ingestion.

The module intentionally keeps OSMnx optional.  Core Quanta installations can
continue to run without OSM support; attempting to use an OSM loader without
installing the optional dependency fails with an actionable error.

The ingestion boundary converts an OSMnx/NetworkX MultiDiGraph into Quanta's
validated ``RoadNode`` and ``RoadEdge`` contracts.  It does not fabricate
requests, vehicles, traffic observations, or forecasts.  Those remain separate
pipeline concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping

from src.contracts import NodeKind, RoadClass, RoadEdge, RoadNode


class OSMIngestionError(ValueError):
    """Raised when an OSM graph cannot be converted safely."""


@dataclass(frozen=True, slots=True)
class OSMIngestionConfig:
    """Explicit policies for converting OSM attributes into Quanta units."""

    default_speed_limit_mps: float = 13.8888888889  # 50 km/h
    default_lane_count: int = 1
    capacity_per_lane_veh_per_hour: float = 900.0
    require_length: bool = True
    require_projected_coordinates: bool = True
    provenance: str = "openstreetmap"

    def __post_init__(self) -> None:
        if self.default_speed_limit_mps <= 0:
            raise ValueError("default_speed_limit_mps must be positive")
        if self.default_lane_count <= 0:
            raise ValueError("default_lane_count must be positive")
        if self.capacity_per_lane_veh_per_hour <= 0:
            raise ValueError("capacity_per_lane_veh_per_hour must be positive")
        if not self.provenance.strip():
            raise ValueError("provenance must not be empty")


@dataclass(frozen=True, slots=True)
class OSMNetwork:
    """Converted Quanta network plus the source CRS metadata."""

    nodes: tuple[RoadNode, ...]
    edges: tuple[RoadEdge, ...]
    source_crs: str | None
    projected: bool


_HIGHWAY_CLASS = {
    "motorway": RoadClass.ARTERIAL,
    "motorway_link": RoadClass.ACCESS_CONNECTOR,
    "trunk": RoadClass.ARTERIAL,
    "trunk_link": RoadClass.ACCESS_CONNECTOR,
    "primary": RoadClass.ARTERIAL,
    "primary_link": RoadClass.ACCESS_CONNECTOR,
    "secondary": RoadClass.COLLECTOR,
    "secondary_link": RoadClass.ACCESS_CONNECTOR,
    "tertiary": RoadClass.COLLECTOR,
    "tertiary_link": RoadClass.ACCESS_CONNECTOR,
    "residential": RoadClass.LOCAL,
    "living_street": RoadClass.LOCAL,
    "unclassified": RoadClass.LOCAL,
    "service": RoadClass.ACCESS_CONNECTOR,
    "road": RoadClass.LOCAL,
}

_SPEED_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _require_osmnx() -> Any:
    try:
        import osmnx as ox  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise OSMIngestionError(
            "OSM ingestion requires the optional 'osmnx' dependency. "
            "Install it in the deployment environment before using OSM loaders."
        ) from exc
    return ox


def load_osm_xml(path: str, *, config: OSMIngestionConfig | None = None) -> OSMNetwork:
    """Load a local OSM XML/PBF-compatible extract through OSMnx.

    Network access is deliberately not performed by this function.  Keeping
    acquisition separate makes deployments reproducible and auditable.
    """
    ox = _require_osmnx()
    cfg = config or OSMIngestionConfig()
    graph = ox.graph_from_xml(path, simplify=False, retain_all=True)
    return graph_to_quanta(graph, config=cfg)


def load_graphml(path: str, *, config: OSMIngestionConfig | None = None) -> OSMNetwork:
    """Load a previously exported OSMnx GraphML network."""
    ox = _require_osmnx()
    cfg = config or OSMIngestionConfig()
    graph = ox.load_graphml(path)
    return graph_to_quanta(graph, config=cfg)


def graph_to_quanta(graph: Any, *, config: OSMIngestionConfig | None = None) -> OSMNetwork:
    """Convert an OSMnx/NetworkX directed graph into Quanta contracts.

    The graph must expose node ``x``/``y`` coordinates in metres when
    ``require_projected_coordinates`` is enabled.  This avoids the dangerous
    mistake of treating longitude/latitude degrees as Euclidean metres.
    """
    cfg = config or OSMIngestionConfig()
    if graph is None or not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
        raise OSMIngestionError("graph must provide NetworkX-compatible nodes and edges")

    graph_attrs = getattr(graph, "graph", {}) or {}
    crs = graph_attrs.get("crs")
    projected = bool(getattr(crs, "is_projected", False))
    if cfg.require_projected_coordinates and not projected:
        raise OSMIngestionError(
            "OSM graph is not projected. Project the graph to a metric CRS before ingestion "
            "so x/y coordinates and edge lengths are expressed in metres."
        )

    node_records: list[RoadNode] = []
    node_ids: dict[Any, str] = {}
    for raw_id, attrs in graph.nodes(data=True):
        if "x" not in attrs or "y" not in attrs:
            raise OSMIngestionError(f"OSM node {raw_id!r} is missing x/y coordinates")
        x = _finite_float(attrs["x"], f"node {raw_id!r} x")
        y = _finite_float(attrs["y"], f"node {raw_id!r} y")
        node_id = f"osm:n:{raw_id}"
        node_ids[raw_id] = node_id
        node_records.append(
            RoadNode(
                node_id=node_id,
                x_m=x,
                y_m=y,
                kind=NodeKind.JUNCTION,
                zone_id=str(attrs.get("zone", "osm")),
                signalized=_as_bool(attrs.get("highway"), "traffic_signals"),
            )
        )

    edge_records: list[RoadEdge] = []
    for ordinal, (u, v, key, attrs) in enumerate(_iter_multiedges(graph)):
        if u not in node_ids or v not in node_ids:
            raise OSMIngestionError(f"edge {u!r}->{v!r} references an unknown node")
        length = _edge_length(attrs)
        if length is None:
            if cfg.require_length:
                raise OSMIngestionError(
                    f"OSM edge {u!r}->{v!r}/{key!r} has no valid length in metres"
                )
            length = _euclidean_length(node_records, node_ids[u], node_ids[v])
        speed = _parse_speed_mps(attrs.get("maxspeed"), cfg.default_speed_limit_mps)
        lanes = _parse_positive_int(attrs.get("lanes"), cfg.default_lane_count)
        highway = _first_tag(attrs.get("highway"))
        road_class = _HIGHWAY_CLASS.get(highway, RoadClass.LOCAL)
        parent = str(attrs.get("ref") or attrs.get("name") or f"osm:{u}-{v}")
        edge_id = f"osm:e:{u}:{v}:{key}:{ordinal}"
        edge_records.append(
            RoadEdge(
                edge_id=edge_id,
                parent_road_id=f"osm:road:{parent}",
                from_node=node_ids[u],
                to_node=node_ids[v],
                length_m=length,
                road_class=road_class,
                speed_limit_mps=speed,
                lane_count=lanes,
                capacity_veh_per_hour=lanes * cfg.capacity_per_lane_veh_per_hour,
                source_edge_id=str(key),
                provenance=cfg.provenance,
                open_by_default=True,
            )
        )

    return OSMNetwork(
        nodes=tuple(sorted(node_records, key=lambda n: n.node_id)),
        edges=tuple(edge_records),
        source_crs=str(crs) if crs is not None else None,
        projected=projected,
    )


def _iter_multiedges(graph: Any) -> Iterable[tuple[Any, Any, Any, Mapping[str, Any]]]:
    try:
        edges = graph.edges(keys=True, data=True)
    except TypeError:
        # A plain DiGraph has no edge keys; retain a deterministic sentinel.
        edges = ((u, v, 0, data) for u, v, data in graph.edges(data=True))
    for u, v, key, attrs in edges:
        yield u, v, key, attrs


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OSMIngestionError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise OSMIngestionError(f"{label} must be finite")
    return result


def _edge_length(attrs: Mapping[str, Any]) -> float | None:
    if "length" not in attrs:
        return None
    try:
        value = float(attrs["length"])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _parse_speed_mps(value: Any, default: float) -> float:
    if value is None:
        return default
    raw = _first_tag(value)
    match = _SPEED_RE.search(raw.replace(",", "."))
    if not match:
        return default
    numeric = float(match.group())
    lower = raw.lower()
    if "mph" in lower:
        numeric *= 0.44704
    else:
        # OSM maxspeed values are conventionally km/h unless explicitly mph.
        numeric /= 3.6
    return numeric if math.isfinite(numeric) and numeric > 0 else default


def _parse_positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    match = _SPEED_RE.search(_first_tag(value))
    if not match:
        return default
    try:
        result = int(float(match.group()))
    except ValueError:
        return default
    return result if result > 0 else default


def _first_tag(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return str(next(iter(value), ""))
    return str(value)


def _as_bool(value: Any, expected: str) -> bool:
    return _first_tag(value).lower() == expected


def _euclidean_length(nodes: list[RoadNode], node_id_a: str, node_id_b: str) -> float:
    lookup = {n.node_id: n for n in nodes}
    a, b = lookup[node_id_a], lookup[node_id_b]
    length = math.hypot(a.x_m - b.x_m, a.y_m - b.y_m)
    if length <= 0:
        raise OSMIngestionError(f"cannot infer a positive length for {node_id_a}->{node_id_b}")
    return length
