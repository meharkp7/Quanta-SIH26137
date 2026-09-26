"""HERE live-traffic overlay for the routing graph (optional, honest).

``HereFeed`` is the only place where HERE flow data meets Quanta's road
graph. It stays behind the same optional-dependency rule as OSMnx
ingestion: no key, no pyproj, or no network means a *clear reason* is
reported (and a hard error only when the caller explicitly demanded live
traffic) — never invented speeds.

How a HERE flow link becomes an edge speed:

1. Project every node to WGS84 (``serialize.to_latlon``) and take three
   probes per directed edge (30% / 50% / 70% along the chord).
2. Fetch ``v7/flow`` for the scenario bounding box (cached 60 s).
3. Grid-index the flow shape points; for each probe keep the nearest
   point-to-polyline hit inside ``MATCH_RADIUS_M``.
4. Override ``RoadEdge.speed_limit_mps`` with that link's current speed
   (m/s). Contract floor of 0.1 m/s keeps the "> 0" validation true;
   links HERE marks ``traversability == "closed"`` are skipped — road
   closures remain the UI's own explicit control, not a hidden override.

The overlaid scenario carries a ``:here-live`` suffix on
``graph_version`` so caches keyed on it cannot serve stale speeds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

from src.contracts.scenario import Scenario
from src.data.here_api import HEREClient, HEREError, HERENotConfigured, FlowSegment
from src.platform.serialize import is_geo_scenario, to_latlon

#: Acceptance radius between an edge probe and a HERE flow shape (metres).
MATCH_RADIUS_M = 150.0
#: Flow fetches and matched overrides are reused for this long.
FEED_TTL_S = 60.0
#: Latitude/longitude grid cell (~0.005° ≈ 465–555 m) for the shape index.
CELL_DEG = 0.005
#: Metres per degree used for grid span math (worst case near the poles
#: of our operating latitudes, so the ring is never too small).
_M_PER_DEG_MIN = 70000.0
#: Contract floor: ``speed_limit_mps`` must stay > 0 (RoadEdge validator).
MIN_SPEED_MPS = 0.1
#: Small caches — evict oldest entry past this many keys.
CACHE_MAX_ENTRIES = 32


@dataclass(frozen=True)
class EdgeProbe:
    """One WGS84 sample point that represents an edge while matching."""

    edge_id: str
    lat: float
    lon: float


def _to_xy(lat: float, lon: float, ref_lat: float) -> tuple[float, float]:
    """Local flat-earth metres relative to ``ref_lat`` (equirectangular).

    Good to well under a metre over the ≤200 m matching distances here.
    """
    radius = 6371000.0
    x = math.radians(lon) * radius * math.cos(math.radians(ref_lat))
    y = math.radians(lat) * radius
    return x, y


def _point_segment_distance_m(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class FlowIndex:
    """Grid index over HERE flow shapes for nearest-polyline queries."""

    def __init__(
        self,
        segments: Sequence[FlowSegment],
        cell_deg: float = CELL_DEG,
    ) -> None:
        self._cell = float(cell_deg)
        self._polylines: list[tuple[FlowSegment, tuple[tuple[float, float], ...]]] = []
        cells: dict[tuple[int, int], list[int]] = {}
        for segment in segments:
            for line in segment.polylines:
                if len(line) < 2:
                    continue
                index = len(self._polylines)
                self._polylines.append((segment, line))
                for lat, lon in line:
                    cells.setdefault(self._cell_of(lat, lon), []).append(index)
        self._cells = cells

    def _cell_of(self, lat: float, lon: float) -> tuple[int, int]:
        return math.floor(lat / self._cell), math.floor(lon / self._cell)

    def __len__(self) -> int:
        return len(self._polylines)

    def nearest(
        self, lat: float, lon: float, radius_m: float
    ) -> tuple[FlowSegment, float] | None:
        """Nearest ``(segment, distance_m)`` within ``radius_m`` or None."""
        if not self._polylines:
            return None
        ix, iy = self._cell_of(lat, lon)
        span = max(1, math.ceil(radius_m / (self._cell * _M_PER_DEG_MIN)))
        candidates: set[int] = set()
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                candidates.update(self._cells.get((ix + dx, iy + dy), ()))
        if not candidates:
            return None
        qx, qy = _to_xy(lat, lon, lat)
        best: tuple[float, FlowSegment] | None = None
        for index in candidates:
            segment, line = self._polylines[index]
            for (lat1, lon1), (lat2, lon2) in zip(line, line[1:]):
                ax, ay = _to_xy(lat1, lon1, lat)
                bx, by = _to_xy(lat2, lon2, lat)
                distance = _point_segment_distance_m(qx, qy, ax, ay, bx, by)
                if best is None or distance < best[0]:
                    best = (distance, segment)
        if best is None or best[0] > radius_m:
            return None
        return best[1], best[0]


def match_edge_speeds(
    probes: Sequence[EdgeProbe],
    index: FlowIndex,
    radius_m: float = MATCH_RADIUS_M,
) -> dict[str, float]:
    """Map edge ids to HERE current speeds (m/s), nearest probe wins.

    Closed links and non-finite/zero speeds are skipped so the graph never
    receives a value its contract would reject or a speed that pretends a
    blocked link is flowing.
    """
    best_distance: dict[str, float] = {}
    matched: dict[str, float] = {}
    for probe in probes:
        hit = index.nearest(probe.lat, probe.lon, radius_m)
        if hit is None:
            continue
        segment, distance = hit
        if segment.traversability == "closed":
            continue
        speed = float(segment.speed_mps)
        if not math.isfinite(speed):
            continue
        speed = max(MIN_SPEED_MPS, speed)
        previous = best_distance.get(probe.edge_id)
        if previous is None or distance < previous:
            best_distance[probe.edge_id] = distance
            matched[probe.edge_id] = speed
    return matched


class HereFeed:
    """Optional HERE-backed live layer shared by ``PlatformService``.

    ``client`` may be injected (tests pass a fake); otherwise it is built
    lazily from the environment so importing this module never reads keys
    and never touches the network.
    """

    def __init__(
        self,
        client: HEREClient | None = None,
        *,
        ttl_s: float = FEED_TTL_S,
        match_radius_m: float = MATCH_RADIUS_M,
    ) -> None:
        self._client = client
        self._ttl_s = float(ttl_s)
        self._match_radius_m = float(match_radius_m)
        # scenario_id → (fetched_at, overrides, meta)
        self._overrides_cache: dict[str, tuple[float, dict, dict]] = {}
        # rounded bbox → (fetched_at, segments, source_updated)
        self._flow_cache: dict[tuple, tuple[float, list[FlowSegment], str | None]] = {}

    @property
    def client(self) -> HEREClient:
        if self._client is None:
            self._client = HEREClient()
        return self._client

    # ── status / search ─────────────────────────────────────────────────

    def status(self) -> dict:
        """Configuration status only — no network probe, no overclaiming."""
        config = self.client.config
        payload: dict = {
            "configured": config.configured,
            "available": config.configured,
            "key_env": config.key_source,
            "source": "HERE Geocoding v1 · Routing v8 · Traffic Flow v7",
            "match_radius_m": self._match_radius_m,
            "cache_ttl_s": self._ttl_s,
        }
        if config.configured:
            payload["note"] = (
                "Key present. Reachability and quota are verified on the "
                f"first real request; live speeds refresh every {self._ttl_s:.0f}s."
            )
        else:
            payload["reason"] = (
                "No HERE API key found. Set HERE_API_KEY (environment or "
                "repo .env) — create a free key at https://portal.here.com. "
                "Until then geocoding, HERE routing, and live traffic stay "
                "off and the UI shows only the static graph."
            )
        return payload

    def geocode(
        self,
        query: str,
        *,
        limit: int = 5,
        at: tuple[float, float] | None = None,
    ) -> dict:
        """Forward geocode a place/address into JSON for the API layer."""
        text = (query or "").strip()
        if not text:
            raise ValueError("search query must not be empty")
        items = self.client.geocode(text, limit=limit, at=at)
        return {
            "query": text,
            "count": len(items),
            "source": "HERE Geocoding v1",
            "results": [
                {
                    "label": item.label,
                    "lat": item.lat,
                    "lon": item.lon,
                    "city": item.city,
                    "state": item.state,
                    "country": item.country,
                    "result_type": item.result_type,
                    "distance_m": item.distance_m,
                }
                for item in items
            ],
        }

    # ── graph coordinates ───────────────────────────────────────────────

    def node_positions(self, scenario: Scenario) -> dict[str, tuple[float, float]]:
        """WGS84 position per node, or a clear ``ValueError`` when impossible."""
        if not is_geo_scenario(scenario):
            raise ValueError(
                f"{scenario.scenario_id} uses synthetic fixture coordinates — "
                "HERE lookups need a geo (OSM-projected) map such as the "
                "DELHI_* scenarios."
            )
        positions: dict[str, tuple[float, float]] = {}
        for node in scenario.nodes:
            converted = to_latlon(float(node.x_m), float(node.y_m))
            if converted is None:
                raise ValueError(
                    f"node {node.node_id} could not be projected to lat/lon — "
                    "HERE lookups need pyproj and a projected OSM graph."
                )
            positions[str(node.node_id)] = converted
        return positions

    @staticmethod
    def _probes(
        scenario: Scenario, positions: dict[str, tuple[float, float]]
    ) -> list[EdgeProbe]:
        """Three samples per edge (30/50/70% along the node chord).

        Endpoints are skipped on purpose: they are junctions where a
        neighbouring road could be the true nearest flow link.
        """
        probes: list[EdgeProbe] = []
        for edge in scenario.edges:
            origin = positions.get(str(edge.from_node))
            destination = positions.get(str(edge.to_node))
            if origin is None or destination is None:
                continue
            for fraction in (0.3, 0.5, 0.7):
                probes.append(
                    EdgeProbe(
                        edge_id=str(edge.edge_id),
                        lat=origin[0] + (destination[0] - origin[0]) * fraction,
                        lon=origin[1] + (destination[1] - origin[1]) * fraction,
                    )
                )
        return probes

    @staticmethod
    def _bbox(positions: dict[str, tuple[float, float]]) -> tuple[float, float, float, float]:
        lats = [lat for lat, _ in positions.values()]
        lons = [lon for _, lon in positions.values()]
        pad = 0.005  # ~500 m of margin so boundary links are inside the box
        return (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)

    # ── flow fetch + match ──────────────────────────────────────────────

    def _flow(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[list[FlowSegment], str | None]:
        key = tuple(round(value, 4) for value in bbox)
        now = time.monotonic()
        cached = self._flow_cache.get(key)
        if cached is not None and now - cached[0] <= self._ttl_s:
            return cached[1], cached[2]
        segments, updated = self.client.flow_bbox(bbox)
        if len(self._flow_cache) >= CACHE_MAX_ENTRIES:
            self._flow_cache.pop(next(iter(self._flow_cache)))
        self._flow_cache[key] = (now, segments, updated)
        return segments, updated

    def edge_speeds(
        self, scenario: Scenario, *, strict: bool = False
    ) -> tuple[dict[str, float], dict]:
        """Matched HERE speeds + honest status metadata for one scenario.

        ``strict=True`` raises (:class:`HERENotConfigured`,
        :class:`HEREError`, or ``ValueError``) when live speeds cannot be
        produced — that is what solve/validate/path ask for when the caller
        explicitly requested live traffic. ``strict=False`` degrades to
        ``( {}, reason )`` for read-only views like the graph payload.
        """
        meta: dict = {
            "requested": True,
            "applied": False,
            "available": False,
            "matched": 0,
            "edges": len(scenario.edges),
            "reason": None,
            "source": "HERE Traffic Flow v7 (speeds in m/s)",
            "source_updated": None,
            "fetched_at": None,
        }
        cached = self._overrides_cache.get(str(scenario.scenario_id))
        if cached is not None and time.monotonic() - cached[0] <= self._ttl_s:
            _at, overrides, cached_meta = cached
            meta.update(dict(cached_meta))
            return dict(overrides), meta
        try:
            if not self.client.config.configured:
                raise HERENotConfigured(
                    "HERE API key is not configured — set HERE_API_KEY "
                    "(free key at https://portal.here.com) to enable live traffic."
                )
            positions = self.node_positions(scenario)
            segments, updated = self._flow(self._bbox(positions))
            meta.update(
                {"source_updated": updated, "fetched_at": time.time()}
            )
            if not segments:
                meta["reason"] = (
                    "HERE traffic flow returned no links inside this map's "
                    "bounding box — the static graph is used unchanged."
                )
                self._remember(str(scenario.scenario_id), {}, meta)
                return {}, meta
            overrides = match_edge_speeds(
                self._probes(scenario, positions),
                FlowIndex(segments),
                self._match_radius_m,
            )
            meta.update(
                {
                    "available": True,
                    "matched": len(overrides),
                    "segments": len(segments),
                }
            )
            if not overrides:
                meta["reason"] = (
                    f"No traversable HERE flow link matched any of the "
                    f"{meta['edges']} edges within "
                    f"{self._match_radius_m:.0f} m — static speeds kept."
                )
            self._remember(str(scenario.scenario_id), overrides, meta)
            return overrides, meta
        except (HEREError, ValueError) as exc:
            if strict:
                raise
            meta["reason"] = str(exc)
            return {}, meta

    def _remember(self, scenario_id: str, overrides: dict, meta: dict) -> None:
        if len(self._overrides_cache) >= CACHE_MAX_ENTRIES:
            self._overrides_cache.pop(next(iter(self._overrides_cache)))
        self._overrides_cache[scenario_id] = (time.monotonic(), dict(overrides), dict(meta))

    def apply(
        self, scenario: Scenario, *, strict: bool = False
    ) -> tuple[Scenario, dict]:
        """Scenario with HERE speeds overlaid (or unchanged) + metadata."""
        overrides, meta = self.edge_speeds(scenario, strict=strict)
        if not overrides:
            return scenario, meta
        speeds = {
            str(edge.edge_id): overrides[str(edge.edge_id)]
            for edge in scenario.edges
            if str(edge.edge_id) in overrides
        }
        if not speeds:
            return scenario, meta
        edges = tuple(
            edge.model_copy(update={"speed_limit_mps": speeds[str(edge.edge_id)]})
            if str(edge.edge_id) in speeds
            else edge
            for edge in scenario.edges
        )
        overlaid = scenario.model_copy(
            update={
                "edges": edges,
                "graph_version": f"{scenario.graph_version}:here-live",
            }
        )
        result = dict(meta)
        result["applied"] = True
        return overlaid, result

    # ── HERE routing between two graph nodes ────────────────────────────

    def route(
        self,
        scenario: Scenario,
        source: str,
        target: str,
        *,
        transport_mode: str = "car",
    ) -> dict:
        """HERE's own traffic-enabled drive time between two graph nodes."""
        if not self.client.config.configured:
            raise HERENotConfigured(
                "HERE API key is not configured — set HERE_API_KEY "
                "(free key at https://portal.here.com) to request HERE routes."
            )
        positions = self.node_positions(scenario)
        origin = positions.get(str(source))
        destination = positions.get(str(target))
        if origin is None:
            raise ValueError(f"node {source} has no lat/lon in {scenario.scenario_id}")
        if destination is None:
            raise ValueError(f"node {target} has no lat/lon in {scenario.scenario_id}")
        summary = self.client.route(
            origin, destination, transport_mode=transport_mode
        )
        return {
            "source": str(source),
            "target": str(target),
            "origin": {"lat": origin[0], "lon": origin[1]},
            "destination": {"lat": destination[0], "lon": destination[1]},
            "transport_mode": transport_mode,
            "duration_s": summary.duration_s,
            "distance_m": summary.length_m,
            "base_duration_s": summary.base_duration_s,
            "delay_s": summary.delay_s,
            "source_api": "HERE Routing v8 (traffic enabled)",
            "note": (
                "HERE's own on-road time for these two nodes, shown next to "
                "the in-graph Dijkstra result — both are reported as measured "
                "and never blended into one number."
            ),
        }
