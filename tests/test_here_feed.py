"""HERE integration tests — every network call goes through a fake client.

No test in this file touches the real HERE API: coverage is the matching
logic, the strict/lenient failure contract, caching, and the honest status
metadata that the UI shows when a key (or pyproj, or a geo map) is missing.
"""

from __future__ import annotations

import pytest

from src.contracts.scenario import Scenario
from src.data.here_api import (
    HEREClient,
    HEREConfig,
    HEREError,
    HERENotConfigured,
    FlowSegment,
    GeocodeItem,
    RouteSummary,
)
from src.platform.here_feed import (
    EdgeProbe,
    FlowIndex,
    HereFeed,
    match_edge_speeds,
)
from src.platform.serialize import to_latlon

# pyproj is an optional dependency (serialize.py degrades without it), so
# every test that must project node coordinates is skipped when absent.
HAS_PYPROJ = to_latlon(300000.0, 2860000.0) is not None
requires_pyproj = pytest.mark.skipif(
    not HAS_PYPROJ, reason="pyproj is not installed (geo projection unavailable)"
)


class FakeHEREClient:
    """Deterministic HERE stand-in: records calls, never opens a socket."""

    def __init__(
        self,
        segments: tuple[FlowSegment, ...] = (),
        items: tuple[GeocodeItem, ...] = (),
        *,
        flow_error: Exception | None = None,
        api_key: str | None = "test-key",
    ) -> None:
        self.config = HEREConfig(api_key=api_key)
        self.segments = list(segments)
        self.items = list(items)
        self.flow_error = flow_error
        self.flow_calls = 0
        self.geocode_calls = 0

    def flow_bbox(self, bbox):
        self.flow_calls += 1
        if self.flow_error is not None:
            raise self.flow_error
        return list(self.segments), "2026-09-25T09:30:00Z"

    def geocode(self, query, *, limit=5, at=None):
        self.geocode_calls += 1
        return list(self.items)[:limit]

    def route(self, origin, destination, *, transport_mode="car"):
        return RouteSummary(
            duration_s=120.0,
            length_m=900.0,
            base_duration_s=100.0,
            delay_s=20.0,
            transport_mode=transport_mode,
        )


def geo_scenario() -> Scenario:
    """Tiny 3-node graph in UTM-43N-like metres (Delhi range).

    ``is_geo_scenario``'s fallback band is 1e5..9e5 easting / 1e6..9e6
    northing, which these coordinates satisfy without any HERE knowledge.
    """
    return Scenario.model_validate(
        {
            "schema_version": "1.0",
            "scenario_id": "TEST_GEO",
            "source_name": "synthetic geo test",
            "source_checksum": None,
            "units": {
                "distance": "m",
                "time": "s",
                "speed": "m/s",
                "demand": "load_units",
            },
            "coordinate_transform": {
                "scale": 1.0,
                "translation_x_m": 0.0,
                "translation_y_m": 0.0,
                "description": "synthetic metres in the UTM 43N band",
            },
            "graph_version": "TEST_GEO:graph:1",
            "nodes": [
                {"node_id": "GN1", "x_m": 300000.0, "y_m": 2860000.0, "kind": "depot", "zone_id": "Z"},
                {"node_id": "GN2", "x_m": 300300.0, "y_m": 2860000.0, "kind": "customer_access", "zone_id": "Z"},
                {"node_id": "GN3", "x_m": 301000.0, "y_m": 2861000.0, "kind": "junction", "zone_id": "Z"},
            ],
            "edges": [
                {
                    "edge_id": "GE1",
                    "parent_road_id": "GR1",
                    "from_node": "GN1",
                    "to_node": "GN2",
                    "length_m": 300.0,
                    "road_class": "arterial",
                    "speed_limit_mps": 8.0,
                    "lane_count": 1,
                    "capacity_veh_per_hour": 600.0,
                },
                {
                    "edge_id": "GE2",
                    "parent_road_id": "GR2",
                    "from_node": "GN2",
                    "to_node": "GN3",
                    "length_m": 1200.0,
                    "road_class": "local",
                    "speed_limit_mps": 5.0,
                    "lane_count": 1,
                    "capacity_veh_per_hour": 300.0,
                },
            ],
            "requests": [],
            "fleet": [],
            "seeds": {
                "scenario_seed": 1,
                "road_seed": 1,
                "traffic_seed": 1,
                "incident_seed": 1,
                "window_seed": 1,
                "optimizer_seed": 1,
                "learning_seed": 1,
            },
            "generator_version": "test",
            "dataset_split": "test",
            "configuration_version": "test",
        }
    )


def edge_midpoint(scenario: Scenario, edge_id: str) -> tuple[float, float]:
    """WGS84 midpoint of one edge (chord between its two nodes)."""
    nodes = {node.node_id: node for node in scenario.nodes}
    edge = next(item for item in scenario.edges if item.edge_id == edge_id)
    origin = to_latlon(nodes[edge.from_node].x_m, nodes[edge.from_node].y_m)
    destination = to_latlon(nodes[edge.to_node].x_m, nodes[edge.to_node].y_m)
    assert origin is not None and destination is not None
    return (origin[0] + destination[0]) / 2, (origin[1] + destination[1]) / 2


def vertical_segment(lat: float, lon: float, **kwargs) -> FlowSegment:
    """A HERE flow link drawn straight through (lat, lon)."""
    return FlowSegment(
        polylines=(
            ((lat - 0.0005, lon), (lat, lon), (lat + 0.0005, lon)),
        ),
        **kwargs,
    )


# ── key-optional behaviour (no pyproj, no geo map, no network) ────────────


def test_status_without_key_is_honest():
    feed = HereFeed(HEREClient(HEREConfig(api_key=None)))
    status = feed.status()
    assert status["configured"] is False
    assert status["available"] is False
    assert "HERE_API_KEY" in status["reason"]
    assert "portal.here.com" in status["reason"]


def test_missing_key_is_lenient_by_default_and_strict_on_demand():
    feed = HereFeed(HEREClient(HEREConfig(api_key=None)))
    scenario = geo_scenario()

    overrides, meta = feed.edge_speeds(scenario, strict=False)
    assert overrides == {}
    assert meta["available"] is False
    assert meta["applied"] is False
    assert "HERE_API_KEY" in meta["reason"]

    with pytest.raises(HERENotConfigured):
        feed.edge_speeds(scenario, strict=True)

    unchanged, strict_meta = feed.apply(scenario, strict=False)
    assert unchanged is scenario  # nothing overlaid → the same object back
    assert strict_meta["applied"] is False


def test_geocode_without_key_raises_instead_of_faking_results():
    feed = HereFeed(HEREClient(HEREConfig(api_key=None)))
    with pytest.raises(HERENotConfigured):
        feed.geocode("Connaught Place")


def test_synthetic_fixture_never_reaches_the_network(scenario: Scenario):
    """A key exists, but the step-3 fixture has no real geography."""
    client = FakeHEREClient()
    feed = HereFeed(client)

    overrides, meta = feed.edge_speeds(scenario, strict=False)
    assert overrides == {}
    assert meta["available"] is False
    assert "synthetic" in meta["reason"] or "geo" in meta["reason"]
    assert client.flow_calls == 0  # rejected before any request

    with pytest.raises(ValueError, match="synthetic|geo"):
        feed.edge_speeds(scenario, strict=True)


# ── pure matching logic (no pyproj required) ──────────────────────────────


def test_match_picks_nearest_link_and_rejects_far_probes():
    lat, lon = 28.6300, 77.2200
    index = FlowIndex([vertical_segment(lat, lon, speed_mps=4.5, traversability="open")])

    near = match_edge_speeds([EdgeProbe("EDGE_NEAR", lat, lon)], index)
    assert near == {"EDGE_NEAR": 4.5}

    # ~0.01° east ≈ 970 m away — well outside the 150 m acceptance radius.
    far = match_edge_speeds([EdgeProbe("EDGE_FAR", lat, lon + 0.01)], index)
    assert far == {}


def test_match_skips_closed_links_and_enforces_the_speed_floor():
    lat, lon = 28.6300, 77.2200
    closed = FlowIndex(
        [vertical_segment(lat, lon, speed_mps=9.0, traversability="closed")]
    )
    assert match_edge_speeds([EdgeProbe("E1", lat, lon)], closed) == {}

    crawling = FlowIndex(
        [vertical_segment(lat, lon, speed_mps=0.004, traversability="open")]
    )
    assert match_edge_speeds([EdgeProbe("E1", lat, lon)], crawling) == {"E1": 0.1}


def test_empty_index_matches_nothing():
    index = FlowIndex([])
    assert len(index) == 0
    assert index.nearest(28.63, 77.22, 150.0) is None
    assert match_edge_speeds([EdgeProbe("E1", 28.63, 77.22)], index) == {}


# ── full overlay on a geo scenario (needs pyproj) ─────────────────────────


@requires_pyproj
def test_live_overlay_overrides_only_the_matched_edge():
    scenario = geo_scenario()
    mid_lat, mid_lon = edge_midpoint(scenario, "GE1")
    client = FakeHEREClient(
        segments=(vertical_segment(mid_lat, mid_lon, speed_mps=3.5, traversability="open"),)
    )
    feed = HereFeed(client)

    overlaid, meta = feed.apply(scenario, strict=True)

    assert meta["available"] is True
    assert meta["applied"] is True
    assert meta["matched"] == 1
    assert meta["edges"] == 2
    assert meta["source_updated"] == "2026-09-25T09:30:00Z"
    assert meta["fetched_at"] is not None

    speeds = {edge.edge_id: edge.speed_limit_mps for edge in overlaid.edges}
    assert speeds["GE1"] == pytest.approx(3.5)  # HERE live speed (m/s)
    assert speeds["GE2"] == pytest.approx(5.0)  # no link nearby → static

    # Free-flow time follows the live speed, so Dijkstra/QPSO see the jam.
    ge1 = next(edge for edge in overlaid.edges if edge.edge_id == "GE1")
    assert ge1.free_flow_time_s == pytest.approx(300.0 / 3.5)

    # The tag makes graph_version-based caches miss stale speeds.
    assert overlaid.graph_version == f"{scenario.graph_version}:here-live"
    # …and the original scenario object is untouched (frozen contract).
    assert scenario.graph_version == "TEST_GEO:graph:1"
    assert scenario.edges[0].speed_limit_mps == pytest.approx(8.0)


@requires_pyproj
def test_flow_fetch_is_cached_for_the_ttl():
    scenario = geo_scenario()
    mid_lat, mid_lon = edge_midpoint(scenario, "GE1")
    client = FakeHEREClient(
        segments=(vertical_segment(mid_lat, mid_lon, speed_mps=3.5),)
    )
    feed = HereFeed(client)

    first, _ = feed.apply(scenario, strict=True)
    second, _ = feed.apply(scenario, strict=True)
    assert client.flow_calls == 1
    assert first.graph_version == second.graph_version


@requires_pyproj
def test_flow_error_is_a_reason_leniently_and_an_error_strictly():
    scenario = geo_scenario()
    client = FakeHEREClient(flow_error=HEREError("HERE API unreachable (timeout)"))
    feed = HereFeed(client)

    overrides, meta = feed.edge_speeds(scenario, strict=False)
    assert overrides == {}
    assert meta["available"] is False
    assert "unreachable" in meta["reason"]

    with pytest.raises(HEREError, match="unreachable"):
        feed.edge_speeds(scenario, strict=True)
    assert client.flow_calls == 2  # each attempt really tried once


@requires_pyproj
def test_geocode_results_pass_through_unchanged():
    item = GeocodeItem(
        label="Connaught Place, New Delhi, India",
        lat=28.6315,
        lon=77.2167,
        city="New Delhi",
        country="India",
        result_type="address",
        distance_m=120.0,
    )
    feed = HereFeed(FakeHEREClient(items=(item,)))
    payload = feed.geocode("connaught place", limit=5)

    assert payload["count"] == 1
    assert payload["results"][0]["label"] == item.label
    assert payload["results"][0]["lat"] == pytest.approx(28.6315)
    assert payload["source"] == "HERE Geocoding v1"


@requires_pyproj
def test_here_route_between_two_graph_nodes():
    scenario = geo_scenario()
    feed = HereFeed(FakeHEREClient())
    payload = feed.route(scenario, "GN1", "GN3")

    assert payload["duration_s"] == pytest.approx(120.0)
    assert payload["base_duration_s"] == pytest.approx(100.0)
    assert payload["delay_s"] == pytest.approx(20.0)  # traffic delay measured
    assert payload["origin"]["lat"] == pytest.approx(edge_midpoint(scenario, "GE1")[0], abs=1e-3)

    with pytest.raises(ValueError, match="no lat/lon"):
        feed.route(scenario, "GN1", "MISSING_NODE")
