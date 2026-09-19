
"""
Acquire the 20 Delhi OSM zone graphs used by the Quanta corpus.

Input config schema:
[
  {
    "name": "connaught_place",
    "graphml": "osm_extracts/connaught_place.graphml"
  },
  ...
]

The script:
- skips GraphML files that already exist;
- geocodes each zone to a polygon;
- applies a metric-radius buffer in EPSG:32643;
- downloads a drivable OSMnx graph;
- keeps the largest connected component;
- validates graph size;
- saves GraphML to the configured path;
- continues through the remaining zones when one zone fails.

Mayur Vihar has explicit polygon-geocoding fallbacks because Nominatim
may resolve the parent locality to a non-polygon result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
ox.settings.overpass_url = "https://overpass.kumi.systems/api"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("configs/delhi_zones_20.json")

# Radius around each geocoded zone center/boundary, in metres.
RADIUS_M = 1800

# UTM zone covering Delhi. Buffering in this CRS avoids buffering longitude /
# latitude degrees directly.
METRIC_CRS = 32643
WGS84_CRS = 4326

# Basic quality gates. These prevent accidentally saving tiny/incomplete maps.
MIN_NODES = 100
MIN_EDGES = 150

# OSMnx settings.
NETWORK_TYPE = "drive"
SIMPLIFY = True
RETAIN_ALL = False
TRUNCATE_BY_EDGE = True

# Be polite to Nominatim/OSM.
REQUEST_PAUSE_S = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_zones(config_path: Path) -> list[dict[str, Any]]:
    """Load and validate the project's name/graphml zone configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Zone config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        zones = json.load(f)

    if not isinstance(zones, list) or not zones:
        raise ValueError(
            f"Expected a non-empty JSON list in {config_path}, "
            f"got {type(zones).__name__}"
        )

    for idx, zone in enumerate(zones):
        if not isinstance(zone, dict):
            raise ValueError(
                f"Invalid zone config at index {idx}: {zone!r}"
            )

        name = zone.get("name")
        graphml = zone.get("graphml")

        if not name or not graphml:
            raise ValueError(
                f"Invalid zone config at index {idx}: {zone!r}. "
                "Expected keys: 'name' and 'graphml'."
            )

    return zones


def zone_query_candidates(zone_name: str) -> list[str]:
    """Return geocoding candidates, including known locality fallbacks."""
    display_name = zone_name.replace("_", " ").title()

    queries = [
        f"{display_name}, Delhi, India",
    ]

    # Nominatim can resolve Mayur Vihar itself to a locality/point rather
    # than a polygon. Try its phases before giving up.
    if zone_name == "mayur_vihar":
        queries.extend(
            [
                "Mayur Vihar Phase 1, Delhi, India",
                "Mayur Vihar Phase 2, Delhi, India",
                "Mayur Vihar Phase 3, Delhi, India",
            ]
        )

    if zone_name == "greater_kailash":
        queries.extend([
            "Greater Kailash I, Delhi, India",
            "Greater Kailash II, Delhi, India",
        ])

    if zone_name == "laxmi_nagar":
        queries.extend([
            "Laxmi Nagar, East Delhi, India",
            "Laxmi Nagar, New Delhi, India",
            "Laxmi Nagar District Centre, Delhi, India",
        ])

    return queries


def geocode_zone_polygon(
    zone_name: str,
) -> tuple[Any, str]:
    """
    Resolve a zone to a Polygon/MultiPolygon.

    Returns:
        (polygon, resolved_query)
    """
    candidates = zone_query_candidates(zone_name)
    last_error: Exception | None = None

    for query in candidates:
        print(f"    geocoding: {query}", flush=True)

        try:
            gdf = ox.geocode_to_gdf(query)

            if gdf.empty:
                print("    -> no geocoding result", flush=True)
                continue

            geom = gdf.geometry.iloc[0]

            if geom is None or geom.is_empty:
                print("    -> empty geometry", flush=True)
                continue

            if geom.geom_type not in {"Polygon", "MultiPolygon"}:
                print(
                    f"    -> skipped non-polygon geometry: {geom.geom_type}",
                    flush=True,
                )
                continue

            print(
                f"    -> polygon resolved using {query!r}",
                flush=True,
            )
            return geom, query

        except (TypeError, ValueError, RuntimeError) as exc:
            last_error = exc
            print(f"    -> geocoding failed: {exc}", flush=True)

        time.sleep(REQUEST_PAUSE_S)

    raise RuntimeError(
        f"Could not obtain a Polygon/MultiPolygon for {zone_name!r}. "
        f"Tried: {candidates}. Last error: {last_error}"
    )


def buffered_zone_polygon(polygon: Any) -> Any:
    """
    Buffer a WGS84 polygon by RADIUS_M using a metric CRS.

    The input is a Shapely Polygon/MultiPolygon, not a GeoSeries, so it is
    wrapped in a one-element GeoSeries before CRS transformation.
    """
    polygon_gs = gpd.GeoSeries([polygon], crs=WGS84_CRS)

    buffered = (
        polygon_gs
        .to_crs(METRIC_CRS)
        .buffer(RADIUS_M)
        .to_crs(WGS84_CRS)
        .iloc[0]
    )

    return buffered


def acquire_graph(zone_name: str) -> Any:
    """Download and construct a drivable OSM graph for a zone."""
    polygon, resolved_query = geocode_zone_polygon(zone_name)

    print(
        f"    buffering resolved polygon by {RADIUS_M} m "
        f"(resolved from {resolved_query!r})",
        flush=True,
    )

    polygon = buffered_zone_polygon(polygon)

    time.sleep(REQUEST_PAUSE_S)

    print(
        f"    downloading OSM graph: network_type={NETWORK_TYPE!r}",
        flush=True,
    )

    graph = ox.graph_from_polygon(
        polygon,
        network_type=NETWORK_TYPE,
        simplify=SIMPLIFY,
        retain_all=RETAIN_ALL,
        truncate_by_edge=TRUNCATE_BY_EDGE,
    )

    return graph


def validate_graph(graph: Any, zone_name: str) -> None:
    """Apply basic graph-size and structural quality gates."""
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)

    print(
        f"    graph size: {node_count:,} nodes, {edge_count:,} edges",
        flush=True,
    )

    if node_count < MIN_NODES:
        raise ValueError(
            f"{zone_name}: graph has only {node_count} nodes; "
            f"minimum is {MIN_NODES}"
        )

    if edge_count < MIN_EDGES:
        raise ValueError(
            f"{zone_name}: graph has only {edge_count} edges; "
            f"minimum is {MIN_EDGES}"
        )

    if node_count == 0 or edge_count == 0:
        raise ValueError(f"{zone_name}: graph is empty")


def save_graph(graph: Any, output_path: Path) -> None:
    """Persist a GraphML graph, creating its parent directory if necessary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # GraphML output is deterministic enough for our acquisition artifact;
    # overwrite only happens when the caller explicitly removed/replaced it.
    ox.save_graphml(graph, filepath=output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    zones = load_zones(CONFIG_PATH)

    print("=" * 72)
    print("Quanta — Delhi OSM Zone Acquisition")
    print("=" * 72)
    print(f"config       : {CONFIG_PATH}")
    print(f"zones        : {len(zones)}")
    print(f"radius       : {RADIUS_M} m")
    print(f"metric CRS   : EPSG:{METRIC_CRS}")
    print(f"network type : {NETWORK_TYPE}")
    print("=" * 72)

    existing = sum(
        1
        for zone in zones
        if Path(str(zone["graphml"])).exists()
    )

    print(
        f"Existing GraphML files: {existing}/{len(zones)}",
        flush=True,
    )
    print()

    saved = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for idx, zone in enumerate(zones, start=1):
        zone_name = str(zone["name"])
        output_path = Path(str(zone["graphml"]))

        print(
            f"[{idx}/{len(zones)}] {zone_name}",
            flush=True,
        )
        print(f"    output: {output_path}", flush=True)

        if output_path.exists():
            print("    -> already exists; skipping", flush=True)
            skipped += 1
            print()
            continue

        try:
            graph = acquire_graph(zone_name)
            validate_graph(graph, zone_name)
            save_graph(graph, output_path)

            node_count = len(graph.nodes)
            edge_count = len(graph.edges)

            print(
                f"    -> SAVED {zone_name}: "
                f"{node_count:,} nodes, {edge_count:,} edges",
                flush=True,
            )
            saved += 1

        except Exception as exc:
            failed.append((zone_name, str(exc)))
            print(
                f"    -> FAILED {zone_name}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            print(
                "    -> continuing with the next zone",
                flush=True,
            )

        print()
        time.sleep(REQUEST_PAUSE_S)

    print("=" * 72)
    print("ACQUISITION SUMMARY")
    print("=" * 72)
    print(f"Configured zones : {len(zones)}")
    print(f"Already existed  : {skipped}")
    print(f"Downloaded now   : {saved}")
    print(f"Failed           : {len(failed)}")

    if failed:
        print()
        print("Failed zones:")
        for zone_name, error in failed:
            print(f"  - {zone_name}: {error}")

    successful_paths = [
        Path(str(zone["graphml"]))
        for zone in zones
        if Path(str(zone["graphml"])).exists()
    ]

    print()
    print(
        f"GraphML coverage: "
        f"{len(successful_paths)}/{len(zones)}"
    )

    if len(successful_paths) == len(zones):
        print("STATUS: PASS — all configured Delhi maps are available.")
    else:
        print(
            "STATUS: INCOMPLETE — rerun this script after fixing "
            "the failed zone(s). Existing maps will be skipped."
        )


if __name__ == "__main__":
    main()
