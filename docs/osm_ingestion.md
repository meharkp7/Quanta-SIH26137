# OpenStreetMap ingestion

Quanta can ingest a **locally acquired and audited** OpenStreetMap road extract through
`src/data/osm_ingestion.py`.

## Design rules

- OSMnx is optional; the core routing stack does not import it at module import time.
- Acquisition is separate from conversion. The ingestion module does not silently call
  Overpass/Nominatim or depend on network availability.
- The graph must be projected to a metric CRS before conversion by default. This prevents
  longitude/latitude degrees from being interpreted as metres.
- OSM edge `length` is treated as metres and is required by default.
- Missing `maxspeed` and `lanes` use explicit configurable defaults; these defaults are
  recorded by the caller and must not be mistaken for measured traffic data.
- Directed OSM edges remain directed Quanta `RoadEdge` objects. A reverse edge is only
  created when it exists in the source graph.
- The converter produces only the road-network layer. Requests, fleet state, traffic
  observations, incidents, forecasts, and optimization decisions remain separate.

## Reproducible deployment path

1. Obtain a licensed/attributed OSM extract for the target study area.
2. Preserve the original extract and its checksum as experiment provenance.
3. Load it with OSMnx and project it to a suitable local metric CRS.
4. Convert with `graph_to_quanta()`.
5. Validate the resulting `RoadNode`/`RoadEdge` contracts before constructing a scenario.
6. Persist the source checksum, CRS, ingestion configuration, and OSM attribution alongside
   the derived benchmark artifact.

The conversion layer intentionally does not claim that OSM geometry is equivalent to live
traffic conditions. Real-time speed, closures, incidents, and demand must enter through the
existing dynamic-state pipeline.
