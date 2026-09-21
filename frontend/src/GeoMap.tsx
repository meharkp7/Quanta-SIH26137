import { useMemo } from "react";
import { CircleMarker, LayersControl, MapContainer, Polyline, TileLayer, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import type { Graph } from "@/api";

export type GeoMover = { id: string; lat: number; lon: number; kind?: string };

// Free-flow speed bands, colored slow (red) to fast (green). These come
// from OSM speed limits in the live network profile — not neural output.
export function speedBandColor(speedMps: number, maxSpeed: number): string {
  const ratio = maxSpeed > 0 ? speedMps / maxSpeed : 0;
  if (ratio >= 0.8) return "#22c55e";
  if (ratio >= 0.6) return "#a3e635";
  if (ratio >= 0.4) return "#f59e0b";
  return "#ef4444";
}

export function GeoMap({ graph, routeEdges, closed, movers = [] }: {
  graph: Graph;
  routeEdges: Set<string>;
  closed: string[];
  movers?: GeoMover[];
}) {
  const nodes = useMemo(() => Object.fromEntries(graph.nodes.map((n) => [n.id, n])), [graph]);
  const maxSpeed = useMemo(
    () => Math.max(1, ...graph.edges.map((e) => e.speed_mps)),
    [graph],
  );
  const center = useMemo(() => {
    const pts = graph.nodes.filter((n) => n.lat != null && n.lon != null);
    const lat = pts.reduce((s, n) => s + (n.lat as number), 0) / Math.max(1, pts.length);
    const lon = pts.reduce((s, n) => s + (n.lon as number), 0) / Math.max(1, pts.length);
    return { lat, lon };
  }, [graph]);
  const closedSet = useMemo(() => new Set(closed), [closed]);
  const depots = useMemo(() => {
    const ids = new Set(graph.fleet.map((v) => v.depot));
    return graph.nodes.filter((n) => ids.has(n.id) && n.lat != null);
  }, [graph]);
  const jobs = useMemo(() => {
    const byNode = new Set(graph.requests.map((r) => r.node));
    return graph.nodes.filter((n) => byNode.has(n.id) && n.lat != null);
  }, [graph]);

  return (
    <div className="geo-map-wrap">
      <MapContainer center={[center.lat, center.lon]} zoom={14} scrollWheelZoom className="geo-map">
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="CartoDB Positron">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="OpenStreetMap">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </LayersControl.BaseLayer>
        </LayersControl>
        {graph.edges.map((edge) => {
          const a = nodes[edge.from], b = nodes[edge.to];
          if (!a || !b || a.lat == null || b.lat == null) return null;
          const isRoute = routeEdges.has(edge.id);
          const isClosed = closedSet.has(edge.id) || !edge.open;
          const color = isClosed ? "#ef4444" : isRoute ? "#65d615" : speedBandColor(edge.speed_mps, maxSpeed);
          return (
            <Polyline
              key={edge.id}
              positions={[[a.lat, a.lon as number], [b.lat as number, b.lon as number]]}
              pathOptions={{
                color,
                weight: isRoute ? 5 : isClosed ? 4 : 2,
                opacity: isRoute || isClosed ? 0.95 : 0.55,
                dashArray: isClosed ? "6 5" : undefined,
              }}
            >
              <Tooltip sticky>{`${edge.id} · ${edge.from} → ${edge.to} · ${edge.speed_mps.toFixed(1)} m/s free flow`}</Tooltip>
            </Polyline>
          );
        })}
        {jobs.map((n) => (
          <CircleMarker key={`job-${n.id}`} center={[n.lat as number, n.lon as number]} radius={4} pathOptions={{ color: "#f59e0b", fillColor: "#f59e0b", fillOpacity: 0.9 }}>
            <Tooltip>{`Delivery ${n.id}`}</Tooltip>
          </CircleMarker>
        ))}
        {depots.map((n) => (
          <CircleMarker key={`depot-${n.id}`} center={[n.lat as number, n.lon as number]} radius={9} pathOptions={{ color: "#a3e635", fillColor: "#1a2e12", fillOpacity: 0.9 }}>
            <Tooltip>{`Depot ${n.id}`}</Tooltip>
          </CircleMarker>
        ))}
        {movers.map((m) => (
          <CircleMarker key={m.id} center={[m.lat, m.lon]} radius={m.kind === "delivery" ? 7 : 4} pathOptions={{ color: m.kind === "delivery" ? "#a3e635" : "#9ca3af", fillOpacity: 0.9 }}>
            <Tooltip>{m.id}</Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>
      <div className="map-legend">
        <span><i className="legend-line route" />Selected route</span>
        <span><i className="legend-line closed" />Closure</span>
        <span><i className="legend-dot job" />Delivery</span>
        <span><i className="legend-line geo-fast" />Fast free flow</span>
        <span><i className="legend-line geo-slow" />Slow free flow</span>
      </div>
    </div>
  );
}

export function isGeoGraph(graph: Graph | null): boolean {
  return !!graph?.geo?.available && graph.nodes.length > 0 && graph.nodes[0].lat != null;
}
