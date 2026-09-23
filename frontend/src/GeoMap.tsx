import { useMemo, useState } from "react";
import { CircleMarker, LayersControl, MapContainer, Polyline, TileLayer, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import type { Graph, ReplayIncident } from "@/api";

export type GeoMover = { id: string; lat: number; lon: number; kind?: string };

// Fixture incident schedule used only when the backend reports no schedule
// (S3_BASE always schedules E23@50s in SUMO). Real Delhi maps have no timed
// schedule — the preview then covers UI-selected closures from t=0.
const FIXTURE_DEFAULT_SCHEDULE: ReplayIncident[] = [
  { incident_id: "INC_E23_CLOSURE", edge_id: "E23", trigger_time_s: 50 },
];

const PREVIEW_MAX_T = 400;

// Free-flow speed bands, colored slow (red) to fast (green). These come
// from OSM speed limits in the live network profile — not neural output.
export function speedBandColor(speedMps: number, maxSpeed: number): string {
  const ratio = maxSpeed > 0 ? speedMps / maxSpeed : 0;
  if (ratio >= 0.8) return "#22c55e";
  if (ratio >= 0.6) return "#a3e635";
  if (ratio >= 0.4) return "#f59e0b";
  return "#ef4444";
}

export function GeoMap({ graph, routeEdges, previousRouteEdges, closed, incidents, scenarioId }: {
  graph: Graph;
  routeEdges: Set<string>;
  previousRouteEdges?: Set<string>;
  closed: string[];
  incidents?: ReplayIncident[];
  scenarioId?: string;
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
  const prevSet = useMemo(() => new Set(previousRouteEdges || []), [previousRouteEdges]);
  const depots = useMemo(() => {
    const ids = new Set(graph.fleet.map((v) => v.depot));
    return graph.nodes.filter((n) => ids.has(n.id) && n.lat != null);
  }, [graph]);
  const jobs = useMemo(() => {
    const byNode = new Set(graph.requests.map((r) => r.node));
    return graph.nodes.filter((n) => byNode.has(n.id) && n.lat != null);
  }, [graph]);

  // Incident timeline preview: scrub 0..400s over the scheduled-closure
  // state. Uses replay incidents when present, else the S3 fixture default;
  // Delhi maps with no schedule preview UI-selected closures from t=0.
  const schedule = useMemo<ReplayIncident[]>(() => {
    if (incidents && incidents.length) return incidents;
    if ((scenarioId || graph.scenario_id) === "S3_BASE") return FIXTURE_DEFAULT_SCHEDULE;
    return [];
  }, [incidents, scenarioId, graph.scenario_id]);
  const [previewT, setPreviewT] = useState<number | null>(null);
  const previewActive = previewT != null;
  const previewClosed = useMemo(() => {
    if (previewT == null) return null;
    const active = new Set<string>();
    for (const id of closedSet) active.add(id);
    for (const inc of schedule) {
      if (previewT >= inc.trigger_time_s) active.add(inc.edge_id);
    }
    return active;
  }, [previewT, closedSet, schedule]);
  const upcoming = useMemo(
    () => (previewT == null ? [] : schedule.filter((inc) => previewT < inc.trigger_time_s)),
    [previewT, schedule],
  );

  // Zone KPI strip, computed frontend-side from the live graph payload
  // (nodes carry zone). Single-zone networks (e.g. Delhi `osm`) fall back
  // to a road_class breakdown so the strip stays informative.
  const zoneKpis = useMemo(() => {
    const nodeZone = new Map(graph.nodes.map((n) => [n.id, n.zone || "?"]));
    const groups = new Map<string, { edges: number; speedSum: number; closed: number }>();
    const byClass = new Map<string, { edges: number; speedSum: number; closed: number }>();
    for (const edge of graph.edges) {
      const zone = nodeZone.get(edge.from) || "?";
      const entry = groups.get(zone) || { edges: 0, speedSum: 0, closed: 0 };
      entry.edges += 1;
      entry.speedSum += edge.speed_mps;
      if (closedSet.has(edge.id) || !edge.open) entry.closed += 1;
      groups.set(zone, entry);
      const cls = byClass.get(edge.road_class) || { edges: 0, speedSum: 0, closed: 0 };
      cls.edges += 1;
      cls.speedSum += edge.speed_mps;
      if (closedSet.has(edge.id) || !edge.open) cls.closed += 1;
      byClass.set(edge.road_class, cls);
    }
    const rows = [...groups.entries()].map(([zone, v]) => ({
      label: zone,
      edges: v.edges,
      meanSpeed: v.speedSum / Math.max(1, v.edges),
      closed: v.closed,
    }));
    const classRows = [...byClass.entries()].map(([label, v]) => ({
      label,
      edges: v.edges,
      meanSpeed: v.speedSum / Math.max(1, v.edges),
      closed: v.closed,
    }));
    return { rows, classRows, singleZone: groups.size <= 1 };
  }, [graph, closedSet]);
  const [kpisOpen, setKpisOpen] = useState(false);

  const edgeColor = (edgeId: string, speed: number, isClosed: boolean, isRoute: boolean) =>
    isClosed ? "#ef4444" : isRoute ? "#65d615" : speedBandColor(speed, maxSpeed);

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
          const isClosed = (previewClosed ? previewClosed.has(edge.id) : closedSet.has(edge.id)) || !edge.open;
          const color = edgeColor(edge.id, edge.speed_mps, isClosed, isRoute);
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
        {[...prevSet].filter((id) => !routeEdges.has(id)).map((id) => {
          const edge = graph.edges.find((e) => e.id === id);
          if (!edge) return null;
          const a = nodes[edge.from], b = nodes[edge.to];
          if (!a || !b || a.lat == null || b.lat == null) return null;
          return (
            <Polyline
              key={`prev-${id}`}
              positions={[[a.lat, a.lon as number], [b.lat as number, b.lon as number]]}
              pathOptions={{ color: "#22d3ee", weight: 3, opacity: 0.85, dashArray: "4 6" }}
            >
              <Tooltip sticky>{`Previous route · ${id}`}</Tooltip>
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
      </MapContainer>
      <div className="map-legend">
        <span><i className="legend-line route" />Selected route</span>
        {prevSet.size > 0 && <span><i className="legend-line prev" />Previous route</span>}
        <span><i className="legend-line closed" />Closure</span>
        <span><i className="legend-dot job" />Delivery</span>
        <span><i className="legend-line geo-fast" />Fast free flow</span>
        <span><i className="legend-line geo-slow" />Slow free flow</span>
      </div>
      <div className="geo-preview">
        <div className="geo-preview-head">
          <span>Incident timeline preview</span>
          {previewActive && <button onClick={() => setPreviewT(null)}>Reset to live</button>}
        </div>
        {schedule.length === 0 && (
          <p className="small-copy">no scheduled incidents — select closures to preview{closedSet.size > 0 ? ` (${closedSet.size} selected, active from t=0)` : ""}</p>
        )}
        {schedule.length > 0 && (
          <>
            <div className="geo-preview-bar">
              <span>t={previewT ?? PREVIEW_MAX_T}s</span>
              <input
                aria-label="Incident schedule preview timeline"
                type="range"
                min={0}
                max={PREVIEW_MAX_T}
                value={previewT ?? PREVIEW_MAX_T}
                onChange={(e) => setPreviewT(Number(e.target.value))}
              />
              <span>{PREVIEW_MAX_T}s</span>
            </div>
            <p className="small-copy">
              {previewClosed ? `${previewClosed.size} closed at t=${previewT}s` : "live closure state"}
              {upcoming.length > 0 && ` · upcoming: ${upcoming.map((u) => `${u.edge_id}@${u.trigger_time_s.toFixed(0)}s`).join(", ")}`}
            </p>
          </>
        )}
      </div>
      <div className="geo-kpis">
        <button className="geo-kpis-toggle" onClick={() => setKpisOpen((v) => !v)} aria-expanded={kpisOpen}>
          {kpisOpen ? "▾" : "▸"} Zone KPIs · {graph.edges.length} roads
        </button>
        {kpisOpen && (
          <table className="geo-kpis-table">
            <thead><tr><th>{zoneKpis.singleZone ? "Road class" : "Zone"}</th><th>Edges</th><th>Mean free-flow</th><th>Closed</th></tr></thead>
            <tbody>
              {(zoneKpis.singleZone ? zoneKpis.classRows : zoneKpis.rows).map((row) => (
                <tr key={row.label}>
                  <td>{row.label}</td>
                  <td>{row.edges}</td>
                  <td>{row.meanSpeed.toFixed(1)} m/s</td>
                  <td>{row.closed}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

export function isGeoGraph(graph: Graph | null): boolean {
  return !!graph?.geo?.available && graph.nodes.length > 0 && graph.nodes[0].lat != null;
}
