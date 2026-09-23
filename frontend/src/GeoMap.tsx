import { useMemo, useState } from "react";
import { CircleMarker, LayersControl, MapContainer, Marker, Polyline, TileLayer, Tooltip } from "react-leaflet";
import { divIcon } from "leaflet";
import "leaflet/dist/leaflet.css";
import type { Graph, ReplayIncident } from "@/api";

export type GeoMover = { id: string; lat: number; lon: number; kind?: string; heading?: number; selected?: boolean };

// Fixture incident schedule used only when the backend reports no schedule
// (S3_BASE always schedules E23@50s in SUMO). Real Delhi maps have no timed
// schedule — the preview then covers UI-selected closures from t=0.
const FIXTURE_DEFAULT_SCHEDULE: ReplayIncident[] = [
  { incident_id: "INC_E23_CLOSURE", edge_id: "E23", trigger_time_s: 50 },
];

const PREVIEW_MAX_T = 400;

// Top-down blue car glyph for the dispatched trip, drawn pointing east
// (0°) — both maps rotate it to the direction of travel. The string form is
// for the Leaflet divIcon, the JSX form for the SVG map; keep them paired.
const CAR_SVG = `<svg width="26" height="16" viewBox="0 0 23 14" xmlns="http://www.w3.org/2000/svg">
  <rect x="4.6" y="0.4" width="5.2" height="2.8" rx="1.2" fill="#0f172a"/>
  <rect x="13.6" y="0.4" width="5.2" height="2.8" rx="1.2" fill="#0f172a"/>
  <rect x="4.6" y="10.8" width="5.2" height="2.8" rx="1.2" fill="#0f172a"/>
  <rect x="13.6" y="10.8" width="5.2" height="2.8" rx="1.2" fill="#0f172a"/>
  <rect x="1" y="2" width="21" height="10" rx="4.6" fill="#38bdf8" stroke="#0369a1" stroke-width="1"/>
  <rect x="13.8" y="3.8" width="5.4" height="6.4" rx="2" fill="#e0f2fe"/>
  <rect x="5.6" y="4.2" width="4.6" height="5.6" rx="1.8" fill="#7dd3fc"/>
  <rect x="20.8" y="3.6" width="1.6" height="2.2" rx="0.7" fill="#fef08a"/>
  <rect x="20.8" y="8.2" width="1.6" height="2.2" rx="0.7" fill="#fef08a"/>
  <rect x="0.9" y="3.8" width="1.4" height="2" rx="0.6" fill="#f87171"/>
  <rect x="0.9" y="8.4" width="1.4" height="2" rx="0.6" fill="#f87171"/>
</svg>`;

// Shared divIcon factory: `iconCls` is the root class Leaflet applies,
// `innerCls` the rotating wrapper inside it (trip marker keeps its glow,
// fleet cars get their own class so the two stay tellable apart).
function carIcon(heading: number | null, iconCls: string, innerCls: string, scale = 1) {
  const rot = heading == null ? 0 : heading.toFixed(1);
  const extra = scale === 1 ? "" : ` scale(${scale})`;
  return divIcon({
    className: iconCls,
    html: `<div class="${innerCls}" style="transform:rotate(${rot}deg)${extra}">${CAR_SVG}</div>`,
    iconSize: [26, 16],
    iconAnchor: [13, 8],
  });
}

// JSX twin of CAR_SVG for the fixture (SVG) map — origin at glyph centre.
export function TripCarGlyph() {
  return (
    <g>
      <rect x="4.6" y="0.4" width="5.2" height="2.8" rx="1.2" fill="#0f172a" />
      <rect x="13.6" y="0.4" width="5.2" height="2.8" rx="1.2" fill="#0f172a" />
      <rect x="4.6" y="10.8" width="5.2" height="2.8" rx="1.2" fill="#0f172a" />
      <rect x="13.6" y="10.8" width="5.2" height="2.8" rx="1.2" fill="#0f172a" />
      <rect x="1" y="2" width="21" height="10" rx="4.6" fill="#38bdf8" stroke="#0369a1" strokeWidth="1" />
      <rect x="13.8" y="3.8" width="5.4" height="6.4" rx="2" fill="#e0f2fe" />
      <rect x="5.6" y="4.2" width="4.6" height="5.6" rx="1.8" fill="#7dd3fc" />
      <rect x="20.8" y="3.6" width="1.6" height="2.2" rx="0.7" fill="#fef08a" />
      <rect x="20.8" y="8.2" width="1.6" height="2.2" rx="0.7" fill="#fef08a" />
      <rect x="0.9" y="3.8" width="1.4" height="2" rx="0.6" fill="#f87171" />
      <rect x="0.9" y="8.4" width="1.4" height="2" rx="0.6" fill="#f87171" />
    </g>
  );
}

// Free-flow speed bands, colored slow (red) to fast (green). These come
// from OSM speed limits in the live network profile — not neural output.
export function speedBandColor(speedMps: number, maxSpeed: number): string {
  const ratio = maxSpeed > 0 ? speedMps / maxSpeed : 0;
  if (ratio >= 0.8) return "#22c55e";
  if (ratio >= 0.6) return "#a3e635";
  if (ratio >= 0.4) return "#f59e0b";
  return "#ef4444";
}

export function GeoMap({ graph, routeEdges, previousRouteEdges, closed, incidents, scenarioId, movers = [], trip, vtrip, selectedVehicle, onPickEdge, onPickVehicle }: {
  graph: Graph;
  routeEdges: Set<string>;
  previousRouteEdges?: Set<string>;
  closed: string[];
  incidents?: ReplayIncident[];
  scenarioId?: string;
  movers?: GeoMover[];
  trip?: { nodeIds: string[]; edgeIds: string[]; pct: number } | null;
  vtrip?: { id: string; nodeIds: string[]; edgeIds: string[]; pct: number } | null;
  selectedVehicle?: string | null;
  onPickEdge?: (edgeId: string) => void;
  onPickVehicle?: (vehicleId: string, fromNode: string) => void;
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
  const edgeById = useMemo(() => new Map(graph.edges.map((e) => [e.id, e])), [graph]);
  // HERE live traffic overlay activates as soon as a key is provided in
  // .env.local (VITE_HERE_API_KEY). Without it the layer stays hidden and
  // the legend shows where to plug the key in.
  const hereKey = String((import.meta.env.VITE_HERE_API_KEY as string | undefined) || "").trim();
  // Dispatched trip geometry: node path → lat/lon polyline, with per-segment
  // lengths taken from the directed-edge lengths so the animated vehicle
  // travels at a distance-accurate pace along the real roads.
  // Shared path geometry: node path → lat/lon polyline, distance-weighted
  // dot, plus heading probes. Used for both the From→To trip marker and a
  // re-dispatched fleet vehicle so they animate identically.
  const buildGeoPath = (path: { nodeIds: string[]; edgeIds: string[]; pct: number }) => {
    if (path.nodeIds.length < 2) return null;
    const pts: [number, number][] = [];
    for (const id of path.nodeIds) {
      const node = nodes[id];
      if (!node || node.lat == null || node.lon == null) return null;
      pts.push([node.lat, node.lon as number]);
    }
    if (pts.length < 2) return null;
    const raw = path.edgeIds.slice(0, pts.length - 1).map((id) => edgeById.get(id)?.length_m ?? 0);
    const lens = raw.length === pts.length - 1 && raw.some((v) => v > 0) ? raw : pts.slice(1).map(() => 1);
    const lerpPt = (a: [number, number], b: [number, number], f: number) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f] as [number, number];
    const dot = interpAlong(pts, lens, path.pct, lerpPt);
    // Heading from a small probe around pct so the car faces the direction
    // of travel. CSS angle: 0° = east, −90° = north (map is north-up).
    const eps = 0.008;
    const a = interpAlong(pts, lens, path.pct <= 0 ? 0 : Math.max(0, path.pct - eps), lerpPt);
    const b = interpAlong(pts, lens, path.pct >= 1 ? 1 : Math.min(1, path.pct + eps), lerpPt);
    const midLat = ((a[0] + b[0]) / 2) * (Math.PI / 180);
    const east = (b[1] - a[1]) * Math.cos(midLat);
    const north = b[0] - a[0];
    const heading = Math.abs(east) + Math.abs(north) > 1e-9 ? (Math.atan2(-north, east) * 180) / Math.PI : null;
    return { pts, dot, heading };
  };
  const tripData = useMemo(() => (trip ? buildGeoPath(trip) : null), [trip, nodes, edgeById]);
  const vtripData = useMemo(() => (vtrip ? buildGeoPath(vtrip) : null), [vtrip, nodes, edgeById]);
  // Icon identity only changes with heading (rounded); position rides on
  // the Marker itself so the DOM churn stays minimal during animation.
  const tripCar = useMemo(() => carIcon(tripData?.heading ?? null, "trip-car-icon", "trip-car"), [Math.round(tripData?.heading ?? 0)]);
  const vtripCar = useMemo(() => carIcon(vtripData?.heading ?? null, "fleet-car-icon", "fleet-car selected"), [Math.round(vtripData?.heading ?? 0)]);
  // Fleet-car divIcons are cached per rounded heading (+selection) so
  // Leaflet only swaps the icon when a car's direction actually changes.
  const fleetIconCache = useMemo(() => new Map<string, ReturnType<typeof carIcon>>(), []);
  const fleetIcon = (heading: number | null, selected: boolean) => {
    const key = `${Math.round(heading ?? 0)}|${selected ? 1 : 0}`;
    let icon = fleetIconCache.get(key);
    if (!icon) {
      icon = carIcon(
        heading,
        selected ? "fleet-car-icon selected" : "fleet-car-icon",
        selected ? "fleet-car selected" : "fleet-car",
        0.9,
      );
      fleetIconCache.set(key, icon);
    }
    return icon;
  };
  // Clicking a fleet car snaps the nearest graph node — the From end for a
  // per-vehicle re-dispatch.
  const nearestNodeId = (lat: number, lon: number) => {
    let best: string | null = null;
    let bestD = Infinity;
    for (const n of graph.nodes) {
      if (n.lat == null || n.lon == null) continue;
      const d = (n.lat - lat) ** 2 + ((n.lon as number) - lon) ** 2;
      if (d < bestD) { bestD = d; best = n.id; }
    }
    return best;
  };

  // Incident timeline preview: scrub 0..400s over the scheduled-closure
  // state. Uses replay incidents when present, else the S3 fixture default;
  // Delhi maps with no schedule preview UI-selected closures from t=0.
  const schedule = useMemo<ReplayIncident[]>(() => {
    if (incidents && incidents.length) return incidents;
    // Fixture demo clock (E23@50s) only while no incident is selected —
    // once the user blocks a road, that road is the sole incident.
    if (!closedSet.size && (scenarioId || graph.scenario_id) === "S3_BASE") return FIXTURE_DEFAULT_SCHEDULE;
    return [];
  }, [incidents, scenarioId, graph.scenario_id, closedSet]);
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
          <LayersControl.BaseLayer checked name="OpenStreetMap">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="CARTO Positron (needs CARTO key since 2025)">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            />
          </LayersControl.BaseLayer>
          {hereKey && (
            <LayersControl.Overlay checked name="HERE live traffic · real-time">
              <TileLayer
                attribution='&copy; <a href="https://www.here.com">HERE</a>'
                url={`https://traffic.maps.hereapi.com/v3/flow/mc/{z}/{x}/{y}/png?apiKey=${encodeURIComponent(hereKey)}&size=256`}
                opacity={0.7}
              />
            </LayersControl.Overlay>
          )}
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
              className="road-hit"
              eventHandlers={{ click: () => onPickEdge?.(edge.id) }}
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
        {tripData && (
          <Polyline positions={tripData.pts} pathOptions={{ color: "#38bdf8", weight: 6, opacity: 0.95, lineCap: "round" }}>
            <Tooltip sticky>Dispatched trip</Tooltip>
          </Polyline>
        )}
        {tripData && trip && (
          <Marker position={tripData.dot} icon={tripCar}>
            <Tooltip direction="top" offset={[0, -10]}>Dispatched vehicle · {Math.round(trip.pct * 100)}%</Tooltip>
          </Marker>
        )}
        {vtripData && vtrip && (
          <>
            <Polyline positions={vtripData.pts} pathOptions={{ color: "#f472b6", weight: 5, opacity: 0.95, dashArray: "8 6", lineCap: "round" }}>
              <Tooltip sticky>{`Vehicle re-route · ${vtrip.id}`}</Tooltip>
            </Polyline>
            <Marker position={vtripData.dot} icon={vtripCar}>
              <Tooltip direction="top" offset={[0, -10]}>{`Vehicle ${vtrip.id} rerouting · ${Math.round(vtrip.pct * 100)}%`}</Tooltip>
            </Marker>
          </>
        )}
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
        {movers
          .filter((m) => Number.isFinite(m.lat) && Number.isFinite(m.lon))
          .map((m) => (
            <Marker
              key={`mover-${m.id}`}
              position={[m.lat, m.lon]}
              icon={fleetIcon(m.heading ?? null, !!m.selected)}
              eventHandlers={{
                click: () => {
                  const from = nearestNodeId(m.lat, m.lon);
                  if (from) onPickVehicle?.(m.id, from);
                },
              }}
            >
              <Tooltip direction="top" offset={[0, -10]}>{m.selected ? `${m.id} · selected` : `${m.id} · click to reroute`}</Tooltip>
            </Marker>
          ))}
      </MapContainer>
      <div className="map-legend">
        <span><i className="legend-line route" />Selected route</span>
        {prevSet.size > 0 && <span><i className="legend-line prev" />Previous route</span>}
        <span><i className="legend-line closed" />Closure</span>
        <span><i className="legend-dot job" />Delivery</span>
        <span className="legend-off">click a road = toggle incident</span>
        {tripData && <span><i className="legend-line trip" />Dispatched trip</span>}
        {vtripData && <span><i className="legend-line vtrip" />Vehicle re-route</span>}
        {hereKey
          ? <span><i className="legend-line geo-traffic" />HERE live traffic</span>
          : <span className="legend-off">HERE layer: add VITE_HERE_API_KEY</span>}
        <span><i className="legend-dot" />Fleet car · click = reroute</span>
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

// Distance-weighted interpolation along a polyline: `lengths[i]` is the
// length of segment points[i] → points[i+1]. Used to animate a dispatched
// vehicle along its planned path on both the geo and SVG maps.
export function interpAlong<T>(points: T[], lengths: number[], pct: number, lerp: (a: T, b: T, f: number) => T): T {
  if (points.length < 2 || !lengths.length) return points[0];
  const clamped = Math.min(1, Math.max(0, pct));
  const total = lengths.reduce((sum, value) => sum + value, 0) || lengths.length;
  let remaining = clamped * total;
  const last = Math.min(lengths.length, points.length - 1) - 1;
  for (let i = 0; i <= last; i += 1) {
    const seg = lengths[i] > 0 ? lengths[i] : total / Math.max(1, lengths.length);
    if (remaining <= seg || i === last) return lerp(points[i], points[i + 1], Math.min(1, remaining / seg));
    remaining -= seg;
  }
  return points[points.length - 1];
}

export function isGeoGraph(graph: Graph | null): boolean {
  return !!graph?.geo?.available && graph.nodes.length > 0 && graph.nodes[0].lat != null;
}
