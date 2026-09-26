import { useEffect, useMemo, useRef, useState } from "react";
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip } from "react-leaflet";
import { divIcon } from "leaflet";
import "leaflet/dist/leaflet.css";
import type { Graph } from "@/api";

export type RoutePath = { id: string; color: string; edgeIds: string[] };

type Pt = [number, number]; // [lat, lon] for geo maps, [y, x] handled separately

// Top-down car glyph for moving vehicles (yellow, like the image's legend).
const CAR_SVG = `<svg width="22" height="13" viewBox="0 0 23 14" xmlns="http://www.w3.org/2000/svg">
  <rect x="4.6" y="0.4" width="5.2" height="2.8" rx="1.2" fill="#0f172a"/>
  <rect x="13.6" y="0.4" width="5.2" height="2.8" rx="1.2" fill="#0f172a"/>
  <rect x="4.6" y="10.8" width="5.2" height="2.8" rx="1.2" fill="#0f172a"/>
  <rect x="13.6" y="10.8" width="5.2" height="2.8" rx="1.2" fill="#0f172a"/>
  <rect x="1" y="2" width="21" height="10" rx="4.6" fill="#facc15" stroke="#854d0e" stroke-width="1"/>
  <rect x="13.8" y="3.8" width="5.4" height="6.4" rx="2" fill="#0f172a" opacity="0.75"/>
  <rect x="5.6" y="4.2" width="4.6" height="5.6" rx="1.8" fill="#0f172a" opacity="0.55"/>
</svg>`;

function carIcon(heading: number | null) {
  const rot = heading == null ? 0 : heading.toFixed(1);
  return divIcon({
    className: "q-car-icon",
    html: `<div class="q-car" style="transform:rotate(${rot}deg)">${CAR_SVG}</div>`,
    iconSize: [22, 13],
    iconAnchor: [11, 6],
  });
}

// edge ids → consecutive node ids (replay geometry for one vehicle).
export function nodesFromEdges(graph: Graph, edgeIds: string[]): string[] {
  const byId = new Map(graph.edges.map((e) => [e.id, e]));
  const path: string[] = [];
  for (const id of edgeIds) {
    const edge = byId.get(id);
    if (!edge) continue;
    if (!path.length) path.push(edge.from);
    else if (path[path.length - 1] !== edge.from) path.push(edge.from);
    path.push(edge.to);
  }
  return path;
}

// ── Vehicle animation pacing ──────────────────────────────────────────────
// One full lap takes `route_km × LAP_SECONDS_PER_KM` seconds, floored at
// MIN_LAP_SECONDS so a short hop is not a blur. Raise the multiplier to slow
// the fleet down further. (The old formula was 60 ÷ km — a 10 km route looped
// in ~6 s, i.e. the longer the route the faster the car.)
const LAP_SECONDS_PER_KM = 3;
const MIN_LAP_SECONDS = 15;

// Distance-weighted position along a polyline at progress p (0..1).
function positionAt(points: Pt[], progress: number, scaleLon: number): { pos: Pt; heading: number } | null {
  if (points.length < 2) return null;
  const segLens: number[] = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    const dy = points[i + 1][0] - points[i][0];
    const dx = (points[i + 1][1] - points[i][1]) * scaleLon;
    segLens.push(Math.hypot(dx, dy));
  }
  const total = segLens.reduce((sum, v) => sum + v, 0) || 1;
  let remaining = Math.min(1, Math.max(0, progress)) * total;
  for (let i = 0; i < segLens.length; i += 1) {
    if (remaining <= segLens[i] || i === segLens.length - 1) {
      const f = segLens[i] > 0 ? Math.min(1, remaining / segLens[i]) : 0;
      const a = points[i];
      const b = points[i + 1];
      const dy = b[0] - a[0];
      const dx = (b[1] - a[1]) * scaleLon;
      const heading = Math.abs(dx) + Math.abs(dy) > 1e-12 ? (Math.atan2(-dy, dx) * 180) / Math.PI : 0;
      return { pos: [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f], heading };
    }
    remaining -= segLens[i];
  }
  return null;
}

function useAnimatedClock(active: boolean, reduced?: boolean) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!active || reduced) return;
    const started = performance.now();
    const id = window.setInterval(() => setSeconds((performance.now() - started) / 1000), 90);
    return () => window.clearInterval(id);
  }, [active, reduced]);
  return seconds;
}

// ── Map panel: real OSM tiles on geo maps, SVG network on the city fixture ──
export function MapPanel({
  graph,
  routes = [],
  closed = [],
  animate = true,
  height = 340,
  edgeColors,
  caption,
  extraLegend,
}: {
  graph: Graph | null;
  routes?: RoutePath[];
  closed?: string[];
  animate?: boolean;
  height?: number;
  edgeColors?: Map<string, string>;
  caption?: string;
  extraLegend?: React.ReactNode;
}) {
  const reduced = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  const clock = useAnimatedClock(animate && routes.length > 0, reduced);
  const nodes = useMemo(() => new Map((graph?.nodes || []).map((n) => [n.id, n])), [graph]);
  const closedSet = useMemo(() => new Set(closed), [closed]);
  const routeEdgeSet = useMemo(() => new Set(routes.flatMap((r) => r.edgeIds)), [routes]);

  // Route geometry: node path per vehicle → polyline + animation cursor.
  const geometry = useMemo(() => {
    if (!graph) return [];
    return routes.map((route, index) => {
      const pathIds = nodesFromEdges(graph, route.edgeIds);
      const pts: Pt[] = pathIds.map((id) => {
        const node = nodes.get(id);
        if (!node) return [0, 0];
        return node.lat != null && node.lon != null ? [node.lat, node.lon] : [node.y, node.x];
      });
      let length = 0;
      for (let i = 0; i < pathIds.length - 1; i += 1) {
        const edge = graph.edges.find((e) => e.from === pathIds[i] && e.to === pathIds[i + 1]);
        length += edge?.length_m || 100;
      }
      return {
        route,
        pts,
        length,
        offset: (index * 0.17) % 1,
        // Lap pacing: one full lap ≈ LAP_SECONDS_PER_KM seconds per km of route
        // (a 10 km route loops in ~50 s), so markers drift calmly instead of
        // sprinting. Raise the constant to slow the fleet down further.
        speed: 1 / Math.max(MIN_LAP_SECONDS, (length / 1000) * LAP_SECONDS_PER_KM),
        label: `Route ${index + 1}`,
        color: route.color,
      };
    });
  }, [graph, routes, nodes]);

  if (!graph) return <div className="q-map" style={{ height }}><span className="q-map-loading">Loading network…</span></div>;

  const depots = new Set(graph.fleet.map((v) => v.depot));
  const customers = new Set(graph.requests.map((r) => r.node));

  const legend = (
    <div className="q-map-legend">
      <span><i className="q-lg depot" />Depot</span>
      <span><i className="q-lg customer" />Customer</span>
      {routes.length > 0 && <span><i className="q-lg vehicle" />Vehicle (moving)</span>}
      {routes.slice(0, 4).map((route, i) => (
        <span key={route.id}><i className="q-lg route" style={{ background: route.color }} />{`Route ${i + 1}`}</span>
      ))}
      {routes.length > 4 && <span className="q-lg-more">+{routes.length - 4} more</span>}
      {closedSet.size > 0 && <span><i className="q-lg closed" />Road closed</span>}
      {extraLegend}
    </div>
  );

  const captionNode = caption ? <div className="q-map-caption">{caption}</div> : null;

  // ── Real OSM map ────────────────────────────────────────────────────────
  if (graph.geo?.available && graph.nodes.some((n) => n.lat != null)) {
    const geoNodes = graph.nodes.filter((n) => n.lat != null && n.lon != null);
    const center = {
      lat: geoNodes.reduce((s, n) => s + (n.lat as number), 0) / Math.max(1, geoNodes.length),
      lon: geoNodes.reduce((s, n) => s + (n.lon as number), 0) / Math.max(1, geoNodes.length),
    };
    const latSpan = Math.max(...geoNodes.map((n) => n.lat as number)) - Math.min(...geoNodes.map((n) => n.lat as number));
    const lonSpan = Math.max(...geoNodes.map((n) => n.lon as number)) - Math.min(...geoNodes.map((n) => n.lon as number));
    const spanM = Math.max(
      400,
      Math.max(latSpan * 111320, lonSpan * 111320 * Math.cos((center.lat * Math.PI) / 180)),
    );
    // Web-Mercator zoom that frames the whole network in a ~720px panel.
    const zoom = Math.max(11, Math.min(16, Math.round(Math.log2((156543.03392 * Math.cos((center.lat * Math.PI) / 180) * 720) / spanM))));
    const scaleLon = Math.cos((center.lat * Math.PI) / 180);
    return (
      <div className="q-map" style={{ height }}>
        <MapContainer center={[center.lat, center.lon]} zoom={zoom} scrollWheelZoom className="q-leaflet">
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {graph.edges.map((edge) => {
            const a = nodes.get(edge.from);
            const b = nodes.get(edge.to);
            if (!a || !b || a.lat == null || b.lat == null) return null;
            const custom = edgeColors?.get(edge.id);
            const isClosed = closedSet.has(edge.id) || !edge.open;
            const isRoute = routeEdgeSet.has(edge.id);
            const color = custom || (isClosed ? "#ef4444" : isRoute ? "#22c55e" : "#94a3b8");
            return (
              <Polyline
                key={edge.id}
                positions={[[a.lat, a.lon as number], [b.lat as number, b.lon as number]]}
                pathOptions={{
                  color,
                  weight: custom ? 3.4 : isRoute ? 5 : isClosed ? 4 : 1.6,
                  opacity: custom ? 0.95 : isRoute || isClosed ? 0.95 : 0.55,
                  dashArray: isClosed ? "6 5" : undefined,
                }}
              />
            );
          })}
          {geometry.map((g, i) => {
            if (g.pts.length < 2) return null;
            return <Polyline key={`route-${g.route.id}-${i}`} positions={g.pts} pathOptions={{ color: g.color, weight: 5, opacity: 0.9, lineCap: "round" }} />;
          })}
          {graph.nodes.filter((n) => customers.has(n.id) && n.lat != null).map((n) => (
            <CircleMarker key={`c-${n.id}`} center={[n.lat as number, n.lon as number]} radius={4.5} pathOptions={{ color: "#38bdf8", fillColor: "#38bdf8", fillOpacity: 0.95 }}>
              <Tooltip>Customer · {n.id.slice(-8)}</Tooltip>
            </CircleMarker>
          ))}
          {graph.nodes.filter((n) => depots.has(n.id) && n.lat != null).map((n) => (
            <CircleMarker key={`d-${n.id}`} center={[n.lat as number, n.lon as number]} radius={8} pathOptions={{ color: "#22c55e", fillColor: "#14532d", fillOpacity: 0.95, weight: 2 }}>
              <Tooltip>Depot · {n.id.slice(-8)}</Tooltip>
            </CircleMarker>
          ))}
          {geometry.map((g, i) => {
            if (!g.pts.length || reduced) return null;
            const hit = positionAt(g.pts, (clock * g.speed + g.offset) % 1, scaleLon);
            if (!hit) return null;
            return (
              <Marker key={`car-${g.route.id}-${i}`} position={hit.pos} icon={carIcon(hit.heading)}>
                <Tooltip direction="top" offset={[0, -8]}>{`${g.route.id} · ${g.label}`}</Tooltip>
              </Marker>
            );
          })}
        </MapContainer>
        {legend}
        {captionNode}
      </div>
    );
  }

  // ── Fixture network (no coordinates) → SVG map ──────────────────────────
  return <SvgNetwork graph={graph} geometry={geometry} closed={closedSet} edgeColors={edgeColors} height={height} legend={legend} caption={captionNode} reduced={!!reduced} />;
}

function SvgNetwork({ graph, geometry, closed, edgeColors, height, legend, caption, reduced }: {
  graph: Graph;
  geometry: { route: RoutePath; pts: Pt[]; offset: number; speed: number; color: string }[];
  closed: Set<string>;
  edgeColors?: Map<string, string>;
  height: number;
  legend: React.ReactNode;
  caption: React.ReactNode;
  reduced: boolean;
}) {
  const [zoom, setZoom] = useState(1);
  const clock = useAnimatedClock(geometry.length > 0 && !reduced, reduced);
  const bounds = useMemo(() => {
    const xs = graph.nodes.map((n) => n.x);
    const ys = graph.nodes.map((n) => n.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  }, [graph]);
  const W = 800;
  const H = 420;
  const pad = 34;
  const spanX = Math.max(1, bounds.maxX - bounds.minX);
  const spanY = Math.max(1, bounds.maxY - bounds.minY);
  const point = (id: string) => {
    const node = graph.nodes.find((n) => n.id === id);
    if (!node) return { x: pad, y: pad };
    return {
      x: pad + ((node.x - bounds.minX) / spanX) * (W - pad * 2),
      y: H - pad - ((node.y - bounds.minY) / spanY) * (H - pad * 2),
    };
  };
  const depots = new Set(graph.fleet.map((v) => v.depot));
  const customers = new Set(graph.requests.map((r) => r.node));
  const routeEdgeSet = new Set(geometry.flatMap((g) => g.route.edgeIds));
  const [carPositions, setCarPositions] = useState<{ id: string; x: number; y: number; heading: number; color: string }[]>([]);
  const rafRef = useRef(0);
  useEffect(() => {
    if (!geometry.length || reduced) return;
    const tick = () => {
      const now = performance.now() / 1000;
      const next = geometry.map((g, index) => {
        const pathIds = nodesFromEdges(graph, g.route.edgeIds);
        const pts: Pt[] = pathIds.map((id) => {
          const p = point(id);
          return [p.y / H, p.x / W] as Pt;
        });
        const hit = positionAt(pts, (now * g.speed + g.offset) % 1, 1);
        if (!hit) return null;
        return { id: g.route.id, x: hit.pos[1] * W, y: hit.pos[0] * H, heading: hit.heading, color: geometry[index]?.color || "#facc15" };
      }).filter(Boolean) as { id: string; x: number; y: number; heading: number; color: string }[];
      setCarPositions(next);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometry, reduced]);

  return (
    <div className="q-map svg" style={{ height }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Directed city road network" style={{ transform: `scale(${zoom})` }}>
        {graph.edges.map((edge) => {
          const a = point(edge.from);
          const b = point(edge.to);
          const isClosed = closed.has(edge.id) || !edge.open;
          const isRoute = routeEdgeSet.has(edge.id);
          const custom = edgeColors?.get(edge.id);
          let color = custom || (isClosed ? "#ef4444" : isRoute ? "#22c55e" : "#475569");
          if (!custom && isRoute) {
            const owner = geometry.find((g) => g.route.edgeIds.includes(edge.id));
            if (owner) color = owner.color;
          }
          return (
            <line
              key={edge.id}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={color}
              strokeWidth={custom ? 3.2 : isRoute ? 4 : 1.6}
              opacity={custom ? 0.95 : isRoute ? 0.95 : 0.5}
              strokeDasharray={isClosed ? "6 5" : undefined}
            />
          );
        })}
        {graph.nodes.map((node) => {
          const p = point(node.id);
          const isDepot = depots.has(node.id);
          const isCustomer = customers.has(node.id);
          if (!isDepot && !isCustomer) return null;
          return (
            <g key={node.id}>
              <circle cx={p.x} cy={p.y} r={isDepot ? 8 : 5} fill={isDepot ? "#22c55e" : "#38bdf8"} stroke="#04121c" strokeWidth="2" />
              <title>{isDepot ? "Depot" : "Customer"} · {node.id}</title>
            </g>
          );
        })}
        {carPositions.map((car) => (
          <g key={`car-${car.id}`} transform={`translate(${car.x.toFixed(1)} ${car.y.toFixed(1)}) rotate(${car.heading.toFixed(1)})`}>
            <rect x={-8} y={-4.5} width={16} height={9} rx={4} fill="#facc15" stroke="#854d0e" strokeWidth="1.4" />
            <title>{car.id}</title>
          </g>
        ))}
      </svg>
      <button type="button" className="q-map-zoom in" aria-label="Zoom in" onClick={() => setZoom((z) => Math.min(2.2, z + 0.25))}>+</button>
      <button type="button" className="q-map-zoom out" aria-label="Zoom out" onClick={() => setZoom((z) => Math.max(1, z - 0.25))}>−</button>
      {legend}
      {caption}
      {geometry.length > 0 && (
        <span className="q-map-note">{geometry.length} {geometry.length === 1 ? "route" : "routes"} drawn from the optimizer plan</span>
      )}
    </div>
  );
}

// Palette for per-vehicle routes (image: Route 1–4 colour chips).
export const ROUTE_COLORS = ["#22c55e", "#38bdf8", "#f59e0b", "#f472b6", "#a78bfa", "#2dd4bf", "#f97316", "#e879f9"];

// Routes from a solve result: each vehicle's driven edge ids → coloured path.
export function routesFromSolve(solve: { evaluation?: { vehicles: { id: string; edge_ids: string[] }[] } } | null): RoutePath[] {
  if (!solve?.evaluation?.vehicles) return [];
  return solve.evaluation.vehicles
    .filter((vehicle) => vehicle.edge_ids?.length)
    .map((vehicle, index) => ({
      id: vehicle.id,
      color: ROUTE_COLORS[index % ROUTE_COLORS.length],
      edgeIds: vehicle.edge_ids,
    }));
}
