import { useMemo, useState } from "react";
import type { ReactNode, ButtonHTMLAttributes } from "react";
import {
  ArrowUpRight,
  Check,
  ChevronRight,
  Circle,
  Layers,
  MapPin,
  Minus,
  Plus,
  Route,
  Truck,
} from "lucide-react";
import { GeoMap, isGeoGraph } from "../GeoMap";
import type { EpisodeData } from "../GeoMap";
import type { Graph, ReplayFrame } from "../api";
import type { Result } from "./data";

export const colors = [
  "#4361ee",
  "#109786",
  "#ec9750",
  "#b269cc",
  "#de6579",
  "#3a94bd",
];
export function Brand({ light = false }: { light?: boolean }) {
  return (
    <span className={`w-brand ${light ? "light" : ""}`}>
      <span className="w-brand-icon">
        <Route size={21} strokeWidth={2.4} />
      </span>
      quanta<span className="w-brand-dot">.</span>
    </span>
  );
}
export function Button({
  children,
  variant = "primary",
  className = "",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "danger";
}) {
  return (
    <button className={`w-button ${variant} ${className}`} {...props}>
      {children}
    </button>
  );
}
export function Tag({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "green" | "amber" | "blue" | "red";
}) {
  return <span className={`w-tag ${tone}`}>{children}</span>;
}
export function Panel({
  title,
  subtitle,
  action,
  children,
  className = "",
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`w-panel ${className}`}>
      {title && (
        <div className="w-panel-head">
          <div>
            <h2>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}
export function PageTitle({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow?: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="w-page-title">
      <div>
        {eyebrow && <div className="w-eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action && <div className="w-title-actions">{action}</div>}
    </div>
  );
}
export function Metric({
  label,
  value,
  detail,
  icon,
  tone = "blue",
}: {
  label: string;
  value: ReactNode;
  detail: string;
  icon: ReactNode;
  tone?: string;
}) {
  return (
    <div className="w-metric">
      <div className="w-metric-top">
        <span>{label}</span>
        <span className={`w-metric-icon ${tone}`}>{icon}</span>
      </div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}
export function Empty({
  title,
  children,
  action,
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="w-empty">
      <span>
        <Layers size={26} />
      </span>
      <h3>{title}</h3>
      <p>{children}</p>
      {action}
    </div>
  );
}
export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="w-field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}
export function Notice({
  children,
  tone = "blue",
}: {
  children: ReactNode;
  tone?: "blue" | "amber" | "red" | "green";
}) {
  return (
    <div
      className={`w-notice ${tone}`}
      role={tone === "red" ? "alert" : "status"}
    >
      {children}
    </div>
  );
}
export function Steps({
  labels,
  current,
  onSelect,
}: {
  labels: string[];
  current: number;
  onSelect: (i: number) => void;
}) {
  return (
    <div className="w-steps">
      {labels.map((label, i) => (
        <button
          key={label}
          className={i === current ? "active" : i < current ? "complete" : ""}
          onClick={() => onSelect(i)}
        >
          <span>{i < current ? <Check size={14} /> : `0${i + 1}`}</span>
          {label}
          {i < labels.length - 1 && <ChevronRight size={15} />}
        </button>
      ))}
    </div>
  );
}

export function NetworkView({
  graph,
  result,
  closed = [],
  movers = [],
  selected,
  selectedVehicle,
  onSelect,
  onCloseRoad,
  onNode,
  pathEdges = [],
  compact = false,
  episode,
}: {
  graph: Graph;
  result?: Result | null;
  closed?: string[];
  movers?: ReplayFrame["vehicles"];
  selected?: string;
  selectedVehicle?: string;
  onSelect?: (id: string) => void;
  onCloseRoad?: (id: string) => void;
  onNode?: (id: string) => void;
  pathEdges?: string[];
  compact?: boolean;
  episode?: EpisodeData | null;
}) {
  const [zoom, setZoom] = useState(1);
  const nodes = useMemo(
    () => new Map(graph.nodes.map((n) => [n.id, n])),
    [graph],
  );
  const routeColors = useMemo(() => {
    const map = new Map<string, string>();
    result?.evaluation.vehicles.forEach((v, i) => {
      if (!selected || v.id === selected)
        v.edge_ids.forEach((e) => map.set(e, colors[i % colors.length]));
    });
    pathEdges.forEach((e) => map.set(e, colors[0]));
    return map;
  }, [result, selected, pathEdges]);
  if (isGeoGraph(graph))
    return (
      <div className={`w-geographic ${compact ? "compact" : ""}`}>
        <GeoMap
          key={graph.scenario_id}
          graph={graph}
          routeEdges={new Set(routeColors.keys())}
          routePalette={routeColors}
          closed={closed}
          scenarioId={graph.scenario_id}
          movers={movers
            .filter((v) => v.lat != null)
            .map((v) => ({ ...v, lat: v.lat!, lon: v.lon!, heading: v.heading ?? 0 }))}
          selectedVehicle={selectedVehicle}
          onPickEdge={onCloseRoad}
          onPickVehicle={(id) => onSelect?.(id)}
          episode={episode}
        />
      </div>
    );
  const xs = graph.nodes.map((n) => n.x),
    ys = graph.nodes.map((n) => n.y);
  const xmin = Math.min(...xs),
    xmax = Math.max(...xs),
    ymin = Math.min(...ys),
    ymax = Math.max(...ys);
  const point = (n: { x: number; y: number }) => ({
    x: 65 + ((n.x - xmin) / Math.max(1, xmax - xmin)) * 670,
    y: 90 + ((n.y - ymin) / Math.max(1, ymax - ymin)) * 305,
  });
  const depotIds = new Set(graph.fleet.map((v) => v.depot));
  const jobIds = new Set(graph.requests.map((j) => j.node));
  return (
    <div className={`w-network ${compact ? "compact" : ""}`}>
      <svg
        viewBox={`${400 - 400 / zoom} ${240 - 240 / zoom} ${800 / zoom} ${480 / zoom}`}
        aria-label="Directed road network with fleet routes"
        role="img"
      >
        <defs>
          <pattern
            id="city-blocks"
            width="108"
            height="88"
            patternUnits="userSpaceOnUse"
          >
            <rect width="108" height="88" fill="#f1f3f5" />
            <rect x="9" y="9" width="84" height="62" rx="7" fill="#e5e9ed" />
            <rect x="18" y="17" width="28" height="46" rx="3" fill="#eaf0e9" />
            <rect x="52" y="17" width="31" height="20" rx="3" fill="#f7f8fa" />
          </pattern>
        </defs>
        <rect
          x="-800"
          y="-500"
          width="2400"
          height="1500"
          fill="url(#city-blocks)"
        />
        <path
          d="M-50 390 Q160 310 280 405 T880 305"
          stroke="#c8e3ed"
          strokeWidth="33"
          fill="none"
        />
        {graph.edges.map((e) => {
          const a = nodes.get(e.from),
            b = nodes.get(e.to);
          if (!a || !b) return null;
          const p = point(a),
            q = point(b);
          return (
            <line
              key={`road-${e.id}`}
              x1={p.x}
              y1={p.y}
              x2={q.x}
              y2={q.y}
              stroke="#fff"
              strokeWidth="13"
            />
          );
        })}
        {graph.edges.map((e) => {
          const a = nodes.get(e.from),
            b = nodes.get(e.to);
          if (!a || !b) return null;
          const p = point(a),
            q = point(b);
          const blocked = closed.includes(e.id) || !e.open;
          const length = Math.max(1, Math.hypot(q.x - p.x, q.y - p.y));
          const dx = (-(q.y - p.y) / length) * 3,
            dy = ((q.x - p.x) / length) * 3;
          return (
            <g key={e.id}>
              <line
                x1={p.x + dx}
                y1={p.y + dy}
                x2={q.x + dx}
                y2={q.y + dy}
                stroke={
                  blocked ? "#d83e4f" : routeColors.get(e.id) || "#b4becb"
                }
                strokeWidth={routeColors.has(e.id) ? 3.5 : 1.6}
                strokeDasharray={blocked ? "7 4" : undefined}
              />
              {onCloseRoad && (
                <line
                  x1={p.x + dx}
                  y1={p.y + dy}
                  x2={q.x + dx}
                  y2={q.y + dy}
                  stroke="transparent"
                  strokeWidth="6"
                  className="w-road-hit"
                  tabIndex={0}
                  role="button"
                  aria-label={`Toggle closure ${e.id}`}
                  onClick={() => onCloseRoad(e.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") onCloseRoad(e.id);
                  }}
                >
                  <title>
                    {e.id} · {e.from} → {e.to}
                  </title>
                </line>
              )}
            </g>
          );
        })}
        {graph.nodes
          .filter((n) => depotIds.has(n.id) || jobIds.has(n.id) || onNode)
          .map((n) => {
            const p = point(n);
            const depot = depotIds.has(n.id);
            return (
              <g
                key={n.id}
                transform={`translate(${p.x},${p.y})`}
                onClick={() => onNode?.(n.id)}
                role={onNode ? "button" : undefined}
                tabIndex={onNode ? 0 : undefined}
                aria-label={`Select node ${n.id}`}
                onKeyDown={(e) => {
                  if (e.key === "Enter") onNode?.(n.id);
                }}
                className={onNode ? "w-node-hit" : ""}
              >
                <circle
                  r={depot ? 14 : 7}
                  fill={depot ? "#172b4d" : "#fff"}
                  stroke={depot ? "#fff" : "#546fe4"}
                  strokeWidth="3"
                />
                {depot && <path d="M-6 1 L0 -5 L6 1 V7 H-6Z" fill="white" />}
                <text
                  x={depot ? 19 : 12}
                  y="-11"
                  fontSize="11"
                  fontWeight="600"
                  fill="#4d6077"
                  stroke="#fff"
                  strokeWidth="3"
                  paintOrder="stroke"
                >
                  {depot ? "Central depot" : n.id}
                </text>
              </g>
            );
          })}
        {movers.map((v, i) => {
          const p = point(v);
          return (
            <g
              key={v.id}
              transform={`translate(${p.x},${p.y})`}
              onClick={() => onSelect?.(v.id)}
              className="w-node-hit"
            >
              <rect
                x="-12"
                y="-8"
                width="24"
                height="16"
                rx="5"
                fill={colors[i % colors.length]}
                stroke="#fff"
                strokeWidth="2"
              />
              <rect
                x="2"
                y="-5"
                width="5"
                height="10"
                rx="1"
                fill="#fff"
                opacity="0.8"
              />
              <title>{v.id}</title>
            </g>
          );
        })}
      </svg>
      <div className="w-map-label">
        <MapPin size={13} /> Sample road network <span>SCHEMATIC</span>
      </div>
      <div className="w-map-tools">
        <button
          aria-label="Zoom in"
          onClick={() => setZoom(Math.min(zoom + 0.25, 2))}
        >
          <Plus size={16} />
        </button>
        <button
          aria-label="Zoom out"
          onClick={() => setZoom(Math.max(zoom - 0.25, 1))}
        >
          <Minus size={16} />
        </button>
      </div>
      <div className="w-map-legend">
        <span>
          <i style={{ background: "#4361ee" }} />
          Route
        </span>
        <span>
          <i style={{ background: "#172b4d" }} />
          Depot
        </span>
        <span>
          <i style={{ background: "#e35959" }} />
          Closure
        </span>
        {onCloseRoad && (
          <small>Select a directed road to toggle a closure</small>
        )}
      </div>
    </div>
  );
}

export function Chart({
  series,
  label = "Objective",
  xLabel = "Search step",
}: {
  series: { name: string; values: number[]; color?: string }[];
  label?: string;
  xLabel?: string;
}) {
  const valid = series.filter((s) => s.values.some(Number.isFinite));
  if (!valid.length)
    return (
      <Empty title="A trace appears after a run">
        Run a swarm optimiser to inspect its search history.
      </Empty>
    );
  const values = valid.flatMap((s) => s.values.filter(Number.isFinite));
  const high = Math.max(...values),
    low = Math.min(...values);
  const delta = Math.max(high - low, Math.abs(high) * 0.05, 1);
  const lo = Math.max(0, low - delta * 0.12),
    hi = high + delta * 0.12;
  const maxSteps = Math.max(...valid.map((s) => s.values.length - 1), 1);
  return (
    <div className="w-chart">
      <div className="w-chart-legend">
        {valid.map((s, i) => (
          <span key={s.name}>
            <i style={{ background: s.color || colors[i] }} />
            {s.name}
          </span>
        ))}
      </div>
      <svg
        viewBox="0 0 650 255"
        role="img"
        aria-label={`${label} by ${xLabel}`}
      >
        {[0, 1, 2, 3, 4].map((i) => (
          <g key={i}>
            <line
              x1="60"
              x2="625"
              y1={25 + i * 43}
              y2={25 + i * 43}
              stroke="#e9edf3"
            />
            <text
              x="48"
              y={29 + i * 43}
              textAnchor="end"
              fontSize="10"
              fill="#7b879a"
            >
              {(hi - (i / 4) * (hi - lo)).toLocaleString("en", {
                maximumFractionDigits: 0,
              })}
            </text>
          </g>
        ))}
        {valid.map((s, i) => (
          <polyline
            key={s.name}
            points={s.values
              .filter(Number.isFinite)
              .map(
                (v, j) =>
                  `${60 + (j / maxSteps) * 565},${197 - ((v - lo) / (hi - lo)) * 172}`,
              )
              .join(" ")}
            fill="none"
            stroke={s.color || colors[i]}
            strokeWidth="2.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
        {[0, 1, 2, 3, 4].map((i) => (
          <text
            key={i}
            x={60 + i * 141.25}
            y="216"
            textAnchor="middle"
            fontSize="10"
            fill="#7b879a"
          >
            {Math.round((i * maxSteps) / 4)}
          </text>
        ))}
        <text x="340" y="243" textAnchor="middle" fontSize="11" fill="#7b879a">
          {xLabel}
        </text>
      </svg>
    </div>
  );
}
export function RouteList({
  result,
  selected,
  onSelect,
}: {
  result: Result;
  selected?: string;
  onSelect?: (id: string) => void;
}) {
  return (
    <div className="w-route-list">
      {result.evaluation.vehicles.map((v, i) => (
        <button
          key={v.id}
          className={selected === v.id ? "selected" : ""}
          onClick={() => onSelect?.(selected === v.id ? "" : v.id)}
        >
          <span
            className="w-route-icon"
            style={{
              color: colors[i % colors.length],
              background: `${colors[i % colors.length]}14`,
            }}
          >
            <Truck size={19} />
          </span>
          <div>
            <strong>{v.id}</strong>
            <small>
              {v.order.length} stops · {v.load} / {v.capacity} units
            </small>
          </div>
          <Tag tone={v.feasible ? "green" : "red"}>
            {v.feasible ? "Valid" : "Review"}
          </Tag>
          <ChevronRight size={15} />
        </button>
      ))}
    </div>
  );
}
