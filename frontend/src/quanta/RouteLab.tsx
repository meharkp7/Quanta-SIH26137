import { useMemo, useState } from "react";
import { GitCompare, MapPin, Route as RouteIcon, Timer, Waypoints } from "lucide-react";
import { postJson, SOLVE_TIMEOUT_MS, type CompareRow, type Graph, type PathResult } from "@/api";
import { useQuanta } from "./store";
import { Box, Btn, Field, Loader, Pill, SectionHead, Select, Slider, Stat, Status, fmtKm, revealStyle } from "./ui";
import { ConvergenceChart, ObjectiveBars } from "./charts";
import { MapPanel, ROUTE_COLORS, type RoutePath } from "./map";

// The backend snapshot edges (app/workspace.py rewrites speed by traffic_factor
// and congestion scope) — used for the client-side distance optimiser so both
// cost types see the exact network the user configured.
type SnapEdge = { id: string; from: string; to: string; length_m: number; speed_mps: number; road_class: string };

type CompareResponse = {
  scenario_id: string;
  closed_edge_ids: string[];
  rows: CompareRow[];
  traces?: Record<string, { best?: number[]; mean?: number[]; diversity?: number[] }>;
};

type FoundPath = {
  nodes: string[];
  edges: SnapEdge[];
  cost: number;
  via: "backend" | "client";
};

/** Directed Dijkstra — matches the backend's DirectedRoadGraph (one-way arcs only). */
function dijkstra(edges: SnapEdge[], src: string, dst: string, weight: (edge: SnapEdge) => number): FoundPath | null {
  if (!src || !dst || src === dst) return null;
  const adjacency = new Map<string, SnapEdge[]>();
  for (const edge of edges) {
    const list = adjacency.get(edge.from);
    if (list) list.push(edge);
    else adjacency.set(edge.from, [edge]);
  }
  const dist = new Map<string, number>([[src, 0]]);
  const previous = new Map<string, { node: string; edge: SnapEdge }>();
  const settled = new Set<string>();
  const queue = [src];
  while (queue.length) {
    let bestIndex = 0;
    let bestDistance = Infinity;
    for (let i = 0; i < queue.length; i += 1) {
      const d = dist.get(queue[i]) ?? Infinity;
      if (d < bestDistance) {
        bestDistance = d;
        bestIndex = i;
      }
    }
    const current = queue.splice(bestIndex, 1)[0];
    if (settled.has(current)) continue;
    settled.add(current);
    if (current === dst) break;
    for (const edge of adjacency.get(current) || []) {
      const next = edge.to;
      if (settled.has(next)) continue;
      const candidate = bestDistance + Math.max(0, weight(edge));
      if (candidate < (dist.get(next) ?? Infinity)) {
        dist.set(next, candidate);
        previous.set(next, { node: current, edge });
        queue.push(next);
      }
    }
  }
  if (!settled.has(dst)) return null;
  const nodes: string[] = [dst];
  const path: SnapEdge[] = [];
  let cursor = dst;
  while (cursor !== src) {
    const step = previous.get(cursor);
    if (!step) return null;
    path.unshift(step.edge);
    nodes.unshift(step.node);
    cursor = step.node;
  }
  return { nodes, edges: path, cost: dist.get(dst) ?? 0, via: "client" };
}

function nodeOptions(graph: Graph | null) {
  if (!graph) return [];
  const depots = new Set(graph.fleet.map((v) => v.depot));
  return graph.nodes.map((n) => ({
    value: n.id,
    label: `${n.id.length > 22 ? `${n.id.slice(0, 22)}…` : n.id}${depots.has(n.id) ? " · depot" : ""}${n.kind === "customer" ? " · customer" : ""}`,
  }));
}

const ALL_METHODS = ["constructive", "qpso", "pso", "alns", "milp"] as const;
const METHOD_LABELS: Record<string, string> = {
  constructive: "Constructive",
  qpso: "QPSO",
  pso: "PSO",
  alns: "ALNS",
  milp: "MILP (exact)",
};

/** The API returns method names in either case (``CONSTRUCTIVE`` / ``qpso``). */
const methodLabel = (method?: string | null): string =>
  (method ? METHOD_LABELS[method.trim().toLowerCase()] || method : "");

// Section 4 — ROUTE LAB.
export function RouteLab() {
  const { graph, config, snapshot, scopes, scenarioId, publishResults } = useQuanta();

  /* ── Shortest path ─────────────────────────────────────────────────── */
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [costType, setCostType] = useState<"time" | "distance">("time");
  const [primary, setPrimary] = useState<FoundPath | null>(null);
  const [alternative, setAlternative] = useState<FoundPath | null>(null);
  const [pathBusy, setPathBusy] = useState(false);
  const [pathError, setPathError] = useState("");

  // Sensible defaults: depot → first delivery node.
  const defaults = useMemo(() => {
    if (!graph) return null;
    const depot = graph.fleet[0]?.depot || graph.nodes[0]?.id || "";
    const customer = graph.requests[0]?.node || graph.nodes.find((n) => n.id !== depot)?.id || "";
    return { depot, customer };
  }, [graph]);

  const options = useMemo(() => nodeOptions(graph), [graph]);
  const from = source || defaults?.depot || "";
  const to = target || defaults?.customer || "";

  // The exact edge set the snapshot sends (traffic + congestion applied).
  function snapshotEdges() {
    const payload = snapshot();
    const raw = payload.graph as { edges?: SnapEdge[] };
    const closed = new Set(payload.closed_edge_ids);
    return {
      edges: (raw.edges || []).filter((edge) => !closed.has(edge.id)),
      timeWeight: (edge: SnapEdge) => edge.length_m / Math.max(0.1, edge.speed_mps),
    };
  }

  async function findPath() {
    if (!graph || !from || !to || from === to) return;
    setPathBusy(true);
    setPathError("");
    setPrimary(null);
    setAlternative(null);
    try {
      const { edges, timeWeight } = snapshotEdges();
      const distancePath = dijkstra(edges, from, to, (edge) => edge.length_m);

      let timePath: FoundPath | null = null;
      const payload = snapshot();
      const backend = await postJson<PathResult>(
        "/api/workspace/path",
        {
          graph: payload.graph,
          closed_edge_ids: payload.closed_edge_ids,
          traffic_factor: payload.traffic_factor,
          method: config.method,
          source: from,
          target: to,
        },
        60000,
      );
      if (backend.feasible && backend.node_ids?.length) {
        const byId = new Map(edges.map((edge) => [edge.id, edge]));
        const pathEdges = (backend.edge_ids || []).map((id) => byId.get(id)).filter((e): e is SnapEdge => !!e);
        timePath = {
          nodes: backend.node_ids,
          edges: pathEdges,
          cost: pathEdges.reduce((sum, edge) => sum + timeWeight(edge), 0),
          via: "backend",
        };
      }

      if (costType === "time") {
        if (!timePath) throw new Error(backend.error || "No time-optimal path exists between those nodes.");
        setPrimary(timePath);
        setAlternative(distancePath ? { ...distancePath, cost: distancePath.cost } : null);
      } else {
        if (!distancePath) throw new Error("No connected route between those nodes on the current network.");
        setPrimary(distancePath);
        setAlternative(timePath);
      }
    } catch (err) {
      setPathError(err instanceof Error ? err.message : String(err));
    } finally {
      setPathBusy(false);
    }
  }

  const route: RoutePath[] = useMemo(
    () => (primary ? [{ id: "shortest-path", color: ROUTE_COLORS[0], edgeIds: primary.edges.map((e) => e.id) }] : []),
    [primary],
  );
  const minutes = (path: FoundPath | null) => (path ? path.cost / 60 : null);
  const altDelta = useMemo(() => {
    if (!primary || !alternative) return null;
    if (costType === "time") {
      const saved = alternative.cost - primary.cost;
      return saved > 0 ? saved / 60 : null;
    }
    const saved = alternative.cost - primary.cost;
    return saved > 0 ? saved / 1000 : null;
  }, [primary, alternative, costType]);

  /* ── Algorithm comparison ──────────────────────────────────────────── */
  const [deliveries, setDeliveries] = useState(12);
  const [selected, setSelected] = useState<string[]>(["constructive", "qpso", "pso", "alns"]);
  const [compare, setCompare] = useState<CompareResponse | null>(null);
  const [cmpBusy, setCmpBusy] = useState(false);
  const [cmpError, setCmpError] = useState("");
  const [traceMethod, setTraceMethod] = useState("");

  const maxDeliveries = Math.max(4, Math.min(160, graph?.requests.length || 160));
  const milpAllowed = deliveries <= 12;

  function toggleMethod(method: string) {
    setSelected((current) =>
      current.includes(method) ? current.filter((m) => m !== method) : [...current, method],
    );
  }

  async function runCompare() {
    if (!graph) return;
    setCmpBusy(true);
    setCmpError("");
    try {
      const payload = snapshot();
      // Truncate deliveries so the MILP eligibility rule matches the shown count.
      const bodyGraph = payload.graph as { requests?: unknown[] };
      const sliced: typeof bodyGraph = { ...bodyGraph, requests: (bodyGraph.requests || []).slice(0, deliveries) };
      const methods = ALL_METHODS.filter((m) => selected.includes(m) && (m !== "milp" || milpAllowed));
      if (!methods.length) throw new Error("Select at least one algorithm.");
      const result = await postJson<CompareResponse>(
        "/api/workspace/compare",
        {
          graph: sliced,
          closed_edge_ids: payload.closed_edge_ids,
          traffic_factor: payload.traffic_factor,
          method: config.method,
          particles: 4,
          evaluations: 8,
          seed: 7,
          methods,
        },
        SOLVE_TIMEOUT_MS,
      );
      setCompare(result);
      // Share the benchmark with the experiment assistant (bottom-right chat).
      publishResults({ compare: { rows: result.rows } });
      const traces = Object.keys(result.traces || {});
      setTraceMethod((current) => (current && traces.includes(current) ? current : traces[0] || ""));
    } catch (err) {
      setCmpError(err instanceof Error ? err.message : String(err));
    } finally {
      setCmpBusy(false);
    }
  }

  const rows = compare?.rows || [];
  const bestObjective = useMemo(() => {
    const values = rows.filter((r) => typeof r.objective === "number" && r.feasible).map((r) => r.objective as number);
    return values.length ? Math.min(...values) : null;
  }, [rows]);
  // Several solvers often land on the same objective — badge the first one
  // that reaches it so the column does not fill up with identical "best" tags.
  const bestMethod = useMemo(() => {
    if (bestObjective == null) return null;
    const tie = rows.find((r) => r.feasible && r.objective === bestObjective);
    return tie?.method ?? null;
  }, [rows, bestObjective]);
  const traces = compare?.traces || {};
  const traceValues = traces[traceMethod]?.best || [];

  return (
    <div className="q-section">
      <SectionHead
        n={4}
        title="ROUTE LAB"
        sub="Trace optimal corridors and benchmark every solver"
        right={
          <div className="q-ops-bar">
            <Pill tone="blue" dot>{scenarioId || "workspace"}</Pill>
            <span className="q-ops-date">
              {graph ? `${graph.nodes.length.toLocaleString()} nodes · ${graph.edges.length.toLocaleString()} roads` : "loading network…"}
            </span>
          </div>
        }
      />

      <div className="q-lab-grid" style={revealStyle(60)}>
        <Box
          title={<><Waypoints size={15} /> Shortest Path Finder</>}
          action={primary ? <span className="q-badge">cost: {costType === "time" ? "travel time" : "distance"}</span> : undefined}
        >
          <div className="q-field-row">
            <Field label="Source node">
              <Select ariaLabel="Source node" value={from} onChange={setSource} options={options} />
            </Field>
            <Field label="Destination node">
              <Select ariaLabel="Destination node" value={to} onChange={setTarget} options={options} />
            </Field>
          </div>
          <Field label="Cost type" hint="Travel time uses the backend Dijkstra on your live snapshot; distance is optimised client-side on the same roads.">
            <Select
              ariaLabel="Cost type"
              value={costType}
              onChange={(value) => setCostType(value as "time" | "distance")}
              options={[
                { value: "time", label: "Travel Time (minutes)" },
                { value: "distance", label: "Distance (km)" },
              ]}
            />
          </Field>
          <div className="q-actions">
            <Btn disabled={pathBusy || !graph || from === to} onClick={() => void findPath()}>
              {pathBusy ? "Finding path…" : "Find Path"}
            </Btn>
            {primary && (
              <button className="q-link" onClick={() => { setPrimary(null); setAlternative(null); setPathError(""); }}>
                Clear
              </button>
            )}
          </div>
          {pathBusy && <Loader label="Running Dijkstra on the live snapshot…" />}
          {pathError && <Status tone="red">{pathError}</Status>}

          {primary && !pathBusy && (
            <>
              <div className="q-kpis q-kpis-tight">
                <Stat icon={<RouteIcon />} value={fmtKm(primary.edges.reduce((s, e) => s + e.length_m, 0))} label="Path distance" />
                <Stat icon={<Timer />} value={costType === "time" ? `${minutes(primary)!.toFixed(1)} min` : `${(primary.cost / 1000).toFixed(2)} km`} label={costType === "time" ? "Travel time" : "Path length"} tone="blue" />
                <Stat icon={<MapPin />} value={primary.edges.length} label="Road segments" tone="amber" />
                <Stat icon={<GitCompare />} value={`${primary.nodes.length}`} label="Nodes crossed" tone="lime" />
              </div>
              <table className="q-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Road</th>
                    <th>From → To</th>
                    <th>Length</th>
                    <th>Class</th>
                  </tr>
                </thead>
                <tbody>
                  {primary.edges.slice(0, 8).map((edge, index) => (
                    <tr key={`${edge.id}-${index}`}>
                      <td>{index + 1}</td>
                      <td className="q-mono">{edge.id.length > 16 ? `${edge.id.slice(0, 16)}…` : edge.id}</td>
                      <td className="q-mono">{edge.from} → {edge.to}</td>
                      <td>{Math.round(edge.length_m)} m</td>
                      <td>{edge.road_class}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {primary.edges.length > 8 && (
                <p className="q-hint">+ {primary.edges.length - 8} more segments in this corridor.</p>
              )}
              {alternative && altDelta != null && (
                <p className="q-hint">
                  The {costType === "time" ? "shortest-distance" : "fastest"} alternative is{" "}
                  <strong>{costType === "time" ? `${altDelta.toFixed(1)} min slower` : `${altDelta.toFixed(2)} km longer`}</strong>{" "}
                  — pick the other cost type to inspect it.
                </p>
              )}
            </>
          )}
        </Box>

        <Box
          title="Corridor on map"
          action={primary ? <Pill tone="green" dot>path found</Pill> : undefined}
          className="q-map-box"
        >
          <MapPanel
            graph={graph}
            routes={route}
            closed={config.closed}
            height={468}
            animate={false}
            caption={
              primary
                ? `${costType === "time" ? "Minimum travel time" : "Minimum distance"} · ${primary.via === "backend" ? "backend Dijkstra" : "client Dijkstra"}`
                : "Choose a source and destination, then press Find Path."
            }
          />
        </Box>
      </div>

      <Box
        title={<><GitCompare size={15} /> Algorithm Comparison</>}
        action={compare ? <Pill tone="green" dot>{rows.length} algorithms</Pill> : cmpBusy ? <Pill tone="amber" dot>running</Pill> : undefined}
        className="q-compare-box"
      >
        <div className="q-compare-controls">
          <div className="q-slider-block">
            <Slider
              label="Deliveries in the benchmark"
              value={deliveries}
              min={4}
              max={maxDeliveries}
              step={1}
              unit=" orders"
              onChange={setDeliveries}
            />
            <em className="q-field-hint">MILP is exact but only certified for 12 deliveries or fewer.</em>
          </div>
          <div className="q-chipset" role="group" aria-label="Algorithms">
            {ALL_METHODS.map((method) => {
              const disabled = method === "milp" && !milpAllowed;
              const active = selected.includes(method);
              return (
                <button
                  key={method}
                  type="button"
                  className={`q-chip ${active ? "on" : ""}`}
                  disabled={disabled}
                  title={disabled ? "MILP runs only with 12 deliveries or fewer" : undefined}
                  onClick={() => toggleMethod(method)}
                >
                  {methodLabel(method)}
                </button>
              );
            })}
          </div>
          <div className="q-actions">
            <Btn disabled={cmpBusy || !graph} onClick={() => void runCompare()}>
              {cmpBusy ? "Benchmarking…" : "Run Comparison"}
            </Btn>
            {!milpAllowed && selected.includes("milp") && (
              <span className="q-hint">MILP disabled above 12 deliveries.</span>
            )}
          </div>
          {cmpBusy && <Loader label="Benchmarking each solver on your snapshot — this runs the full pipeline…" />}
          {cmpError && <Status tone="red">{cmpError}</Status>}
        </div>

        {rows.length > 0 && (
          <div className="q-compare-results">
            <table className="q-table">
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>Status</th>
                  <th>Objective</th>
                  <th>Distance</th>
                  <th>Route time</th>
                  <th>Congestion</th>
                  <th>Solve time</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const isBest = bestObjective != null && bestMethod === row.method
                    && row.feasible && row.objective === bestObjective;
                  return (
                    <tr key={row.method} className={isBest ? "q-row-best" : ""}>
                      <td><strong>{methodLabel(row.method)}</strong></td>
                      <td>
                        {row.error ? (
                          <Status tone="red">failed</Status>
                        ) : row.feasible ? (
                          <Status tone="green">feasible</Status>
                        ) : (
                          <Status tone="amber">infeasible</Status>
                        )}
                      </td>
                      <td className="q-num">{row.objective != null ? row.objective.toLocaleString(undefined, { maximumFractionDigits: 0 }) : "—"}</td>
                      <td>{row.distance_m != null ? fmtKm(row.distance_m) : "—"}</td>
                      <td>{row.time_s != null ? `${(row.time_s / 60).toFixed(1)} min` : "—"}</td>
                      <td>{(row as CompareRow & { congestion_s?: number }).congestion_s != null ? `${((row as CompareRow & { congestion_s?: number }).congestion_s as number).toFixed(0)} s` : "—"}</td>
                      <td>{row.elapsed_s != null ? `${row.elapsed_s.toFixed(2)} s` : "—"}</td>
                      <td>{isBest && <Pill tone="green">best</Pill>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {rows.some((r) => r.error) && (
              <p className="q-hint">
                {rows.filter((r) => r.error).map((r) => `${methodLabel(r.method)}: ${r.error}`).join(" · ")}
              </p>
            )}
            <div className="q-compare-charts">
              <div>
                <h4 className="q-subhead">Objective by algorithm</h4>
                <ObjectiveBars rows={rows.filter((r) => typeof r.objective === "number").map((r) => ({
                  method: methodLabel(r.method),
                  objective: r.objective as number,
                  feasible: r.feasible,
                }))} />
              </div>
              <div>
                <h4 className="q-subhead">
                  Convergence trace
                  {Object.keys(traces).length > 1 && (
                    <Select
                      ariaLabel="Trace algorithm"
                      value={traceMethod}
                      onChange={setTraceMethod}
                      options={Object.keys(traces).map((m) => ({ value: m, label: methodLabel(m) }))}
                    />
                  )}
                </h4>
                <ConvergenceChart
                  values={traceValues}
                  mean={traces[traceMethod]?.mean}
                  meanLabel="population mean"
                  label={`${methodLabel(traceMethod) || "solver"} best-so-far`}
                />
              </div>
            </div>
          </div>
        )}

        {!rows.length && !cmpBusy && (
          <p className="q-empty">
            No benchmark yet — pick algorithms and press <strong>Run Comparison</strong>. Every row is solved on the same snapshot
            (fleet {config.fleetSize || graph?.fleet.length || 0}, capacity {config.capacity || 1}
            {scopes.length && config.congestionScope ? ` · congestion in scope` : ""}
            {config.closed.length ? ` · ${config.closed.length} closure(s)` : ""}).
          </p>
        )}
      </Box>
    </div>
  );
}
