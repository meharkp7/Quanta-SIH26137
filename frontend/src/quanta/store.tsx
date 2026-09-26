import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { api, postJson, type Graph, type ScenarioSummary, type SolveResult, SOLVE_TIMEOUT_MS } from "@/api";
import { useToast } from "@/components/ui/Toast";
import { scenarioOptions, type ParsedOrders } from "./ui";

// The image's "Optimization Controls" + "Disruption Controls" are one shared
// scenario configuration: the dashboard edits fleet/orders/algorithm, the
// what-if section edits traffic/congestion/closure/delay, and both build the
// same backend snapshot (POST /api/workspace/*) so every section solves the
// exact network the user configured.
export type ScenarioConfig = {
  method: string;
  fleetSize: number;
  capacity: number;
  orders: ParsedOrders | null;
  trafficPct: number;      // traffic increase 0–80%  → speed ÷ (1 + pct)
  congestionScope: string; // zone / road-class group, "" = whole network
  congestionPct: number;   // extra slowdown inside the scope 0–80%
  delayMin: number;        // added service minutes per stop
  closed: string[];        // road closures chosen in the what-if section
};

type Scope = { key: string; label: string; edges: number };

/**
 * Results every section publishes for the experiment assistant to reason over.
 * The shape mirrors the `context` contract of `POST /api/assistant/ask`
 * (see `app/assistant.py`): only real numbers computed by this app belong here,
 * so an answer can never quote a number that did not come from a run.
 */
export type SharedResults = {
  /** Last algorithm benchmark (Route Lab / Forecasting optimiser panel). */
  compare?: { rows: unknown[] } | null;
  /** Baseline-vs-disrupted plan from the what-if section. */
  whatif?: {
    baseline?: { objective?: number; total_distance_m?: number; total_time_s?: number };
    disrupted?: { objective?: number; total_distance_m?: number; total_time_s?: number };
    affected_vehicles?: number;
    unserved?: number;
  } | null;
  /** Selected forecast metrics from the forecasting section. */
  forecast?: {
    zone?: string;
    episode_id?: string;
    model_version?: string;
    horizon_min?: number;
    predicted_kmh?: number;
    observed_kmh?: number;
    horizons?: { horizon: number; mae: number; rmse?: number | null; n?: number }[];
  } | null;
};

type QuantaStore = {
  ready: boolean;
  error: string;
  setError: (value: string) => void;
  scenarios: ScenarioSummary[];
  scenarioId: string;
  setScenarioId: (id: string) => void;
  graph: Graph | null;
  graphLoading: boolean;
  config: ScenarioConfig;
  patch: (next: Partial<ScenarioConfig>) => void;
  scopes: Scope[];
  /** Snapshot payload for /api/workspace/* — `baseline` ignores disruptions. */
  snapshot: (options?: { baseline?: boolean; patch?: Partial<ScenarioConfig> }) => {
    graph: Record<string, unknown>;
    closed_edge_ids: string[];
    traffic_factor: number;
    config: ScenarioConfig;
  };
  solve: SolveResult | null;
  adoptSolve: (result: SolveResult | null) => void;
  solving: boolean;
  solveNote: string;
  runSolve: (options?: {
    patch?: Partial<ScenarioConfig>;
    baseline?: boolean;
    adopt?: boolean;
    quiet?: boolean;
    label?: string;
  }) => Promise<SolveResult | null>;
  /** True once the what-if section has run a replan (highlights the rail). */
  replanned: boolean;
  markReplanned: () => void;
  /** Cross-section results shared with the experiment assistant. */
  results: SharedResults;
  publishResults: (patch: Partial<SharedResults>) => void;
};

const Context = createContext<QuantaStore | null>(null);

const DEFAULT_METHOD = "qpso";

// Backend snapshot limits (app/workspace.py) — mirrored for honest client checks.
const LIMITS = { nodes: [2, 5000], edges: [1, 15000], requests: [1, 200], fleet: [1, 50] } as const;

function resizeFleet(graph: Graph, size: number, capacity: number) {
  const base = graph.fleet.length ? graph.fleet : [{ id: "V1", capacity: 1, depot: graph.nodes[0]?.id || "N0" }];
  const count = Math.max(LIMITS.fleet[0], Math.min(LIMITS.fleet[1], Math.round(size)));
  return Array.from({ length: count }, (_, i) => {
    const template = base[i % base.length];
    return {
      id: i < base.length ? template.id : `${template.id.replace(/-\d+$/, "")}-${i}`,
      capacity,
      depot: template.depot,
    };
  });
}

export function QuantaProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [graph, setGraph] = useState<Graph | null>(null);
  const [graphLoading, setGraphLoading] = useState(true);
  const [solve, setSolve] = useState<SolveResult | null>(null);
  const [solving, setSolving] = useState(false);
  const [solveNote, setSolveNote] = useState("");
  const [replanned, setReplanned] = useState(false);
  const [results, setResults] = useState<SharedResults>({});
  const [config, setConfig] = useState<ScenarioConfig>({
    method: DEFAULT_METHOD,
    fleetSize: 0,
    capacity: 0,
    orders: null,
    trafficPct: 0,
    congestionScope: "",
    congestionPct: 0,
    delayMin: 0,
    closed: [],
  });
  const { push: toast } = useToast();
  const graphSeq = useRef(0);

  // ── Scenario catalog: pick a real Delhi map first (matches the demo) ─────
  useEffect(() => {
    api<{ scenarios: ScenarioSummary[] }>("/api/scenarios")
      .then((res) => {
        const list = res.scenarios || [];
        setScenarios(list);
        const preferred = list.find((s) => s.available && s.id === "DELHI_CNP") ||
          list.find((s) => s.available && s.id.startsWith("DELHI_")) ||
          list.find((s) => s.available && s.id === "S3_BASE") ||
          list.find((s) => s.available);
        if (preferred) setScenarioId(preferred.id);
        else setError("No scenarios are available from the backend.");
        setReady(true);
      })
      .catch((err: Error) => {
        setError(err.message);
        setReady(true);
      });
  }, []);

  // ── Graph for the selected scenario ─────────────────────────────────────
  useEffect(() => {
    if (!scenarioId) return;
    const seq = ++graphSeq.current;
    setGraphLoading(true);
    setSolve(null);
    api<Graph>(`/api/scenarios/${encodeURIComponent(scenarioId)}`)
      .then((next) => {
        if (graphSeq.current !== seq) return;
        setGraph(next);
        setSolve(null);
        setReplanned(false);
        setResults({});
        setConfig((current) => ({
          ...current,
          fleetSize: current.fleetSize || next.fleet.length,
          capacity: current.capacity || Math.max(1, ...next.fleet.map((v) => Math.round(v.capacity))),
          orders: null,
          trafficPct: 0,
          congestionScope: "",
          congestionPct: 0,
          delayMin: 0,
          closed: [],
        }));
        setGraphLoading(false);
      })
      .catch((err: Error) => {
        if (graphSeq.current !== seq) return;
        setError(err.message);
        setGraphLoading(false);
      });
  }, [scenarioId]);

  const patch = useCallback((next: Partial<ScenarioConfig>) => {
    setConfig((current) => ({ ...current, ...next }));
  }, []);

  // Congestion scope groups: real node zones when the map has more than one,
  // otherwise road classes (Delhi OSM maps are single-zone by construction).
  const scopes = useMemo<Scope[]>(() => {
    if (!graph) return [];
    const zones = new Map<string, number>();
    const classes = new Map<string, number>();
    const zoneOf = new Map(graph.nodes.map((n) => [n.id, n.zone || "osm"]));
    for (const edge of graph.edges) {
      const zone = zoneOf.get(edge.from) || "?";
      zones.set(zone, (zones.get(zone) || 0) + 1);
      const cls = edge.road_class.replace(/^RoadClass\./, "").toLowerCase();
      classes.set(cls, (classes.get(cls) || 0) + 1);
    }
    const source = zones.size > 1 ? zones : classes;
    const label = zones.size > 1 ? "zone" : "road class";
    return [...source.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([key, edges]) => ({ key, label: `${key} · ${label} · ${edges} roads`, edges }));
  }, [graph]);

  const snapshot = useCallback(
    (options?: { baseline?: boolean; patch?: Partial<ScenarioConfig> }): ReturnType<QuantaStore["snapshot"]> => {
      const cfg: ScenarioConfig = { ...config, ...(options?.patch || {}) };
      const baseline = !!options?.baseline;
      if (!graph) return { graph: {}, closed_edge_ids: [], traffic_factor: 1, config: cfg };

      const trafficFactor = baseline ? 1 : 1 / (1 + Math.max(0, cfg.trafficPct) / 100);
      const zoneOf = new Map(graph.nodes.map((n) => [n.id, n.zone || "osm"]));
      const classOf = (edge: Graph["edges"][number]) => edge.road_class.replace(/^RoadClass\./, "").toLowerCase();
      const inScope = (edge: Graph["edges"][number]) =>
        cfg.congestionScope === "__all__" ||
        zoneOf.get(edge.from) === cfg.congestionScope ||
        classOf(edge) === cfg.congestionScope;
      const scopeSet =
        !baseline && cfg.congestionScope ? new Set(graph.edges.filter(inScope).map((edge) => edge.id)) : null;
      const congestion = baseline ? 0 : cfg.congestionPct / 100;
      const delay = baseline ? 0 : cfg.delayMin * 60;

      const requests = (cfg.orders?.rows?.length ? cfg.orders.rows : graph.requests)
        .slice(0, LIMITS.requests[1])
        .map((row) => ({
          id: row.id,
          node: row.node,
          demand: row.demand,
          earliest_s: row.earliest_s,
          latest_s: row.latest_s,
          service_s: (row.service_s || 0) + delay,
        }));

      const payload = {
        geo: graph.geo,
        nodes: graph.nodes.map((n) => ({ id: n.id, x: n.x, y: n.y, kind: n.kind, zone: n.zone })),
        edges: graph.edges.map((edge) => ({
          id: edge.id,
          from: edge.from,
          to: edge.to,
          length_m: edge.length_m,
          speed_mps: edge.speed_mps * (scopeSet?.has(edge.id) ? 1 - congestion : 1),
          road_class: edge.road_class,
          open: edge.open,
        })),
        requests,
        fleet: resizeFleet(graph, cfg.fleetSize || graph.fleet.length, cfg.capacity || 1),
      };

      return {
        graph: payload as unknown as Record<string, unknown>,
        closed_edge_ids: baseline ? [] : cfg.closed,
        traffic_factor: Number(trafficFactor.toFixed(4)),
        config: cfg,
      };
    },
    [config, graph, scopes],
  );

  const runSolve = useCallback(
    async (options?: {
      patch?: Partial<ScenarioConfig>;
      baseline?: boolean;
      adopt?: boolean;
      quiet?: boolean;
      label?: string;
    }): Promise<SolveResult | null> => {
      if (!graph) return null;
      setSolving(true);
      setSolveNote(options?.label || "Solving routes…");
      setError("");
      try {
        const built = snapshot({ baseline: options?.baseline, patch: options?.patch });
        const request: Record<string, unknown> = {
          graph: built.graph,
          method: built.config.method,
          particles: 2,
          evaluations: 4,
          seed: 7,
          closed_edge_ids: built.closed_edge_ids,
          traffic_factor: built.traffic_factor,
        };
        if (built.config.method !== "constructive") {
          request.particles = 4;
          request.evaluations = 8;
        }
        const result = await postJson<SolveResult>("/api/workspace/solve", request, SOLVE_TIMEOUT_MS);
        if (options?.adopt !== false) setSolve(result);
        if (!options?.quiet) {
          if (result.evaluation.feasible) {
            toast({
              kind: "success",
              title: result.cached ? "Solve cached" : "Solve complete",
              message: `${result.method} · objective ${result.evaluation.objective.toFixed(0)} · ${(result.elapsed_s || 0).toFixed(1)}s`,
              duration: 3500,
            });
          } else {
            toast({
              kind: "warning",
              title: "Plan has constraint issues",
              message: result.evaluation.violations[0]?.detail || "The independent validator flagged this plan.",
              duration: 5000,
            });
          }
        }
        return result;
      } catch (err) {
        const message = (err as Error).message;
        setError(message);
        toast({ kind: "error", title: "Solve failed", message, duration: 6000 });
        return null;
      } finally {
        setSolving(false);
        setSolveNote("");
      }
    },
    [graph, snapshot, toast],
  );

  const adoptSolve = useCallback((result: SolveResult | null) => setSolve(result), []);

  const publishResults = useCallback((patch: Partial<SharedResults>) => {
    setResults((current) => ({ ...current, ...patch }));
  }, []);

  const value: QuantaStore = {
    ready,
    error,
    setError,
    scenarios,
    scenarioId,
    setScenarioId: (id: string) => {
      setScenarioId(id);
      setError("");
    },
    graph,
    graphLoading,
    config,
    patch,
    scopes,
    snapshot,
    solve,
    adoptSolve,
    solving,
    solveNote,
    runSolve,
    replanned,
    markReplanned: () => setReplanned(true),
    results,
    publishResults,
  };

  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function useQuanta() {
  const value = useContext(Context);
  if (!value) throw new Error("QuantaProvider missing");
  return value;
}

export { scenarioOptions, LIMITS };
