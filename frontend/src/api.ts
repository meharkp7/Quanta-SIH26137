import { identityHeaders } from './workspace/supabase';
export type ScenarioSummary = { id: string; label: string; role: string; description: string; available: boolean };
export type GraphNode = { id: string; x: number; y: number; kind: string; zone: string; lat?: number; lon?: number };
export type GraphEdge = { id: string; from: string; to: string; length_m: number; speed_mps: number; road_class: string; open: boolean; free_flow_s: number };
export type Graph = { scenario_id: string; graph_version: string; geo?: { available: boolean; crs: string; source_crs: string }; nodes: GraphNode[]; edges: GraphEdge[]; requests: { id: string; node: string; demand: number }[]; fleet: { id: string; capacity: number; depot: string }[]; reference_plan: Record<string, string[]> };
export type VehicleResult = { id: string; order: string[]; load: number; capacity: number; elapsed_s: number; edge_ids: string[]; stops: { job: string; start_s: number }[]; feasible: boolean };
export type Evaluation = { feasible: boolean; all_served: boolean; capacity: boolean; windows: boolean; connectivity: boolean; depot?: boolean; objective: number; time_s: number; distance_m: number; congestion_s: number; vehicles: VehicleResult[]; violations: { name: string; detail: string }[] };
export type SolveResult = { method: string; status: string; elapsed_s: number; evaluations?: number; budget_note?: string; cached?: boolean; evaluation: Evaluation; trace?: { best: number[]; mean: number[]; diversity: number[] }; plan: Record<string, string[]>; closed_edge_ids: string[] };
export type StoryAction = { http_method: "GET" | "POST"; endpoint: string; params: Record<string, unknown> };
export type StoryStep = { id: string; title: string; caption: string; action: StoryAction | null };
export type Story = { scenario_id: string; steps: StoryStep[] };
export type CompareRow = { method: string; feasible: boolean; objective?: number; time_s?: number; distance_m?: number; elapsed_s?: number; error?: string };
export type CompareResult = { scenario_id: string; closed_edge_ids: string[]; rows: CompareRow[] };

export async function api<T>(path: string, options?: RequestInit, timeoutMs?: number): Promise<T> {
  const controller = new AbortController();
  const external = options?.signal;
  const signal = external && typeof AbortSignal.any === "function"
    ? AbortSignal.any([external, controller.signal])
    : external || controller.signal;
  const timer = timeoutMs ? window.setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const response = await fetch(path, { ...options, headers: { "Content-Type": "application/json", ...identityHeaders(), ...options?.headers }, signal });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error((payload as { detail?: string }).detail || response.statusText);
    return payload as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      if (external && (external as AbortSignal).aborted) throw err;
      throw new Error(`Request to ${path} timed out after ${Math.round((timeoutMs || 0) / 1000)}s — try a smaller budget or another scenario.`);
    }
    throw err;
  } finally {
    if (timer) window.clearTimeout(timer);
  }
}

export const SOLVE_TIMEOUT_MS = 150000;

export const postJson = <T,>(path: string, body: unknown, timeoutMs?: number, signal?: AbortSignal) => api<T>(path, { method: "POST", body: JSON.stringify(body), ...(signal ? { signal } : {}) }, timeoutMs);
export type ReplayFrame = {
  t: number;
  vehicles: { id: string; x: number; y: number; lat?: number; lon?: number; kind?: string; stopped?: boolean }[];
  closed?: string[];
};
export type ReplayIncident = { incident_id: string; edge_id: string; trigger_time_s: number };
export type ReplayResult = { frames: ReplayFrame[]; incidents?: ReplayIncident[]; message?: string; mode?: string };
export type Evidence = { available: boolean; artifact?: string; model_mae?: number | null; temporal_baseline_mae?: number | null; uncertainty?: { validation?: { actual_coverage?: number }; test?: { actual_coverage?: number }; nominal_coverage?: number }; test_metrics?: Record<string, { mae?: number; rmse?: number }>; training_cutoff_s?: number; split_policy?: string; reason?: string; demo?: boolean; provenance?: string; step14_demo?: { corridor_cost_delta_s?: number; reroute_proof?: string; fifo_holds?: boolean } | null; step9?: { milp_certified_objective?: number; detail?: string } | null; data_audit?: { headline?: string } | null };
export type PathResult = { method: string; exact: boolean; feasible: boolean; source?: string; target?: string; node_ids?: string[]; edge_ids?: string[]; distance_m?: number; time_s?: number; congestion_s?: number; elapsed_s?: number; error?: string };
export type DrlDemo = { action: string; scope?: string; job_id?: string; vehicle_id?: string; reason?: string; provenance?: string; demo?: boolean };
