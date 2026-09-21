export type ScenarioSummary = { id: string; label: string; role: string; description: string; available: boolean };
export type GraphNode = { id: string; x: number; y: number; kind: string; zone: string };
export type GraphEdge = { id: string; from: string; to: string; length_m: number; speed_mps: number; road_class: string; open: boolean; free_flow_s: number };
export type Graph = { scenario_id: string; graph_version: string; nodes: GraphNode[]; edges: GraphEdge[]; requests: { id: string; node: string; demand: number }[]; fleet: { id: string; capacity: number; depot: string }[]; reference_plan: Record<string, string[]> };
export type VehicleResult = { id: string; order: string[]; load: number; capacity: number; elapsed_s: number; edge_ids: string[]; stops: { job: string; start_s: number }[]; feasible: boolean };
export type Evaluation = { feasible: boolean; all_served: boolean; capacity: boolean; windows: boolean; connectivity: boolean; depot?: boolean; objective: number; time_s: number; distance_m: number; congestion_s: number; vehicles: VehicleResult[]; violations: { name: string; detail: string }[] };
export type SolveResult = { method: string; status: string; elapsed_s: number; evaluations?: number; evaluation: Evaluation; trace?: { best: number[]; mean: number[]; diversity: number[] }; plan: Record<string, string[]>; closed_edge_ids: string[] };
export type ReplayFrame = { t: number; vehicles: { id: string; x: number; y: number; kind?: string; stopped?: boolean }[]; closed?: string[] };
export type Evidence = { available: boolean; artifact?: string; model_mae?: number | null; temporal_baseline_mae?: number | null; uncertainty?: { validation?: { actual_coverage?: number }; test?: { actual_coverage?: number }; nominal_coverage?: number }; test_metrics?: Record<string, { mae?: number; rmse?: number }>; training_cutoff_s?: number; split_policy?: string; reason?: string; demo?: boolean; provenance?: string; step14_demo?: { corridor_cost_delta_s?: number; reroute_proof?: string; fifo_holds?: boolean } | null; step9?: { milp_certified_objective?: number; detail?: string } | null; data_audit?: { headline?: string } | null };
export type DrlDemo = { action: string; scope?: string; job_id?: string; vehicle_id?: string; reason?: string; provenance?: string; demo?: boolean };

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error((payload as { detail?: string }).detail || response.statusText);
  return payload as T;
}

export const postJson = <T,>(path: string, body: unknown) => api<T>(path, { method: "POST", body: JSON.stringify(body) });
