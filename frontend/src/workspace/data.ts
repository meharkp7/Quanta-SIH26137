import type { Graph, SolveResult, CompareResult, CompareRow } from "../api";

export type Delivery = Graph["requests"][number] & {
  earliest_s: number;
  latest_s: number;
  service_s: number;
};
export type Network = Omit<Graph, "requests"> & { requests: Delivery[] };
export type Method = "qpso" | "pso" | "alns" | "constructive" | "milp";
export type Draft = {
  id: string;
  name: string;
  description: string;
  networkId: string;
  source: "generated" | "catalog" | "upload";
  graph: Network;
  traffic: number;
  closures: string[];
  method: Method;
  particles: number;
  evaluations: number;
  seed: number;
  updatedAt: string;
};
export type Result = SolveResult & {
  certified?: boolean;
  gap?: number;
  error?: string;
};
export type Comparison = Omit<CompareResult, "rows"> & {
  traces?: Record<string, Result["trace"]>;
  rows: (CompareRow & { status?: string; certified?: boolean; gap?: number })[];
};
export type Run = {
  id: string;
  createdAt: string;
  draft: Draft;
  result: Result;
  comparison?: Comparison;
  provenance: "computed" | "illustrative";
};
export type Company = {
  id: string;
  name: string;
  industry: string;
  role: "owner" | "dispatcher" | "viewer";
};
export const demoCompany: Company = {
  id: "demo",
  name: "Meridian Logistics",
  industry: "Last-mile delivery",
  role: "owner",
};
export const methods: { id: Method; label: string; detail: string }[] = [
  {
    id: "qpso",
    label: "QPSO",
    detail: "Quantum-inspired search with independent route validation.",
  },
  {
    id: "pso",
    label: "Standard PSO",
    detail:
      "Classical particle swarm reference, using the same routing constraints.",
  },
  {
    id: "alns",
    label: "ALNS",
    detail: "Adaptive large neighbourhood search for vehicle routing.",
  },
  {
    id: "milp",
    label: "MILP",
    detail: "Exact reference for small scenarios, up to 12 deliveries.",
  },
  {
    id: "constructive",
    label: "Constructive",
    detail: "Fast feasible starting plan for operational replanning.",
  },
];
export const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value));
export const uid = () => crypto.randomUUID();

export function generateNetwork(
  size = 24,
  count = 12,
  vehicles = 4,
  seed = 7,
): Network {
  const cols = Math.ceil(Math.sqrt(size * 1.5));
  const nodes: Graph["nodes"] = Array.from({ length: size }, (_, i) => ({
    id: `N${i}`,
    x: (i % cols) * 360,
    y: Math.floor(i / cols) * 340,
    kind: i === 0 ? "depot" : "customer_access",
    zone: `Z${Math.floor(i / cols) + 1}`,
  }));
  const edges: Graph["edges"] = [];
  function connect(a: number, b: number) {
    for (const [from, to] of [
      [a, b],
      [b, a],
    ]) {
      const length = Math.hypot(
        nodes[from].x - nodes[to].x,
        nodes[from].y - nodes[to].y,
      );
      edges.push({
        id: `E${from}-${to}`,
        from: `N${from}`,
        to: `N${to}`,
        length_m: length,
        speed_mps: 8 + ((from + to + seed) % 5),
        road_class: "local",
        open: true,
        free_flow_s: length / 10,
      });
    }
  }
  nodes.forEach((_, i) => {
    if (i % cols !== cols - 1 && i + 1 < size) connect(i, i + 1);
    if (i + cols < size) connect(i, i + cols);
  });
  return {
    scenario_id: `SYNTHETIC_${seed}`,
    graph_version: "workspace-v1",
    nodes,
    edges,
    reference_plan: {},
    requests: Array.from({ length: Math.min(count, size - 1) }, (_, i) => ({
      id: `D${String(i + 1).padStart(3, "0")}`,
      node: `N${1 + ((i * 7 + seed) % (size - 1))}`,
      demand: 8 + ((i + seed) % 6) * 2,
      earliest_s: 0,
      latest_s: 14400,
      service_s: 60,
    })),
    fleet: Array.from({ length: vehicles }, (_, i) => ({
      id: `V${i + 1}`,
      capacity: 100,
      depot: "N0",
    })),
  };
}

export function createDraft(
  name = "City distribution",
  graph = generateNetwork(),
): Draft {
  return {
    id: uid(),
    name,
    description: "A distribution scenario for the morning delivery window.",
    networkId: graph.scenario_id,
    source: "generated",
    graph,
    traffic: 1,
    closures: [],
    method: "qpso",
    particles: 12,
    evaluations: 40,
    seed: 7,
    updatedAt: new Date().toISOString(),
  };
}

export function seedDrafts(): Draft[] {
  const base = createDraft("Morning distribution");
  return [
    base,
    {
      ...clone(base),
      id: uid(),
      name: "Rush-hour delivery",
      traffic: 0.55,
      description:
        "Compare the same fleet under a simulated reduction in road speeds.",
    },
    {
      ...clone(base),
      id: uid(),
      name: "Incident response",
      closures: ["E7-8"],
      description:
        "A closed road tests route feasibility and the ability to find a detour.",
    },
  ];
}

export function requestBody(draft: Draft) {
  return {
    graph: draft.graph,
    method: draft.method,
    particles: draft.particles,
    evaluations: draft.evaluations,
    seed: draft.seed,
    traffic_factor: draft.traffic,
    closed_edge_ids: draft.closures,
  };
}

export function validateDraft(draft: Draft): string | null {
  if (draft.name.trim().length < 2)
    return "Give this scenario a name of at least two characters.";
  if (!draft.graph.nodes.length || !draft.graph.edges.length)
    return "Choose or upload a road network.";
  if (!draft.graph.fleet.length || !draft.graph.requests.length)
    return "Add at least one vehicle and delivery.";
  const nodes = new Set(draft.graph.nodes.map((n) => n.id));
  for (const vehicle of draft.graph.fleet) {
    if (
      !Number.isFinite(vehicle.capacity) ||
      vehicle.capacity <= 0 ||
      !nodes.has(vehicle.depot)
    )
      return "Every vehicle needs a positive capacity and a valid depot.";
  }
  if (
    new Set(draft.graph.fleet.map((v) => v.id)).size !==
    draft.graph.fleet.length
  )
    return "Vehicle IDs must be unique.";
  if (
    new Set(draft.graph.requests.map((j) => j.id)).size !==
    draft.graph.requests.length
  )
    return "Delivery IDs must be unique.";
  for (const job of draft.graph.requests) {
    if (!nodes.has(job.node))
      return `Delivery ${job.id} references an unknown road node.`;
    if (
      ![job.demand, job.earliest_s, job.latest_s, job.service_s].every(
        (n) => Number.isFinite(n) && n >= 0,
      )
    )
      return "Delivery quantities and times must be non-negative numbers.";
    if (job.latest_s < job.earliest_s)
      return `Delivery ${job.id}: the window end precedes its start.`;
  }
  if (draft.evaluations < draft.particles)
    return "The evaluation budget must be at least the population size.";
  if (draft.method === "milp" && draft.graph.requests.length > 12)
    return "MILP supports up to 12 deliveries; choose another algorithm.";
  return null;
}

export function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let row: string[] = [],
    field = "",
    quoted = false;
  const input = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < input.length; i++) {
    const c = input[i];
    if (c === '"') {
      if (quoted && input[i + 1] === '"') {
        field += '"';
        i++;
      } else quoted = !quoted;
    } else if (!quoted && (c === "," || c === "\n")) {
      row.push(field.trim());
      field = "";
      if (c === "\n") {
        rows.push(row);
        row = [];
      }
    } else if (c !== "\r" || quoted) field += c;
  }
  if (quoted) throw new Error("CSV contains an unclosed quoted value.");
  row.push(field.trim());
  if (row.some(Boolean)) rows.push(row);
  if (rows.length < 2)
    throw new Error("CSV needs a header and at least one data row.");
  const headers = rows.shift()!;
  return rows
    .filter((r) => r.some(Boolean))
    .map((r, i) => {
      if (r.length !== headers.length)
        throw new Error(`CSV row ${i + 2} has the wrong number of columns.`);
      return Object.fromEntries(headers.map((h, n) => [h.toLowerCase(), r[n]]));
    });
}

export function download(
  name: string,
  text: string,
  type = "application/json",
) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function csvCell(value: unknown) {
  let text = String(value ?? "");
  if (/^[=+@-]/.test(text)) text = `'${text}`;
  return `"${text.replace(/"/g, '""')}"`;
}
export function exportRun(run: Run) {
  const rows = [
    [
      "Vehicle",
      "Stop order",
      "Load",
      "Capacity",
      "Travel time (s)",
      "Feasible",
    ],
    ...run.result.evaluation.vehicles.map((v) => [
      v.id,
      v.order.join(" > "),
      v.load,
      v.capacity,
      v.elapsed_s,
      v.feasible,
    ]),
  ];
  download(
    `quanta-${run.id.slice(0, 8)}.csv`,
    rows.map((row) => row.map(csvCell).join(",")).join("\r\n"),
    "text/csv",
  );
}
export const dateLabel = (value: string) =>
  new Date(value).toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
export const duration = (seconds: number) =>
  seconds >= 3600
    ? `${(seconds / 3600).toFixed(1)} h`
    : seconds >= 60
      ? `${(seconds / 60).toFixed(1)} min`
      : `${seconds.toFixed(1)} s`;
