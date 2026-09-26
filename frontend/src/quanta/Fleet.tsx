import { useEffect, useMemo } from "react";
import { CircleParking, Gauge, LayoutGrid, Package, Truck, X } from "lucide-react";
import { useQuanta } from "./store";
import { Btn, Empty, Pill, Status, fmtEta, fmtKm } from "./ui";
import { routesFromSolve } from "./map";

type FleetStatus = "moving" | "delayed" | "idle";

type Row = {
  id: string;
  color: string;
  capacity: number;
  depot: string;
  status: FleetStatus;
  stops: number;
  distanceM: number | null;
  elapsedS: number | null;
  late: number;
};

const STATUS_LABEL: Record<FleetStatus, { tone: "green" | "red" | "grey"; text: string; hint: string }> = {
  moving: { tone: "green", text: "Moving", hint: "Animated on the map" },
  delayed: { tone: "red", text: "Delayed", hint: "Failed a time window" },
  idle: { tone: "grey", text: "At depot", hint: "No route assigned yet" },
};

// Section 2 helper — "Fleet & Vehicles": every vehicle the current scenario
// knows about, whether it has a plan, and whether it is moving right now.
export function FleetPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { graph, solve, config, scenarioId } = useQuanta();
  const routes = useMemo(() => routesFromSolve(solve), [solve]);

  // Close on Escape.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const rows = useMemo<Row[]>(() => {
    const solved = solve?.evaluation.vehicles || [];
    const byId = new Map(solved.map((v) => [v.id, v]));
    const colorOf = new Map(routes.map((r) => [r.id, r.color]));
    const fleet = graph?.fleet || [];
    const ids = [...new Set([...fleet.map((v) => v.id), ...byId.keys()])];

    return ids.map((id, index) => {
      const spec = fleet.find((v) => v.id === id);
      const plan = byId.get(id);
      const late = (plan?.stops || []).filter((s) => s.window_ok === false).length;
      const assigned = !!plan && !!plan.edge_ids?.length;
      const status: FleetStatus = !assigned ? "idle" : !plan?.feasible || late > 0 ? "delayed" : "moving";
      return {
        id,
        color: colorOf.get(id) || `hsl(${(index * 47) % 360} 60% 55%)`,
        capacity: spec?.capacity ?? 0,
        depot: spec?.depot || "—",
        status,
        stops: plan?.order.length ?? 0,
        distanceM: plan?.distance_m ?? null,
        elapsedS: plan?.elapsed_s ?? null,
        late,
      };
    });
  }, [graph, solve, routes]);

  if (!open) return null;

  const moving = rows.filter((r) => r.status === "moving").length;
  const delayed = rows.filter((r) => r.status === "delayed").length;
  const idle = rows.filter((r) => r.status === "idle").length;
  const capacity = rows.reduce((sum, r) => sum + (r.capacity || 0), 0);
  const stopTotal = rows.reduce((sum, r) => sum + r.stops, 0);
  const planned = rows.some((r) => r.status !== "idle");

  return (
    <div className="q-modal" role="dialog" aria-modal="true" aria-label="Fleet and vehicles" onClick={onClose}>
      <div className="q-modal-card q-fleet" onClick={(e) => e.stopPropagation()}>
        <header className="q-fleet-head">
          <span className="q-stat-icon">
            <Truck />
          </span>
          <div>
            <h2>Fleet &amp; Vehicles</h2>
            <p>
              Every vehicle in <strong>{graph?.scenario_id || scenarioId || "this scenario"}</strong>
              {config.closed.length > 0 && ` · ${config.closed.length} road closure(s) active`}
            </p>
          </div>
          <button className="q-icon-btn" aria-label="Close fleet panel" onClick={onClose}>
            <X size={16} />
          </button>
        </header>

        <div className="q-kpis q-kpis-tight q-fleet-kpis">
          <div className="q-fleet-kpi">
            <LayoutGrid size={15} />
            <strong>{rows.length}</strong>
            <small>vehicles in scenario</small>
          </div>
          <div className="q-fleet-kpi">
            <Gauge size={15} />
            <strong>{moving}</strong>
            <small>moving now</small>
          </div>
          <div className="q-fleet-kpi">
            <CircleParking size={15} />
            <strong>{idle}</strong>
            <small>waiting at depot</small>
          </div>
          <div className="q-fleet-kpi">
            <Package size={15} />
            <strong>{capacity || "—"}</strong>
            <small>total load capacity</small>
          </div>
        </div>

        {rows.length === 0 ? (
          <Empty>No vehicles were returned for this scenario — reload the page or pick another one.</Empty>
        ) : (
          <div className="q-table-scroll">
            <table className="q-table">
              <thead>
                <tr>
                  <th>Vehicle</th>
                  <th>Status</th>
                  <th>Capacity</th>
                  <th>Depot</th>
                  <th>Stops</th>
                  <th>Distance</th>
                  <th>Finish</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const meta = STATUS_LABEL[row.status];
                  return (
                    <tr key={row.id}>
                      <td>
                        <span className="q-fleet-id">
                          <i style={{ background: row.color }} />
                          <span className="q-mono">{row.id}</span>
                        </span>
                      </td>
                      <td>
                        <span title={meta.hint}>
                          <Status tone={meta.tone}>
                            {meta.text}
                            {row.status === "delayed" && row.late > 0 ? ` · ${row.late} late` : ""}
                          </Status>
                        </span>
                      </td>
                      <td>{row.capacity || "—"}</td>
                      <td className="q-mono">{row.depot}</td>
                      <td>{row.stops || "—"}</td>
                      <td>{row.distanceM != null ? fmtKm(row.distanceM) : "—"}</td>
                      <td>{row.elapsedS != null ? fmtEta(row.elapsedS) : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <footer className="q-fleet-foot">
          <span>
            <Pill tone="green" dot>{moving} moving</Pill>
            <Pill tone="amber" dot>{delayed} delayed</Pill>
            <Pill tone="blue" dot>{idle} idle</Pill>
          </span>
          <span className="q-hint">
            {planned
              ? `Plan covers ${stopTotal} stops · vehicles marked “Moving” are the animated markers on the map.`
              : "No plan yet — run Solve Routes and every assigned vehicle starts moving on the map."}
          </span>
          {!planned && <Btn kind="ghost" onClick={onClose}>Back to dashboard</Btn>}
        </footer>
      </div>
    </div>
  );
}
