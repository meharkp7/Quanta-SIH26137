import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Clock,
  GitBranch,
  PlayCircle,
  Route as RouteIcon,
  Truck,
} from "lucide-react";
import type { SolveResult } from "@/api";
import { useQuanta } from "./store";
import { Box, Btn, Field, Loader, Pill, SectionHead, Select, Slider, Stat, Status, fmtEta, fmtHours, fmtKm, revealStyle } from "./ui";
import { ObjectiveBars } from "./charts";
import { MapPanel, ROUTE_COLORS, routesFromSolve } from "./map";

// Section 3 — WHAT-IF SCENARIOS.
export function WhatIf() {
  const { graph, config, patch, scopes, runSolve, solving, markReplanned, replanned, solveNote, scenarioId, publishResults } = useQuanta();

  const [baseline, setBaseline] = useState<SolveResult | null>(null);
  const [disrupted, setDisrupted] = useState<SolveResult | null>(null);
  const [busy, setBusy] = useState<"" | "apply" | "replan">("");
  const [error, setError] = useState("");

  const hasDisruption =
    config.trafficPct > 0 || config.congestionPct > 0 || config.delayMin > 0 || config.closed.length > 0;

  // Candidate roads for the closure picker: longest roads read as "corridors".
  const closureCandidates = useMemo(() => {
    if (!graph) return [];
    return [...graph.edges]
      .sort((a, b) => b.length_m - a.length_m)
      .slice(0, 60)
      .map((edge) => ({
        value: edge.id,
        label: `${edge.id} · ${edge.road_class} · ${Math.round(edge.length_m)} m`,
      }));
  }, [graph]);

  function toggleClosure(id: string) {
    const closed = config.closed.includes(id) ? config.closed.filter((c) => c !== id) : [...config.closed, id];
    patch({ closed: closed.slice(0, 25) });
  }

  async function applyScenario() {
    setError("");
    setBusy("apply");
    try {
      // Baseline ignores every disruption so the delta is attributable to the scenario.
      const before = await runSolve({ baseline: true, adopt: false, quiet: true, label: "Baseline solve…" });
      if (!before) throw new Error("The baseline solve did not return a plan.");
      const after = hasDisruption
        ? await runSolve({ adopt: false, quiet: true, label: "Disrupted solve…" })
        : before;
      if (!after) throw new Error("The disrupted solve did not return a plan.");
      setBaseline(before);
      setDisrupted(after);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("");
    }
  }

  async function replan() {
    setError("");
    setBusy("replan");
    try {
      const result = await runSolve({ label: "Replanning with scenario…" });
      if (!result) throw new Error("Replan did not return a plan.");
      setBaseline(null);
      setDisrupted(null);
      markReplanned();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("");
    }
  }

  // ── Impact deltas ───────────────────────────────────────────────────────
  const impact = useMemo(() => {
    if (!baseline || !disrupted) return null;
    const before = new Map(baseline.evaluation.vehicles.map((v) => [v.id, v]));
    const after = disrupted.evaluation.vehicles;
    let affected = 0;
    let extraStops = 0;
    for (const vehicle of after) {
      const prior = before.get(vehicle.id);
      const same = prior && prior.edge_ids.join("|") === vehicle.edge_ids.join("|");
      if (!same) affected += 1;
      extraStops += Math.max(0, vehicle.order.length - (prior?.order.length || 0));
    }
    const dTime = disrupted.evaluation.time_s - baseline.evaluation.time_s;
    const dDistance = disrupted.evaluation.distance_m - baseline.evaluation.distance_m;
    const dObjective = disrupted.evaluation.objective - baseline.evaluation.objective;
    const pct = (delta: number, of: number) => (of === 0 ? 0 : (delta / Math.abs(of)) * 100);
    return {
      affected,
      unserved: Math.max(0, baseline.evaluation.vehicles.reduce((s, v) => s + v.order.length, 0)
        - disrupted.evaluation.vehicles.reduce((s, v) => s + v.order.length, 0)),
      extraStops,
      dTime,
      dDistance,
      dObjective,
      timePct: pct(dTime, baseline.evaluation.time_s),
      distancePct: pct(dDistance, baseline.evaluation.distance_m),
      objectivePct: pct(dObjective, baseline.evaluation.objective),
    };
  }, [baseline, disrupted]);

  // Share baseline-vs-disrupted numbers with the experiment assistant so it
  // can answer "how much did this closure cost us?" with real deltas.
  useEffect(() => {
    if (!impact || !baseline || !disrupted) return;
    publishResults({
      whatif: {
        baseline: {
          objective: baseline.evaluation.objective,
          total_distance_m: baseline.evaluation.distance_m,
          total_time_s: baseline.evaluation.time_s,
        },
        disrupted: {
          objective: disrupted.evaluation.objective,
          total_distance_m: disrupted.evaluation.distance_m,
          total_time_s: disrupted.evaluation.time_s,
        },
        affected_vehicles: impact.affected,
        unserved: impact.unserved,
      },
    });
  }, [impact, baseline, disrupted, publishResults]);

  const comparisonRows = useMemo(() => {
    if (!impact) return [];
    return [
      { method: "Baseline plan", objective: Math.max(1, baseline!.evaluation.objective), feasible: baseline!.evaluation.feasible },
      { method: hasDisruption ? "With disruption" : "Current plan", objective: Math.max(1, disrupted!.evaluation.objective), feasible: disrupted!.evaluation.feasible },
    ];
  }, [impact, baseline, disrupted, hasDisruption]);

  const routeRows = useMemo(() => {
    if (!baseline || !disrupted) return [];
    const before = new Map(baseline.evaluation.vehicles.map((v) => [v.id, v]));
    return disrupted.evaluation.vehicles.map((vehicle) => {
      const prior = before.get(vehicle.id);
      const changed = !prior || prior.edge_ids.join("|") !== vehicle.edge_ids.join("|");
      return {
        id: vehicle.id,
        stopsBefore: prior?.order.length || 0,
        stopsAfter: vehicle.order.length,
        distBefore: prior?.distance_m || 0,
        distAfter: vehicle.distance_m || 0,
        timeBefore: prior?.elapsed_s || 0,
        timeAfter: vehicle.elapsed_s,
        changed,
        feasible: vehicle.feasible,
      };
    });
  }, [baseline, disrupted]);

  const baselineRoutes = useMemo(() => routesFromSolve(baseline), [baseline]);
  const disruptedRoutes = useMemo(() => routesFromSolve(disrupted), [disrupted]);

  return (
    <div className="q-section">
      <SectionHead
        n={3}
        title="WHAT-IF SCENARIOS"
        sub="Model disruptions, measure the impact, replan with confidence"
        right={
          <div className="q-ops-bar">
            <Pill tone={hasDisruption ? "amber" : "green"} dot>{hasDisruption ? "scenario armed" : "network nominal"}</Pill>
            <span className="q-ops-date">{scenarioId}</span>
          </div>
        }
      />

      <div className="q-whatif-grid" style={revealStyle(60)}>
        <Box title={<><AlertTriangle size={15} /> Disruption Controls</>} className="q-disruption-box">
          <Slider
            label="Traffic increase"
            value={config.trafficPct}
            min={0}
            max={80}
            step={5}
            unit="%"
            onChange={(value) => patch({ trafficPct: value })}
          />
          <p className="q-hint">
            Scales every road speed by 1 ÷ (1 + {config.trafficPct / 100}) — {config.trafficPct === 0 ? "no change applied" : `speeds at ${(100 / (1 + config.trafficPct / 100)).toFixed(0)}% of free flow`}.
          </p>

          <Field
            label="Congestion scope"
            hint={scopes.length ? "Slowdown applies only inside the selected zone or road class." : "No zones on this map."}
          >
            <Select
              ariaLabel="Congestion scope"
              value={config.congestionScope}
              onChange={(value) => patch({ congestionScope: value })}
              options={[
                { value: "", label: "Whole network" },
                ...scopes.map((scope) => ({ value: scope.key, label: scope.label })),
              ]}
            />
          </Field>
          <Slider
            label={config.congestionScope ? "Congestion in scope" : "Network congestion"}
            value={config.congestionPct}
            min={0}
            max={80}
            step={5}
            unit="%"
            onChange={(value) => patch({ congestionPct: value })}
            disabled={!config.congestionScope && !scopes.length}
          />
          <p className="q-hint">
            {config.congestionScope
              ? `${scopes.find((s) => s.key === config.congestionScope)?.edges || 0} road(s) slow by ${config.congestionPct}%.`
              : "Pick a scope to confine the slowdown to one zone or road class."}
          </p>

          <Slider
            label="Vehicle delay per stop"
            value={config.delayMin}
            min={0}
            max={30}
            step={1}
            unit=" min"
            onChange={(value) => patch({ delayMin: value })}
          />
          <p className="q-hint">Added to every delivery's service time — models loading docks and driver handovers.</p>

          <Field
            label={`Road closures (${config.closed.length} selected)`}
            hint="Closures remove the road from the snapshot; the solver must route around it."
          >
            <div className="q-closure-list">
              {config.closed.map((id) => (
                <button key={id} type="button" className="q-chip on danger" onClick={() => toggleClosure(id)} title="Remove closure">
                  {id} ✕
                </button>
              ))}
              <Select
                ariaLabel="Add road closure"
                value=""
                onChange={(value) => value && toggleClosure(value)}
                options={[{ value: "", label: "Add a road closure…" }, ...closureCandidates.filter((c) => !config.closed.includes(c.value))]}
              />
            </div>
          </Field>

          <div className="q-actions">
            <Btn disabled={busy !== "" || solving || !graph} onClick={() => void applyScenario()}>
              {busy === "apply" ? "Running…" : "Apply Scenario"}
            </Btn>
            <Btn kind="ghost" disabled={busy !== "" || solving || !graph || !hasDisruption} onClick={() => void replan()}>
              {busy === "replan" ? "Replanning…" : "Replan with Scenario"}
            </Btn>
          </div>
          {(busy !== "" || solving) && <Loader label={solveNote || "Solving routes on the backend snapshot…"} />}
          {error && <Status tone="red">{error}</Status>}
          {replanned && busy === "" && (
            <Status tone="green">Replanned — the dashboard now runs on this disrupted scenario.</Status>
          )}
          {!hasDisruption && busy === "" && (
            <p className="q-hint">Raise traffic, add congestion, delay vehicles or close a road, then press Apply Scenario.</p>
          )}
        </Box>

        <div className="q-whatif-panels">
          <Box
            title="Before / After network"
            action={impact ? <span className="q-badge">{impact.affected} vehicle(s) rerouted</span> : <span className="q-badge">awaiting scenario</span>}
            className="q-map-box"
          >
            <div className="q-before-after">
              <figure>
                <figcaption><span className="q-dot q-dot-baseline" /> Baseline plan</figcaption>
                <MapPanel
                  graph={graph}
                  routes={baselineRoutes}
                  height={252}
                  animate={false}
                  caption={baseline ? `${fmtKm(baseline.evaluation.distance_m)} · ${fmtHours(baseline.evaluation.time_s)}` : "Press Apply Scenario"}
                />
              </figure>
              <figure>
                <figcaption><span className="q-dot q-dot-disrupted" /> Disrupted plan</figcaption>
                <MapPanel
                  graph={graph}
                  routes={disruptedRoutes}
                  closed={config.closed}
                  height={252}
                  animate={false}
                  caption={disrupted ? `${fmtKm(disrupted.evaluation.distance_m)} · ${fmtHours(disrupted.evaluation.time_s)}` : "Press Apply Scenario"}
                />
              </figure>
            </div>
          </Box>
        </div>
      </div>

      {impact && (
        <>
          <div className="q-kpis" style={revealStyle(120)}>
            <Stat icon={<Truck />} value={impact.affected} label="Affected vehicles" tone="amber" />
            <Stat
              icon={<Clock />}
              value={`${impact.dTime >= 0 ? "+" : "−"}${fmtHours(Math.abs(impact.dTime))}`}
              label={`Extra time (${impact.timePct >= 0 ? "+" : "−"}${Math.abs(impact.timePct).toFixed(1)}%)`}
              tone="lime"
            />
            <Stat
              icon={<RouteIcon />}
              value={`${impact.dDistance >= 0 ? "+" : "−"}${fmtKm(Math.abs(impact.dDistance))}`}
              label={`Extra distance (${impact.distancePct >= 0 ? "+" : "−"}${Math.abs(impact.distancePct).toFixed(1)}%)`}
              tone="blue"
            />
            <Stat icon={<Activity />} value={`${impact.dObjective >= 0 ? "+" : "−"}${Math.abs(impact.dObjective).toFixed(0)}`} label="Objective delta" tone="green" />
            <Stat icon={<GitBranch />} value={impact.unserved} label="Deliveries unserved" tone={impact.unserved > 0 ? "red" : "green"} />
          </div>

          <div className="q-whatif-lower" style={revealStyle(180)}>
            <Box title="Impact Summary" action={hasDisruption ? <Pill tone="amber" dot>disruption applied</Pill> : <Pill tone="green" dot>no disruption set</Pill>}>
              <p className="q-lead">
                {impact.affected === 0
                  ? "Every vehicle keeps its original corridor — the scenario does not change the optimal plan."
                  : `${impact.affected} of ${disrupted!.evaluation.vehicles.length} vehicles must reroute to keep the plan feasible.`}
                {impact.extraStops > 0 && ` ${impact.extraStops} additional stop(s) appear in the disrupted plan.`}
              </p>
              <div className="q-impact-bars">
                <ObjectiveBars rows={comparisonRows} />
              </div>
              <table className="q-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Disrupted</th>
                    <th>Delta</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Total time</td>
                    <td>{fmtHours(baseline!.evaluation.time_s)}</td>
                    <td>{fmtHours(disrupted!.evaluation.time_s)}</td>
                    <td className={impact.dTime > 0 ? "q-bad" : "q-good"}>{impact.dTime > 0 ? "+" : ""}{fmtHours(impact.dTime)}</td>
                  </tr>
                  <tr>
                    <td>Total distance</td>
                    <td>{fmtKm(baseline!.evaluation.distance_m)}</td>
                    <td>{fmtKm(disrupted!.evaluation.distance_m)}</td>
                    <td className={impact.dDistance > 0 ? "q-bad" : "q-good"}>{impact.dDistance > 0 ? "+" : ""}{fmtKm(impact.dDistance)}</td>
                  </tr>
                  <tr>
                    <td>Objective</td>
                    <td>{baseline!.evaluation.objective.toFixed(0)}</td>
                    <td>{disrupted!.evaluation.objective.toFixed(0)}</td>
                    <td className={impact.dObjective > 0 ? "q-bad" : "q-good"}>{impact.dObjective > 0 ? "+" : ""}{impact.dObjective.toFixed(0)}</td>
                  </tr>
                  <tr>
                    <td>Validator</td>
                    <td>{baseline!.evaluation.feasible ? "feasible" : "issues"}</td>
                    <td>{disrupted!.evaluation.feasible ? "feasible" : "issues"}</td>
                    <td>{disrupted!.evaluation.feasible ? "still feasible" : "needs attention"}</td>
                  </tr>
                </tbody>
              </table>
            </Box>

            <Box title="Route Comparison" action={<span className="q-badge">{routeRows.filter((r) => r.changed).length} changed</span>}>
              <table className="q-table">
                <thead>
                  <tr>
                    <th>Vehicle</th>
                    <th>Stops</th>
                    <th>Distance</th>
                    <th>ETA</th>
                    <th>Change</th>
                  </tr>
                </thead>
                <tbody>
                  {routeRows.map((row) => (
                    <tr key={row.id}>
                      <td className="q-mono">{row.id}</td>
                      <td>
                        {row.stopsBefore}
                        {row.stopsAfter !== row.stopsBefore && (
                          <span className="q-delta"> <ArrowRight size={11} /> {row.stopsAfter}</span>
                        )}
                      </td>
                      <td>
                        {fmtKm(row.distBefore)}
                        {row.distAfter !== row.distBefore && (
                          <span className={`q-delta ${row.distAfter > row.distBefore ? "up" : "down"}`}>
                            <ArrowRight size={11} /> {fmtKm(row.distAfter)}
                          </span>
                        )}
                      </td>
                      <td>{fmtEta(row.timeAfter)}</td>
                      <td>
                        <Status tone={row.changed ? "amber" : "grey"}>
                          {row.changed ? "rerouted" : "unchanged"}
                        </Status>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="q-actions q-actions-end">
                <Btn kind="ghost" disabled={busy !== "" || solving || !hasDisruption} onClick={() => void replan()}>
                  <PlayCircle size={15} /> Replan with Scenario
                </Btn>
              </div>
              <p className="q-hint">
                Route colours match the map: {ROUTE_COLORS.slice(0, 4).map((c, i) => (
                  <span key={c} className="q-inline-swatch"><i style={{ background: c }} /> Route {i + 1}</span>
                ))}
              </p>
            </Box>
          </div>
        </>
      )}

      {!impact && !busy && (
        <p className="q-empty" style={revealStyle(120)}>
          Press <strong>Apply Scenario</strong> to solve the network twice — once clean, once with your disruptions — and see
          exactly which vehicles reroute, how much time and distance the incident costs, and whether the plan stays feasible.
        </p>
      )}
    </div>
  );
}
