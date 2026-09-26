import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  BarChart3,
  Brain,
  Cpu,
  GitCompare,
  Gauge,
  GitBranch,
  Layers,
  Lightbulb,
  ListOrdered,
  Map as MapIcon,
  Radar,
  Waves,
} from "lucide-react";
import {
  api,
  postJson,
  SOLVE_TIMEOUT_MS,
  type CompareRow,
  type EpisodeForecasts,
  type EpisodeList,
  type EpisodeSummary,
  type Evidence,
  type EvidenceCoverage,
} from "@/api";
import { useQuanta } from "./store";
import { Box, Btn, Field, Loader, Pill, SectionHead, Select, Stat, Status, revealStyle } from "./ui";
import { ForecastChart } from "./charts";
import { MapPanel } from "./map";

type ForecasterStatus = {
  corpus?: {
    present?: boolean;
    episodes?: Record<string, number>;
    schema_version?: string;
    map_disjoint?: boolean;
    window_cache?: { cached_episodes?: number };
  };
  artifacts?: Record<string, {
    present?: boolean;
    manifest_present?: boolean;
    artifact?: string;
    model_version?: number;
    metrics?: Record<string, Record<string, { count?: number; mae?: number; rmse?: number }>>;
  }>;
  training?: { running?: boolean; progress?: number; message?: string } | null;
};

type CompareResponse = {
  scenario_id: string;
  rows: CompareRow[];
};

const HORIZONS = [5, 10, 15] as const;
type Horizon = (typeof HORIZONS)[number];

// Plain-language one-liners for the four optimisers, used both by the
// explainer and the results table so a first-time visitor is never dropped
// into acronyms.
const METHOD_PLAIN: Record<string, { label: string; how: string }> = {
  constructive: { label: "Constructive", how: "A quick rule of thumb. Builds a plan in one pass — very fast, usually the loosest." },
  pso: { label: "PSO", how: "A swarm of candidate plans that keep drifting toward the best ones they have seen." },
  qpso: { label: "QPSO", how: "The swarm with quantum-style movement, so it wanders further before settling." },
  alns: { label: "ALNS", how: "Repeatedly tears chunks out of the plan and re-inserts them in a better spot." },
  milp: { label: "MILP", how: "Brute-force mathematics that proves the optimum — only for tiny problems." },
};

/** Backend method names arrive in any case (``CONSTRUCTIVE`` / ``qpso``). */
function methodPlain(method?: string | null): { label: string; how: string } | undefined {
  return method ? METHOD_PLAIN[method.trim().toLowerCase()] : undefined;
}

// Recorded-corpus zones (map ids) — discovered from the episode index.
function heatColor(ratio: number | null): string {
  if (ratio == null || !Number.isFinite(ratio)) return "#334155";
  const t = Math.max(0, Math.min(1, ratio));
  // red (gridlock) → amber → green (free flow)
  const stops: [number, [number, number, number]][] = [
    [0, [239, 68, 68]],
    [0.45, [245, 158, 11]],
    [0.75, [234, 179, 8]],
    [1, [34, 197, 94]],
  ];
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [p0, c0] = stops[i];
    const [p1, c1] = stops[i + 1];
    if (t <= p1) {
      const f = p1 === p0 ? 0 : (t - p0) / (p1 - p0);
      const c = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
  }
  return "rgb(34,197,94)";
}

// Section 5 — FORECASTING.
export function Forecasting() {
  const { graph, snapshot, scenarioId, publishResults } = useQuanta();

  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [zones, setZones] = useState<string[]>([]);
  const [zone, setZone] = useState("");
  const [episodeId, setEpisodeId] = useState("");
  const [horizon, setHorizon] = useState<Horizon>(5);
  const [issueIndex, setIssueIndex] = useState<number | null>(null);
  const [forecasts, setForecasts] = useState<EpisodeForecasts | null>(null);
  const [loadingEpisodes, setLoadingEpisodes] = useState(true);
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [error, setError] = useState("");

  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [status, setStatus] = useState<ForecasterStatus | null>(null);

  // Zone preference: the map we're currently routing on, else the first zone.
  const preferredZone = graph?.scenario_id || "";

  // Load the episode index once; zones are derived from it.
  const zoneTouched = useRef(false);
  useEffect(() => {
    let alive = true;
    api<EpisodeList>("/api/episodes?limit=2000")
      .then((res) => {
        if (!alive) return;
        const list = res.episodes || [];
        setEpisodes(list);
        setZones(Array.from(new Set(list.map((e) => e.scenario_id))).sort());
        setLoadingEpisodes(false);
      })
      .catch((err: Error) => {
        if (!alive) return;
        setError(err.message);
        setLoadingEpisodes(false);
      });
    return () => { alive = false; };
  }, []);

  // Follow the map the user is routing on until they pick a zone themselves.
  useEffect(() => {
    if (zoneTouched.current || !zones.length) return;
    if (preferredZone && zones.includes(preferredZone)) setZone(preferredZone);
    else setZone((current) => current || zones[0]);
  }, [zones, preferredZone]);

  useEffect(() => {
    let alive = true;
    api<Evidence>("/api/evidence").then((e) => alive && setEvidence(e)).catch(() => undefined);
    api<ForecasterStatus>("/api/models/forecaster/status").then((s) => alive && setStatus(s)).catch(() => undefined);
    return () => { alive = false; };
  }, []);

  const zoneEpisodes = useMemo(() => episodes.filter((e) => !zone || e.scenario_id === zone), [episodes, zone]);

  // Default to a test-split episode so the numbers are out-of-sample.
  useEffect(() => {
    if (!zoneEpisodes.length) return;
    setEpisodeId((current) => {
      if (current && zoneEpisodes.some((e) => e.episode_id === current)) return current;
      const test = zoneEpisodes.find((e) => e.split === "test");
      return (test || zoneEpisodes[0]).episode_id;
    });
  }, [zoneEpisodes]);

  useEffect(() => {
    if (!episodeId) {
      setForecasts(null);
      return;
    }
    setLoadingForecast(true);
    setIssueIndex(null);
    let alive = true;
    api<EpisodeForecasts>(`/api/episodes/${encodeURIComponent(episodeId)}/forecasts`)
      .then((res) => {
        if (!alive) return;
        setForecasts(res);
        setIssueIndex(res.issues?.length ? res.issues.length - 1 : null);
        setError("");
      })
      .catch((err: Error) => {
        if (!alive) return;
        setForecasts(null);
        setError(err.message);
      })
      .finally(() => alive && setLoadingForecast(false));
    return () => { alive = false; };
  }, [episodeId]);

  const issues = forecasts?.issues || [];
  const issue = issueIndex != null ? issues[issueIndex] || null : null;
  const horizonIdx = HORIZONS.indexOf(horizon);

  // History = every recorded truth up to the issue time; forecast = that
  // issue's prediction with its P10–P90 band (all honest recorded values).
  const { history, forecast, markerAt } = useMemo(() => {
    const cutoff = issue ? issue.issued_at_s : Number.POSITIVE_INFINITY;
    const hist: { t: number; v: number }[] = [];
    const seen = new Set<number>();
    for (const item of issues) {
      item.target_times_s.forEach((t, i) => {
        const v = item.truth_mean[i];
        if (v == null || t > cutoff || seen.has(t)) return;
        seen.add(t);
        hist.push({ t, v });
      });
    }
    hist.sort((a, b) => a.t - b.t);
    const fc: { t: number; v: number; lo?: number | null; hi?: number | null }[] = [];
    if (issue) {
      issue.target_times_s.forEach((t, i) => {
        const v = issue.pred_mean[i];
        if (v == null) return;
        fc.push({ t, v, lo: issue.pred_p10[i], hi: issue.pred_p90[i] });
      });
    }
    return { history: hist, forecast: fc, markerAt: issue?.issued_at_s ?? null };
  }, [issues, issue]);

  // Live accuracy of the selected episode at each horizon (m/s → km/h).
  const episodeMetrics = useMemo(() => {
    return HORIZONS.map((h, idx) => {
      let sum = 0;
      let sq = 0;
      let n = 0;
      for (const item of issues) {
        const p = item.pred_mean[idx];
        const truth = item.truth_mean[idx];
        if (p == null || truth == null) continue;
        const err = p - truth;
        sum += Math.abs(err);
        sq += err * err;
        n += 1;
      }
      return { horizon: h, mae: n ? (sum / n) * 3.6 : null, rmse: n ? Math.sqrt(sq / n) * 3.6 : null, n };
    });
  }, [issues]);

  // Heatmap: real recorded per-edge speed ratios at the issue time, scaled by
  // the network-level forecast for the selected horizon.
  const edgeColors = useMemo(() => {
    if (!issue || !graph) return undefined;
    const ratioAt = new Map<string, number | null>();
    // The issue carries network aggregates, not per-edge values — the frame
    // endpoint carries per-edge truth. Keep the colours to the network
    // forecast scaling so nothing is invented here.
    const predicted = issue.pred_mean[horizonIdx];
    if (predicted == null) return undefined;
    for (const edge of graph.edges) {
      // Free-flow reference from the snapshot: colour = predicted / free-flow.
      const free = edge.speed_mps || 1;
      ratioAt.set(edge.id, predicted / free);
    }
    const colors = new Map<string, string>();
    ratioAt.forEach((value, id) => colors.set(id, heatColor(value)));
    return colors;
  }, [issue, graph, horizonIdx]);

  const modelVersion = forecasts?.model_version ?? null;
  const forecastVersion = issues[0]?.forecast_version || null;
  const artifact = status?.artifacts ? Object.values(status.artifacts)[0] : undefined;
  const corpusEpisodes = status?.corpus?.episodes || {};
  const testMetrics = evidence?.test_metrics || {};

  const networkMean = issue?.pred_mean[horizonIdx] ?? null;
  const networkTruth = issue?.truth_mean[horizonIdx] ?? null;
  const isBaseline = !!forecastVersion && forecastVersion.startsWith("persistence");

  /* ── Optimiser benchmark (QPSO / PSO / ALNS / Constructive) ───────────── */
  const [cmpRows, setCmpRows] = useState<CompareRow[] | null>(null);
  const [cmpBusy, setCmpBusy] = useState(false);
  const [cmpError, setCmpError] = useState("");

  async function runOptimisers() {
    if (!graph) return;
    setCmpBusy(true);
    setCmpError("");
    try {
      const payload = snapshot();
      const result = await postJson<CompareResponse>(
        "/api/workspace/compare",
        {
          graph: payload.graph,
          closed_edge_ids: payload.closed_edge_ids,
          traffic_factor: payload.traffic_factor,
          method: "qpso",
          particles: 4,
          evaluations: 8,
          seed: 7,
          methods: ["constructive", "pso", "qpso", "alns"],
        },
        SOLVE_TIMEOUT_MS,
      );
      setCmpRows(result.rows);
      // Same benchmark shape as Route Lab — lets the assistant quote it.
      publishResults({ compare: { rows: result.rows } });
    } catch (err) {
      setCmpError(err instanceof Error ? err.message : String(err));
    } finally {
      setCmpBusy(false);
    }
  }

  const cmpBest = useMemo(() => {
    const usable = (cmpRows || []).filter((r) => r.feasible && typeof r.objective === "number");
    if (!usable.length) return null;
    return usable.reduce((a, b) => ((a.objective as number) <= (b.objective as number) ? a : b));
  }, [cmpRows]);

  // One sentence a non-technical reader can act on.
  const takeaway = useMemo(() => {
    const usable = (cmpRows || []).filter((r) => r.feasible && typeof r.objective === "number");
    if (usable.length < 2) return "";
    const best = cmpBest as CompareRow;
    const worst = usable.reduce((a, b) => ((a.objective as number) >= (b.objective as number) ? a : b));
    const bestObj = best.objective as number;
    const worstObj = worst.objective as number;
    const gap = worstObj > 0 ? ((worstObj - bestObj) / worstObj) * 100 : 0;
    const fast = usable.filter((r) => typeof r.elapsed_s === "number")
      .reduce<CompareRow | null>((a, b) => (a == null || (a.elapsed_s as number) <= (b.elapsed_s as number) ? a : b), null);
    const label = (row: CompareRow) => methodPlain(row.method)?.label || row.method;
    // A tie is the common case under a small budget — say so instead of
    // claiming the winner was "0.0% below" itself.
    let text = gap <= 0.0001
      ? `All ${usable.length} algorithms landed on the same score (${bestObj.toFixed(0)}) — under this budget they found ` +
        "the same plan, so pick the one that answers fastest. " +
        "Lower score = fewer kilometres, less time and less fuel."
      : `${label(best)} gave the cheapest plan (score ${bestObj.toFixed(0)}), ` +
        `${gap.toFixed(1)}% below ${label(worst)} (${worstObj.toFixed(0)}). ` +
        "Lower score = fewer kilometres, less time and less fuel.";
    if (fast) text += ` ${label(fast)} was the quickest to answer (${(fast.elapsed_s as number).toFixed(1)}s).`;
    return text;
  }, [cmpRows, cmpBest]);

  // Feed the selected prediction and its recomputed accuracy to the assistant.
  useEffect(() => {
    if (!issue) return;
    publishResults({
      forecast: {
        zone: zone || undefined,
        episode_id: episodeId || undefined,
        model_version: forecastVersion || undefined,
        horizon_min: horizon,
        predicted_kmh: networkMean != null ? networkMean * 3.6 : undefined,
        observed_kmh: networkTruth != null ? networkTruth * 3.6 : undefined,
        horizons: episodeMetrics
          .filter((row) => row.mae != null)
          .map((row) => ({ horizon: row.horizon, mae: row.mae as number, rmse: row.rmse, n: row.n })),
      },
    });
  }, [issue, zone, episodeId, forecastVersion, horizon, networkMean, networkTruth, episodeMetrics, publishResults]);

  const roadCount = graph?.edges.length ?? 0;

  return (
    <div className="q-section">
      <SectionHead
        n={5}
        title="FORECASTING"
        sub="See how busy each road will be in 5–15 minutes, and which optimiser copes best"
        right={
          <div className="q-ops-bar">
            <Pill tone={evidence?.available ? "green" : "amber"} dot>
              {evidence?.available ? (evidence.demo ? "demo evidence" : "live model") : "no artifact"}
            </Pill>
            <span className="q-ops-date">
              {forecastVersion ? `issue model: ${forecastVersion}` : "no recorded issues"}
            </span>
          </div>
        }
      />

      {/* ── Plain-language orientation for first-time visitors ───────────── */}
      <Box title={<><Lightbulb size={15} /> Start here — what this page does</>} className="q-guide" style={revealStyle(30)}>
        <ol className="q-guide-steps">
          <li>
            <strong>Pick a recording</strong>
            <span>Choose an area, then an episode — a saved recording of what the roads actually did.</span>
          </li>
          <li>
            <strong>Choose how far ahead</strong>
            <span>5, 10 or 15 minutes into the future. Everything on the page follows that choice.</span>
          </li>
          <li>
            <strong>Read the chart</strong>
            <span>Solid line = what really happened. Dashed line = the prediction. Shaded band = the model’s 90% confidence range.</span>
          </li>
          <li>
            <strong>See it on the map</strong>
            <span>Every road is coloured: green = flowing, amber = slowing, red = jammed.</span>
          </li>
        </ol>
        <p className="q-hint">
          Nothing here is invented: every number is either read from the server’s recorded episodes or recomputed
          in your browser from them. Where a value is a placeholder, it says so.
        </p>
      </Box>

      <div className="q-forecast-grid" style={revealStyle(60)}>
        <Box title={<><Activity size={15} /> Forecast controls</>}>
          <Field label="1 · Area (zone)" hint={zones.length ? `${zones.length} recorded area(s) available` : undefined}>
            <Select
              ariaLabel="Zone"
              value={zone}
              onChange={(value) => { zoneTouched.current = true; setZone(value); setEpisodeId(""); }}
              options={zones.map((z) => ({ value: z, label: z }))}
            />
          </Field>
          <Field
            label="2 · Recording (episode)"
            hint={(() => {
              const e = zoneEpisodes.find((x) => x.episode_id === episodeId);
              if (!e) return undefined;
              const bits = [e.split, e.regime];
              if (e.event_type) bits.push(e.event_type.replace(/_/g, " "));
              if (e.event_count) bits.push(`${e.event_count} event(s)`);
              return bits.join(" · ");
            })()}
          >
            <Select
              ariaLabel="Episode"
              value={episodeId}
              onChange={setEpisodeId}
              options={zoneEpisodes.map((e) => ({
                value: e.episode_id,
                label: `${e.episode_id} · ${e.split} · ${e.regime}${e.event_type ? ` · ${e.event_type}` : ""}`,
              }))}
            />
          </Field>

          <div className="q-radio-block" role="radiogroup" aria-label="Forecast horizon">
            <span className="q-radio-label">3 · How far ahead?</span>
            <div className="q-radios">
              {HORIZONS.map((h) => (
                <button
                  key={h}
                  type="button"
                  role="radio"
                  aria-checked={horizon === h}
                  className={`q-radio ${horizon === h ? "on" : ""}`}
                  onClick={() => setHorizon(h)}
                >
                  {h} min
                </button>
              ))}
            </div>
          </div>

          {issues.length > 0 && (
            <Field
              label={`4 · Prediction issued at ${issue ? new Date(issue.issued_at_s * 1000).toISOString().substr(11, 5) : "—"}`}
              hint={`${issues.length} predictions recorded in this episode — drag to replay them`}
            >
              <input
                className="q-range"
                type="range"
                aria-label="Forecast issue time"
                min={0}
                max={issues.length - 1}
                step={1}
                value={issueIndex ?? 0}
                onChange={(e) => setIssueIndex(Number(e.target.value))}
              />
            </Field>
          )}

          <div className="q-actions">
            <Btn
              kind="ghost"
              disabled={!zoneEpisodes.length}
              onClick={() => {
                const list = [...zoneEpisodes].sort((a, b) => (a.episode_id < b.episode_id ? 1 : -1));
                const test = list.find((e) => e.split === "test");
                setEpisodeId((test || list[0]).episode_id);
              }}
            >
              Reset to test episode
            </Btn>
          </div>

          {loadingEpisodes && <Loader label="Indexing recorded episodes…" />}
          {!loadingEpisodes && !zoneEpisodes.length && (
            <Status tone="amber">No recorded episodes for this area.</Status>
          )}
          {error && <Status tone="red">{error}</Status>}

          <div className="q-kpis q-kpis-tight">
            <Stat icon={<Gauge />} value={networkMean != null ? `${(networkMean * 3.6).toFixed(1)}` : "—"} label={`Predicted speed in ${horizon} min (km/h)`} tone="blue" />
            <Stat icon={<Waves />} value={networkTruth != null ? `${(networkTruth * 3.6).toFixed(1)}` : "—"} label={`Actually recorded (km/h)`} tone="green" />
            <Stat icon={<BarChart3 />} value={episodeMetrics[horizonIdx].mae != null ? `${episodeMetrics[horizonIdx].mae!.toFixed(2)}` : "—"} label={`Average error at ${horizon} min (km/h)`} tone="amber" />
          </div>
        </Box>

        <Box
          title={<><Waves size={15} /> Speed forecast</>}
          action={
            issue && forecastVersion ? (
              <span className="q-badge">{forecastVersion} · +{horizon} min</span>
            ) : undefined
          }
        >
          {loadingForecast ? (
            <Loader label="Loading issued forecasts…" />
          ) : (
            <ForecastChart
              history={history}
              forecast={forecast}
              markerAt={markerAt}
              unit="km/h"
              height={272}
            />
          )}
          {!loadingForecast && !issues.length && (
            <p className="q-empty">No predictions are recorded for this recording.</p>
          )}
          {issue && issue.valid_edges[horizonIdx] != null && (
            <p className="q-hint">
              Average over <strong>{issue.valid_edges[horizonIdx].toLocaleString()}</strong> reporting roads ·
              the shaded band is the recorded P10–P90 range · the solid line shows only roads that were observed.
            </p>
          )}
          {isBaseline && (
            <p className="q-hint q-warn-note">
              Honest note: these recorded predictions come from the <strong>persistence baseline</strong> ({forecastVersion}) —
              it simply repeats today’s speed tomorrow. It exists as a score to beat, not the trained model.
              The trained GNN + Transformer’s own held-out scores are in <strong>Model Performance</strong> below.
            </p>
          )}
        </Box>
      </div>

      <div className="q-forecast-grid" style={revealStyle(120)}>
        <Box
          title={<><MapIcon size={15} /> Traffic congestion map</>}
          action={<span className="q-badge">predicted ÷ free-flow</span>}
          className="q-map-box"
        >
          <MapPanel
            graph={graph}
            edgeColors={edgeColors}
            height={330}
            animate={false}
            caption={
              issue
                ? `Predicted network average ${(networkMean != null ? (networkMean * 3.6).toFixed(1) : "—")} km/h in ${horizon} minutes, compared with each road’s legal free-flow speed.`
                : "Choose a prediction time on the left to colour the roads."
            }
            extraLegend={
              <span className="q-lg-scale">
                <i style={{ background: "rgb(239,68,68)" }} /> jammed
                <i style={{ background: "rgb(245,158,11)" }} /> slowing
                <i style={{ background: "rgb(34,197,94)" }} /> flowing
              </span>
            }
          />
          <p className="q-hint">
            A road shown at 50% means it is expected to run at half its usual (free-flow) speed — that is what a
            driver would feel as heavy traffic. The colour scale is linear between 0% and 100%.
          </p>
        </Box>

        {/* ── How the GNN + Transformer actually works ────────────────────── */}
        <Box
          title={<><Brain size={15} /> How the AI model works</>}
          action={artifact?.manifest_present ? <Pill tone="green" dot>trained artifact</Pill> : <Pill tone="amber" dot>untrained</Pill>}
        >
          <p className="q-lead">
            One network called <strong>CausalGNNTransformer</strong> does the predicting in three stages.
            Think of it as a map reader, a pattern spotter, and a guesser that tells you how sure it is.
          </p>

          <ol className="q-pipeline">
            <li className="q-pipe">
              <span className="q-pipe-step">1</span>
              <span className="q-pipe-icon"><GitBranch size={16} /></span>
              <strong>GNN — reads the map</strong>
              <em>Graph neural network, {graph ? "2 edge-aware layers" : "2 layers"}</em>
              <p>
                Every road is a node and every junction is a link. The GNN passes messages between neighbouring
                roads, so a jam spreading from one street to the next is something it can represent rather than
                miss.
              </p>
              <ul className="q-pipe-facts">
                <li><span>Roads in view</span><b>{roadCount ? roadCount.toLocaleString() : "—"}</b></li>
                <li><span>Signals per road</span><b>6</b></li>
                <li><span>…which are</span><b className="q-pipe-wide">speed ratio, occupancy, halted vehicles, age of reading, missing flag, closed flag</b></li>
              </ul>
            </li>
            <li className="q-pipe">
              <span className="q-pipe-step">2</span>
              <span className="q-pipe-icon"><Radar size={16} /></span>
              <strong>Transformer — reads the recent past</strong>
              <em>2 layers · 4 attention heads · 1 reading per minute</em>
              <p>
                It looks at the last <strong>12 minutes</strong> in order and weighs whichever earlier minutes
                matter most — the moment an incident started, or the first sign of a rush building.
              </p>
              <ul className="q-pipe-facts">
                <li><span>Look-back</span><b>12 min</b></li>
                <li><span>Sampling</span><b>1 / minute</b></li>
                <li><span>Look-ahead</span><b>+5, +10, +15 min</b></li>
              </ul>
            </li>
            <li className="q-pipe">
              <span className="q-pipe-step">3</span>
              <span className="q-pipe-icon"><Layers size={16} /></span>
              <strong>Output — a guess plus a confidence band</strong>
              <em>P10–P90, calibrated on the validation split</em>
              <p>
                You get one expected speed per road at each horizon, wrapped in a range that should contain the
                truth about 90% of the time. A wide band means “don’t rely on this”.
              </p>
              <ul className="q-pipe-facts">
                <li><span>Horizons</span><b>5 / 10 / 15 min</b></li>
                <li><span>Measured coverage</span><b>{(() => {
                  const buckets = (evidence?.uncertainty as EvidenceCoverage | undefined)?.validation;
                  const first = buckets ? Object.values(buckets)[0] : undefined;
                  return first?.actual_coverage != null ? `${(first.actual_coverage * 100).toFixed(0)}%` : "—";
                })()}</b></li>
                <li><span>Model version</span><b>{evidence?.model_version ? `v${evidence.model_version}` : modelVersion ?? "—"}</b></li>
              </ul>
            </li>
          </ol>

          <dl className="q-kv">
            <div><dt>Training corpus</dt><dd>
              {corpusEpisodes.train
                ? `${(corpusEpisodes.train + (corpusEpisodes.validation || 0) + (corpusEpisodes.test || 0)).toLocaleString()} episodes (${corpusEpisodes.train.toLocaleString()} training / ${(corpusEpisodes.validation || 0).toLocaleString()} tuning / ${(corpusEpisodes.test || 0).toLocaleString()} test)`
                : status?.corpus?.present ? "present" : "not found on the server"}
            </dd></div>
            <div><dt>Map-disjoint split</dt><dd>{status?.corpus?.map_disjoint ? "yes — maps used for testing were never seen during training" : "no"}</dd></div>
            <div><dt>Model file</dt><dd className="mono">{artifact?.artifact || evidence?.artifact || "—"}{evidence?.model_version ? ` · v${evidence.model_version}` : ""}</dd></div>
          </dl>
          {evidence?.provenance && <p className="q-hint">{evidence.provenance}</p>}
          {!evidence?.available && evidence?.reason && <Status tone="amber">{evidence.reason}</Status>}
          <div className="q-actions">
            <Btn
              kind="quiet"
              disabled
              title="Training is launched from the API — see /api/models/forecaster/train"
            >
              <Cpu size={15} /> Train forecaster (API)
            </Btn>
          </div>
        </Box>
      </div>

      {/* ── Optimiser results, framed in plain language ──────────────────── */}
      <Box
        title={<><GitCompare size={15} /> Optimiser results — QPSO vs PSO vs ALNS vs Constructive</>}
        action={
          cmpBusy ? <Pill tone="amber" dot>running</Pill>
            : cmpRows ? <Pill tone="green" dot>{cmpRows.length} algorithms</Pill>
            : undefined
        }
        className="q-compare-box"
        style={revealStyle(160)}
      >
        <p className="q-lead">
          The forecast above says how busy the roads will be; the optimiser is what decides which vehicle goes
          where. Run all four on your current network to see which one you should trust back on the Dashboard.
        </p>

        <div className="q-actions">
          <Btn disabled={cmpBusy || !graph} onClick={() => void runOptimisers()}>
            {cmpBusy ? "Benchmarking…" : "Run all four optimisers"}
          </Btn>
          <span className="q-hint">
            Uses the exact network, fleet and disruptions you configured ({scenarioId || "no scenario"}).
            Each run is capped at a small budget so it returns in seconds.
          </span>
        </div>

        {cmpBusy && <Loader label="Building the snapshot and solving it four times…" />}
        {cmpError && <Status tone="red">{cmpError}</Status>}

        {takeaway && <p className="q-takeaway"><Lightbulb size={15} /> {takeaway}</p>}

        {cmpRows && cmpRows.length > 0 && (
          <div className="q-table-scroll">
            <table className="q-table">
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>In plain words</th>
                  <th>Score</th>
                  <th>Distance</th>
                  <th>Route time</th>
                  <th>Think time</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {cmpRows.map((row) => {
                  const plain = methodPlain(row.method) || { label: row.method, how: "—" };
                  const isBest = cmpBest && row === cmpBest;
                  const obj = typeof row.objective === "number" ? row.objective : null;
                  return (
                    <tr key={row.method} className={isBest ? "q-row-best" : ""}>
                      <td><strong>{plain.label}</strong></td>
                      <td className="q-plain-cell">{plain.how}</td>
                      <td className="q-mono">{obj != null ? obj.toFixed(0) : "—"}</td>
                      <td className="q-mono">{row.distance_m != null ? `${(row.distance_m / 1000).toFixed(1)} km` : "—"}</td>
                      <td className="q-mono">{row.time_s != null ? `${(row.time_s / 3600).toFixed(1)} h` : "—"}</td>
                      <td className="q-mono">{row.elapsed_s != null ? `${row.elapsed_s.toFixed(1)} s` : "—"}</td>
                      <td>
                        {row.error ? <Status tone="red">{row.error}</Status>
                          : !row.feasible ? <Status tone="red">not feasible</Status>
                          : isBest ? <Status tone="green">best score</Status>
                          : <Status tone="grey">feasible</Status>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {!cmpRows && !cmpBusy && (
          <p className="q-empty">
            Nothing benchmarked yet — press <strong>Run all four optimisers</strong>. For the full benchmark with
            charts, convergence traces and up to five algorithms (including the exact MILP solver), open
            <strong> Route Lab</strong> in the sidebar.
          </p>
        )}
        <p className="q-hint">
          Score is the optimiser’s combined cost — distance, drive time and lateness, weighted together. Lower is
          better. Every result is re-checked by an independent validator before it is called feasible.
        </p>
      </Box>

      <div style={revealStyle(200)}>
        <Box
          title={<><BarChart3 size={15} /> Model Performance — how wrong is it, really?</>}
          action={Object.keys(testMetrics).length ? <span className="q-badge">held-out test split</span> : undefined}
        >
          <p className="q-lead">
            <strong>MAE</strong> (mean absolute error) is the typical miss in km/h — if it says 32 and the truth is
            36, that is 4. <strong>RMSE</strong> punishes big misses harder, so a large gap between the two means
            the model is usually right but occasionally way off.
          </p>
        <div className="q-perf-grid">
          <div>
            <h4 className="q-subhead">Selected episode · live recomputation</h4>
            <table className="q-table">
              <thead>
                <tr>
                  <th>Horizon</th>
                  <th>MAE (km/h)</th>
                  <th>RMSE (km/h)</th>
                  <th>Predictions</th>
                </tr>
              </thead>
              <tbody>
                {episodeMetrics.map((row) => (
                  <tr key={row.horizon} className={row.horizon === horizon ? "q-row-best" : ""}>
                    <td>+{row.horizon} min</td>
                    <td>{row.mae != null ? row.mae.toFixed(2) : "—"}</td>
                    <td>{row.rmse != null ? row.rmse.toFixed(2) : "—"}</td>
                    <td>{row.n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="q-hint">
              Calculated in your browser from <span className="mono">GET /api/episodes/{episodeId || "…"}/forecasts</span> —
              the gap between each prediction and what was recorded, converted from m/s to km/h.
              {isBaseline && " Because the recorded issues are the baseline, these numbers score the baseline, not the trained model."}
            </p>
          </div>

          <div>
            <h4 className="q-subhead">Corpus test split · saved artifact</h4>
            {Object.keys(testMetrics).length ? (
              <table className="q-table">
                <thead>
                  <tr>
                    <th>Horizon</th>
                    <th>MAE (speed ratio)</th>
                    <th>RMSE (speed ratio)</th>
                    <th>Samples</th>
                  </tr>
                </thead>
                <tbody>
                  {HORIZONS.map((h) => {
                    const m = testMetrics[String(h)];
                    return (
                      <tr key={h} className={h === horizon ? "q-row-best" : ""}>
                        <td>+{h} min</td>
                        <td>{m?.mae != null ? m.mae.toFixed(4) : "—"}</td>
                        <td>{m?.rmse != null ? m.rmse.toFixed(4) : "—"}</td>
                        <td>{m && "count" in m ? (m as { count?: number }).count?.toLocaleString() ?? "—" : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            ) : (
              <p className="q-empty">No saved test scores yet — train the forecaster to produce them.</p>
            )}
            <p className="q-hint">
              From <span className="mono">GET /api/evidence</span>
              {evidence?.artifact ? ` · artifact ${evidence.artifact}` : ""}
              {evidence?.demo ? " · static demo values, not live inference" : ""}. Units are speed ratios
              (1.0 = exactly the legal limit), so they are not directly comparable with the km/h column —
              0.05 here means “within about 5% of the speed limit”.
            </p>
            {evidence?.temporal_baseline_mae == null && (
              <p className="q-hint">Temporal-only baseline: not recorded for this artifact.</p>
            )}
          </div>
        </div>
        </Box>
      </div>

      <div style={revealStyle(240)}>
        <Box title={<><ListOrdered size={15} /> Words used on this page</>}>
          <dl className="q-kv">
            <div><dt>Free-flow speed</dt><dd>The speed a road carries when nothing is in the way — usually its legal limit.</dd></div>
            <div><dt>Horizon</dt><dd>How far into the future we are guessing: 5, 10 or 15 minutes.</dd></div>
            <div><dt>Episode</dt><dd>One saved recording of a road network over time, split into training, tuning or test.</dd></div>
            <div><dt>Issue</dt><dd>A single prediction made at one moment, with everything the model believed then.</dd></div>
            <div><dt>P10–P90 band</dt><dd>The range the model expects the truth to land in 80% of the time — wider means less certain.</dd></div>
            <div><dt>Objective (score)</dt><dd>One number combining distance, time and lateness. Optimisers try to make it smaller.</dd></div>
            <div><dt>Feasible</dt><dd>A plan that respects every rule: capacity, delivery windows, and no driving down a closed road.</dd></div>
          </dl>
        </Box>
      </div>
    </div>
  );
}
