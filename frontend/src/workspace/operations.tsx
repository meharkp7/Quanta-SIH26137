import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useSearchParams } from "react-router-dom";
import {
  Activity,
  ArrowLeftRight,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Download,
  MapPin,
  Pause,
  Play,
  Route,
  ShieldCheck,
  TrafficCone,
  Truck,
  X,
} from "lucide-react";
import {
  api,
  EpisodeDetail,
  EpisodeFrame,
  EpisodeList,
  EpisodeSummary,
  Graph,
  PathResult,
  postJson,
  ReplayFrame,
  ReplayResult,
  ScenarioSummary,
  SOLVE_TIMEOUT_MS,
} from "../api";
import type { EpisodeEdgeObs } from "../GeoMap";
import { clone, Draft, duration, download, Network, requestBody, Run } from "./data";
import { useWorkspace } from "./Store";
import { useRunner } from "./Runner";
import {
  Button,
  Chart,
  Empty,
  Field,
  Metric,
  NetworkView,
  Notice,
  PageTitle,
  Panel,
  RouteList,
  Tag,
} from "./ui";

// Recorded Delhi corpus map opened by default when the control room is
// entered directly (/control-room) — the simulation map boots straight onto
// a real network with its recorded episodes feeding the map.
const CORPUS_DEFAULT = "DELHI_CP";

// On-demand Draft for a recorded corpus map. Graph windows are backfilled
// with generous defaults (recorded truth rides on the map layer; a plan is
// only computed if the operator asks for one). Never persisted to the
// store — corpus graphs can be megabytes.
function corpusDraft(id: string, graph: Graph): Draft {
  return {
    id,
    name: id,
    description: "Recorded Delhi OSM network from the SUMO corpus.",
    networkId: graph.scenario_id,
    source: "catalog",
    graph: {
      ...graph,
      requests: graph.requests.map((r) => ({
        ...r,
        earliest_s: (r as { earliest_s?: number }).earliest_s ?? 0,
        latest_s: (r as { latest_s?: number }).latest_s ?? 14400,
        service_s: (r as { service_s?: number }).service_s ?? 60,
      })),
    } as Network,
    traffic: 1,
    closures: [],
    method: "qpso",
    particles: 12,
    evaluations: 40,
    seed: 7,
    updatedAt: new Date().toISOString(),
  };
}

function useSelectedDraft() {
  const store = useWorkspace(),
    [params, setParams] = useSearchParams(),
    location = useLocation();
  const selected =
    params.get("scenario") ||
    (location.pathname === "/control-room"
      ? CORPUS_DEFAULT
      : store.drafts[0]?.id || "");
  const source = store.drafts.find((d) => d.id === selected);
  const [draft, setDraft] = useState<Draft | null>(() =>
    source ? clone(source) : null,
  );
  // Recorded corpus maps (Delhi OSM) listed alongside the workspace drafts;
  // their graphs load on demand so localStorage never holds them.
  const [corpus, setCorpus] = useState<ScenarioSummary[]>([]);
  const [loadingCorpus, setLoadingCorpus] = useState(() =>
    selected.startsWith("DELHI_"),
  );
  const [corpusError, setCorpusError] = useState("");
  useEffect(() => {
    api<{ scenarios: ScenarioSummary[] }>("/api/scenarios")
      .then((res) =>
        setCorpus(
          (res.scenarios || []).filter(
            (s) => s.available && s.id.startsWith("DELHI_"),
          ),
        ),
      )
      .catch(() => setCorpus([]));
  }, []);
  useEffect(() => {
    const found = store.drafts.find((d) => d.id === selected);
    setCorpusError("");
    if (found) {
      setLoadingCorpus(false);
      setDraft(clone(found));
      return;
    }
    if (!selected.startsWith("DELHI_")) {
      setLoadingCorpus(false);
      setDraft(null);
      return;
    }
    let alive = true;
    setLoadingCorpus(true);
    setDraft(null);
    api<Graph>(`/api/scenarios/${encodeURIComponent(selected)}`)
      .then((graph) => {
        if (!alive) return;
        setDraft(corpusDraft(selected, graph));
        setLoadingCorpus(false);
      })
      .catch((err) => {
        if (!alive) return;
        setDraft(null);
        setLoadingCorpus(false);
        setCorpusError((err as Error).message);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);
  return {
    draft,
    setDraft,
    selected,
    choose: (id: string) => setParams({ scenario: id }),
    store,
    corpus,
    loadingCorpus,
    corpusError,
  };
}
export function Operations({ controlRoom = false }: { controlRoom?: boolean } = {}) {
  const {
    draft,
    setDraft,
    selected,
    choose,
    store,
    corpus,
    loadingCorpus,
    corpusError,
  } = useSelectedDraft();
  const runner = useRunner();
  const [current, setCurrent] = useState<Run | null>(
    () => store.runs.find((r) => r.draft.id === selected) || null,
  );
  const [previous, setPrevious] = useState<Run | null>(null),
    [selectedVehicle, setSelectedVehicle] = useState("");
  const [replay, setReplay] = useState<ReplayResult | null>(null),
    [frame, setFrame] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1);
  const [error, setError] = useState(""),
    [loadingReplay, setLoadingReplay] = useState(false),
    [auto, setAuto] = useState(true),
    [tour, setTour] = useState<number | null>(null);
  const autoTimer = useRef<ReturnType<typeof setTimeout>>();
  const selectionRef = useRef(selected);
  selectionRef.current = selected;
  const busy = runner.job?.status === "running";

  // ── Recorded episode feed ─────────────────────────────────────────────
  // Which recorded episodes exist for the loaded network (keyed on the
  // corpus scenario_id, e.g. dwarka_sector12-map003 — workspace fixtures
  // match nothing → row hidden). Feeds observed road truth + fleet movement
  // onto the Leaflet map.
  const scenarioKey = draft?.graph.scenario_id ?? "";
  const [episodes, setEpisodes] = useState<EpisodeSummary[]>([]);
  const [epSel, setEpSel] = useState("");
  const [epDetail, setEpDetail] = useState<EpisodeDetail | null>(null);
  const [epFrame, setEpFrame] = useState<EpisodeFrame | null>(null);
  const [epT, setEpT] = useState(0);
  const [epPlaying, setEpPlaying] = useState(false);
  const epFrameSeq = useRef(0);
  useEffect(() => {
    setEpisodes([]);
    setEpSel("");
    setEpDetail(null);
    setEpFrame(null);
    setEpT(0);
    setEpPlaying(false);
    if (!scenarioKey) return;
    let alive = true;
    api<EpisodeList>(
      `/api/episodes?scenario_id=${encodeURIComponent(scenarioKey)}&limit=200`,
    )
      .then((res) => {
        if (!alive) return;
        setEpisodes(res.episodes || []);
        if (res.episodes?.length) setEpSel(res.episodes[0].episode_id);
      })
      .catch(() => {
        if (alive) setEpisodes([]);
      });
    return () => {
      alive = false;
    };
  }, [scenarioKey]);
  // Episode detail → default playhead lands mid-episode where traffic is
  // richest, and playback starts so fleet movement shows immediately.
  useEffect(() => {
    if (!epSel) {
      setEpDetail(null);
      setEpFrame(null);
      setEpT(0);
      setEpPlaying(false);
      return;
    }
    let alive = true;
    api<EpisodeDetail>(`/api/episodes/${epSel}`)
      .then((detail) => {
        if (!alive) return;
        setEpDetail(detail);
        setEpT(
          detail.times[Math.floor(detail.times.length / 2)] ??
            detail.times[0] ??
            0,
        );
        setEpPlaying(detail.times.length > 1);
      })
      .catch(() => {
        if (alive) {
          setEpDetail(null);
          setEpSel("");
        }
      });
    return () => {
      alive = false;
    };
  }, [epSel]);
  // Frame fetch: debounced so scrubbing doesn't stampede the backend,
  // sequence-guarded so a stale response can't overwrite a newer playhead
  // (frames arrive in ~6ms warm, 35ms cold).
  useEffect(() => {
    if (!epSel || !epT) {
      setEpFrame(null);
      return;
    }
    const seq = ++epFrameSeq.current;
    const timer = window.setTimeout(() => {
      api<EpisodeFrame>(`/api/episodes/${epSel}/frame?t=${epT}`)
        .then((f) => {
          if (epFrameSeq.current === seq) setEpFrame(f);
        })
        .catch(() => {
          if (epFrameSeq.current === seq) setEpFrame(null);
        });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [epSel, epT]);
  // Playback: one recorded timestep per tick, looping back to the first
  // recorded time at the end.
  useEffect(() => {
    if (!epPlaying || !epDetail?.times.length) return;
    const timer = window.setInterval(() => {
      setEpT((value) => {
        const times = epDetail.times;
        const i = times.indexOf(value);
        return i >= 0 && i < times.length - 1 ? times[i + 1] : times[0];
      });
    }, 600);
    return () => window.clearInterval(timer);
  }, [epPlaying, epDetail]);
  // Derived episode state: aligns frame arrays to canonical edge ids and
  // places recorded vehicles on the geo map — fleet cars become movers with
  // an edge-bearing heading, background trajectories become small dots.
  const epView = useMemo(() => {
    if (!epDetail || !epFrame || !draft) return null;
    const edgeObs = new Map<string, EpisodeEdgeObs>();
    epDetail.edge_ids.forEach((id, i) => {
      edgeObs.set(id, {
        ratio: epFrame.ratio[i] ?? null,
        observed: epFrame.observed[i] ?? null,
        closed: epFrame.closed[i] === 1,
      });
    });
    const nodeById = new Map(draft.graph.nodes.map((n) => [n.id, n]));
    const edgeById = new Map(draft.graph.edges.map((e) => [e.id, e]));
    const movers: ReplayFrame["vehicles"] = [];
    const dots: { id: string; lat: number; lon: number }[] = [];
    for (const v of epFrame.vehicles) {
      const edge = edgeById.get(v.edge);
      const a = edge && nodeById.get(edge.from);
      const b = edge && nodeById.get(edge.to);
      if (!edge || !a || !b || a.lat == null || b.lat == null || a.lon == null || b.lon == null) continue;
      const lat = a.lat + ((b.lat as number) - a.lat) * v.pct;
      const lon = (a.lon as number) + ((b.lon as number) - a.lon) * v.pct;
      if (v.kind === "bg") {
        if (dots.length < 400) dots.push({ id: v.id, lat, lon });
        continue;
      }
      const midLat = ((a.lat + b.lat) / 2) * (Math.PI / 180);
      const east = ((b.lon as number) - a.lon) * Math.cos(midLat);
      const north = (b.lat as number) - a.lat;
      const heading =
        Math.abs(east) + Math.abs(north) > 1e-9
          ? (Math.atan2(-north, east) * 180) / Math.PI
          : 0;
      movers.push({ id: v.id, x: 0, y: 0, lat, lon, kind: v.kind, heading });
    }
    return { t: epFrame.t, edgeObs, dots, movers, summary: epFrame.summary };
  }, [epDetail, epFrame, draft]);

  useEffect(() => {
    setCurrent(store.runs.find((r) => r.draft.id === selected) || null);
    setPrevious(null);
    setReplay(null);
    setPlaying(false);
    setFrame(0);
    setSelectedVehicle("");
    clearTimeout(autoTimer.current);
  }, [selected]);
  useEffect(() => () => clearTimeout(autoTimer.current), []);
  useEffect(() => {
    if (!playing || !replay?.frames.length) return;
    const timer = setInterval(
      () =>
        setFrame((n) => {
          if (n >= replay.frames.length - 1) {
            setPlaying(false);
            return n;
          }
          return Math.min(n + speed, replay.frames.length - 1);
        }),
      100,
    );
    return () => clearInterval(timer);
  }, [playing, replay, speed]);
  async function compute(next = draft) {
    if (!next) return;
    setError("");
    setPlaying(false);
    setReplay(null);
    const selection = selectionRef.current;
    try {
      const run = await runner.start(next);
      if (selectionRef.current !== selection) return;
      if (run) {
        setPrevious(current);
        setCurrent(run);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }
  function closure(edge: string) {
    if (!draft || busy || !store.canEdit) return;
    const next = {
      ...draft,
      closures: draft.closures.includes(edge)
        ? draft.closures.filter((e) => e !== edge)
        : [...draft.closures, edge],
    };
    setDraft(next);
    setReplay(null);
    setPlaying(false);
    clearTimeout(autoTimer.current);
    if (auto && current)
      autoTimer.current = setTimeout(() => compute(next), 650);
  }
  async function loadReplay() {
    if (!current || !draft) return;
    setError("");
    setLoadingReplay(true);
    const selection = selectionRef.current;
    try {
      const data = await postJson<ReplayResult>(
        "/api/workspace/replay",
        { ...requestBody(current.draft), plan: current.result.plan },
        SOLVE_TIMEOUT_MS,
      );
      if (selectionRef.current !== selection) return;
      setReplay(data);
      setFrame(0);
      setPlaying(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoadingReplay(false);
    }
  }
  if (loadingCorpus)
    return <div className="w-loading">Loading recorded Delhi network…</div>;
  if (!draft)
    return corpusError ? (
      <Empty
        title="Couldn’t load the recorded map"
        action={
          <Link className="w-button primary" to="/app/live">
            Back to live operations
          </Link>
        }
      >
        {corpusError}
      </Empty>
    ) : (
      <Empty
        title="Set up your first operation"
        action={
          <Link className="w-button primary" to="/app/scenarios/new">
            Create a scenario
          </Link>
        }
      >
        Choose a network, fleet and delivery requirements to start.
      </Empty>
    );
  const result = current?.result,
    frames = replay?.frames || [],
    active = frames[frame];
  const changed =
    current &&
    JSON.stringify(requestBody(current.draft)) !==
      JSON.stringify(requestBody(draft));
  const steps = [
    [
      "Start with a scenario",
      "This workspace contains connected fleet, demand and network data. Inspect the map and scenario controls before running.",
    ],
    [
      "Compute a baseline",
      "Run the selected optimiser on the current network. The validator checks the returned plan.",
    ],
    [
      "Watch the fleet",
      "Load a kinematic simulation of the computed routes. Use the timeline to inspect positions.",
    ],
    [
      "Introduce a disruption",
      "Close a directed road on the map. With auto replan enabled, Quanta computes a new plan.",
    ],
    [
      "Inspect the change",
      "Compare the previous and current objective below, inspect vehicle routes, then open the full result.",
    ],
    [
      "Compare the methods",
      "Open Comparisons to benchmark algorithms against the same scenario snapshot.",
    ],
  ];
  return (
    <>
      <PageTitle
        eyebrow={controlRoom ? "CONTROL ROOM" : "OPERATIONS CENTRE"}
        title={controlRoom ? "Control room" : "Live operations"}
        description={
          controlRoom
            ? "The simulation map — recorded Delhi episodes drive road truth and fleet movement."
            : "Plan your fleet, inspect movement and respond to changing road conditions."
        }
        action={
          <>
            <Button variant="secondary" onClick={() => setTour(0)}>
              Guided walkthrough
            </Button>
            <Button disabled={busy || !store.canEdit} onClick={() => compute()}>
              <Play size={16} />
              {busy ? "Computing…" : "Optimise routes"}
            </Button>
          </>
        }
      />
      <div className="w-operations-toolbar">
        <div>
          <span className="w-label">SCENARIO</span>
          <select
            aria-label="Operation scenario"
            value={selected}
            disabled={busy}
            onChange={(e) => choose(e.target.value)}
          >
            <optgroup label="Workspace scenarios">
              {store.drafts.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </optgroup>
            {corpus.length > 0 && (
              <optgroup label="Delhi recorded maps">
                {corpus.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
        </div>
        <div>
          <span className="w-label">ALGORITHM</span>
          <select
            aria-label="Operation algorithm"
            value={draft.method}
            disabled={busy || !store.canEdit}
            onChange={(e) =>
              setDraft({ ...draft, method: e.target.value as Draft["method"] })
            }
          >
            {[
              "qpso",
              "pso",
              "alns",
              "constructive",
              ...(draft.graph.requests.length <= 12 ? ["milp"] : []),
            ].map((m) => (
              <option key={m} value={m}>
                {m.toUpperCase()}
              </option>
            ))}
          </select>
        </div>
        <span className="w-toolbar-divider" />
        <Tag tone="blue">
          {draft.source === "generated"
            ? "Synthetic network"
            : draft.source === "catalog"
              ? "Recorded OSM network"
              : "Scenario network"}
        </Tag>
        <Tag tone={draft.closures.length ? "amber" : "green"}>
          {draft.closures.length
            ? `${draft.closures.length} closures`
            : "Roads open"}
        </Tag>
        <label className="w-switch">
          <input
            type="checkbox"
            checked={auto}
            onChange={(e) => setAuto(e.target.checked)}
          />
          Auto replan
        </label>
      </div>
      {error && <Notice tone="red">{error}</Notice>}
      {runner.job?.status === "error" && (
        <Notice tone="red">{runner.job.error}</Notice>
      )}
      {changed && !busy && (
        <Notice tone="amber">
          Inputs changed since the displayed result. Optimise again to validate
          this configuration.
        </Notice>
      )}
      <div className="w-metrics">
        <Metric
          label="Planned deliveries"
          value={
            result
              ? `${result.evaluation.vehicles.reduce((s, v) => s + v.order.length, 0)} / ${draft.graph.requests.length}`
              : draft.graph.requests.length
          }
          detail={
            result ? "Assigned in the computed plan" : "Ready for optimisation"
          }
          icon={<Truck size={18} />}
        />
        <Metric
          label="Route distance"
          value={
            result
              ? `${(result.evaluation.distance_m / 1000).toFixed(2)} km`
              : "—"
          }
          detail="Total planned fleet distance"
          icon={<Route size={18} />}
          tone="teal"
        />
        <Metric
          label="Fleet travel time"
          value={result ? duration(result.evaluation.time_s) : "—"}
          detail="Aggregate route evaluation"
          icon={<Clock3 size={18} />}
          tone="amber"
        />
        <Metric
          label="Constraint status"
          value={
            result
              ? result.evaluation.feasible
                ? "Validated"
                : "Review"
              : "Not run"
          }
          detail={
            result
              ? `${result.method} · ${result.elapsed_s.toFixed(2)} s compute`
              : "Independent route validator"
          }
          icon={<ShieldCheck size={18} />}
          tone="purple"
        />
      </div>
      <div className="w-operation-grid">
        <Panel
          title="Fleet operations map"
          subtitle={
            busy
              ? "Computing a new plan against the selected inputs…"
              : epView
                ? "Recorded episode playback — observed truth from the SUMO corpus."
                : "Select a road to stage a closure. Route colours identify vehicles."
          }
          action={
            <Tag tone={epView ? "green" : replay ? "amber" : "blue"}>
              {epView
                ? "Recorded episode"
                : replay
                  ? "Simulation playback"
                  : result
                    ? "Computed routes"
                    : "Network preview"}
            </Tag>
          }
          className="w-map-panel"
        >
          <NetworkView
            graph={draft.graph}
            result={result}
            closed={active?.closed || draft.closures}
            movers={epView ? epView.movers : active?.vehicles || []}
            selected={selectedVehicle}
            selectedVehicle={selectedVehicle}
            onSelect={setSelectedVehicle}
            onCloseRoad={busy || !store.canEdit ? undefined : closure}
            episode={
              epView
                ? { t: epView.t, edgeObs: epView.edgeObs, dots: epView.dots }
                : null
            }
          />
          {episodes.length > 0 && (
            <div className="w-episode">
              <select
                aria-label="Recorded episode"
                value={epSel}
                onChange={(e) => setEpSel(e.target.value)}
              >
                {episodes.map((ep) => (
                  <option key={ep.episode_id} value={ep.episode_id}>
                    {ep.episode_id} · {ep.split} ·{" "}
                    {ep.regime.replace(/_/g, " ")}
                    {ep.event_count
                      ? ` · ${ep.event_count} event${ep.event_count > 1 ? "s" : ""} (${ep.event_type})`
                      : " · no events"}
                  </option>
                ))}
              </select>
              <Button
                variant="secondary"
                disabled={!epDetail?.times.length}
                onClick={() => setEpPlaying((v) => !v)}
              >
                {epPlaying ? <Pause size={15} /> : <Play size={15} />}
                {epPlaying ? "Pause episode" : "Play episode"}
              </Button>
              <span>
                {epDetail
                  ? `t=${epT}s / ${epDetail.duration_s}s`
                  : "loading episode…"}
              </span>
              <input
                aria-label="Episode timeline"
                type="range"
                min={epDetail?.times[0] ?? 0}
                max={epDetail?.duration_s ?? 0}
                step={epDetail?.interval_s ?? 60}
                value={epT}
                disabled={!epDetail}
                onChange={(e) => {
                  setEpPlaying(false);
                  setEpT(Number(e.target.value));
                }}
              />
              {epFrame && (
                <span>
                  {epFrame.summary.reporting}/{epFrame.summary.total} roads ·
                  mean{" "}
                  {epFrame.summary.mean_ratio != null
                    ? `${Math.round(epFrame.summary.mean_ratio * 100)}%`
                    : "—"}{" "}
                  · {epFrame.summary.fleet} fleet · {epFrame.summary.bg}{" "}
                  background
                </span>
              )}
            </div>
          )}
          <div className="w-playback">
            <Button
              variant="secondary"
              disabled={
                loadingReplay ||
                busy ||
                !result?.evaluation.feasible ||
                !!changed ||
                !store.canEdit
              }
              onClick={() => (replay ? setPlaying(!playing) : loadReplay())}
            >
              {playing ? <Pause size={15} /> : <Play size={15} />}
              {loadingReplay
                ? "Loading…"
                : replay
                  ? playing
                    ? "Pause"
                    : "Play"
                  : "Load playback"}
            </Button>
            <span>{active ? duration(active.t) : "Simulation timeline"}</span>
            <input
              aria-label="Simulation timeline"
              type="range"
              min={0}
              max={Math.max(0, frames.length - 1)}
              value={frame}
              disabled={!frames.length}
              onChange={(e) => {
                setPlaying(false);
                setFrame(Number(e.target.value));
              }}
            />
            <select
              aria-label="Playback speed"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
            >
              {[1, 2, 4].map((s) => (
                <option key={s} value={s}>
                  {s}×
                </option>
              ))}
            </select>
          </div>
          <div className="w-map-footnote">
            {episodes.length > 0
              ? "Recorded SUMO episode — road speeds, closures and vehicle positions are recorded truth. Recorded closures are display-only and never trigger a re-plan."
              : replay
                ? "Kinematic simulation of validated road paths. Vehicle positions are simulated, not GPS telemetry."
                : "Road speeds come from the scenario; the configured traffic factor is applied during routing."}
          </div>
        </Panel>
        <div className="w-operation-side">
          <Panel
            title="Vehicle routes"
            subtitle={`${draft.graph.fleet.length} vehicles in this scenario`}
            action={<Truck size={18} />}
          >
            {result ? (
              <RouteList
                result={result}
                selected={selectedVehicle}
                onSelect={setSelectedVehicle}
              />
            ) : (
              <div className="w-available-fleet">
                {draft.graph.fleet.map((v, i) => (
                  <div key={v.id}>
                    <span className="w-fleet-number">0{i + 1}</span>
                    <span>
                      <strong>{v.id}</strong>
                      <small>
                        {v.capacity} units · depot {v.depot}
                      </small>
                    </span>
                    <Tag>Ready</Tag>
                  </div>
                ))}
              </div>
            )}
          </Panel>
          <Panel title="Incident management" action={<TrafficCone size={18} />}>
            <div className="w-form-body">
              {draft.closures.length ? (
                <div className="w-incident-list">
                  {draft.closures.map((e) => (
                    <div key={e}>
                      <TrafficCone size={17} />
                      <span>
                        <strong>Road closure</strong>
                        <small>{e} · active from start</small>
                      </span>
                      <button
                        aria-label={`Reopen ${e}`}
                        disabled={busy || !store.canEdit}
                        onClick={() => closure(e)}
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="w-clear-state">
                  <CheckCircle2 size={24} />
                  <strong>No selected incidents</strong>
                  <p>Select a directed road on the map to test a disruption.</p>
                </div>
              )}
            </div>
          </Panel>
        </div>
      </div>
      <Panel
        title="Stage a road incident"
        subtitle="Choose the exact road direction to close or reopen."
      >
        <div className="w-form-body">
          <Field label="Directed road">
            <select
              aria-label="Stage a road closure"
              value=""
              disabled={busy || !store.canEdit}
              onChange={(e) => {
                if (e.target.value) closure(e.target.value);
              }}
            >
              <option value="">Select a road…</option>
              {draft.graph.edges.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.id} · {e.from} → {e.to}
                  {draft.closures.includes(e.id) ? " · Reopen" : " · Close"}
                </option>
              ))}
            </select>
          </Field>
        </div>
      </Panel>
      <div className="w-two-columns">
        <Panel
          title="Optimisation trace"
          subtitle="Actual best-objective history from the latest swarm run."
        >
          <Chart
            series={[
              {
                name: result?.method || "QPSO",
                values: result?.trace?.best || [],
              },
            ]}
          />
        </Panel>
        <Panel
          title="Decision record"
          subtitle="Inputs, validation and change impact for this operation."
        >
          <div className="w-form-body">
            <div className="w-event">
              <span>1</span>
              <div>
                <strong>Observe the scenario</strong>
                <p>
                  {draft.graph.nodes.length} nodes ·{" "}
                  {draft.graph.requests.length} deliveries ·{" "}
                  {Math.round(draft.traffic * 100)}% road speed factor.
                </p>
              </div>
            </div>
            <div className="w-event">
              <span>2</span>
              <div>
                <strong>
                  {result
                    ? `${result.method} route computation`
                    : "Waiting for computation"}
                </strong>
                <p>
                  {result
                    ? `${result.evaluations ?? "—"} evaluations. ${result.evaluation.feasible ? "All hard constraints passed." : "Inspect the violations in Results."}`
                    : "Run the optimiser to get a validated plan."}
                </p>
              </div>
            </div>
            {previous && result && (
              <div className="w-event">
                <span>3</span>
                <div>
                  <strong>Change from previous run</strong>
                  <p>
                    Objective {previous.result.evaluation.objective.toFixed(1)}{" "}
                    → {result.evaluation.objective.toFixed(1)}. Delta{" "}
                    {(
                      result.evaluation.objective -
                      previous.result.evaluation.objective
                    ).toFixed(1)}
                    .
                  </p>
                </div>
              </div>
            )}
            {current && (
              <Link
                className="w-button secondary"
                to={`/app/results/${current.id}`}
              >
                Inspect full result <ArrowRight size={15} />
              </Link>
            )}
          </div>
        </Panel>
      </div>
      {tour !== null && (
        <div className="w-tour">
          <button
            className="w-tour-close"
            aria-label="Close walkthrough"
            onClick={() => setTour(null)}
          >
            <X size={18} />
          </button>
          <span className="w-eyebrow">
            GUIDED WALKTHROUGH · {tour + 1} / {steps.length}
          </span>
          <h3>{steps[tour][0]}</h3>
          <p>{steps[tour][1]}</p>
          <div className="w-tour-actions">
            {tour === 1 && (
              <Button disabled={busy} onClick={() => compute()}>
                Compute baseline
              </Button>
            )}
            {tour === 2 && (
              <Button
                disabled={!result?.evaluation.feasible || loadingReplay}
                onClick={loadReplay}
              >
                Load simulation
              </Button>
            )}
            {tour === 5 && (
              <Link
                className="w-button primary"
                to={`/app/comparisons?scenario=${draft.id}`}
              >
                Open comparisons
              </Link>
            )}
            <Button
              variant="secondary"
              onClick={() =>
                setTour(tour === steps.length - 1 ? null : tour + 1)
              }
            >
              {tour === steps.length - 1 ? "Finish" : "Next"}
              <ArrowRight size={14} />
            </Button>
          </div>
        </div>
      )}
    </>
  );
}

export function ShortestPath() {
  const { draft, setDraft, selected, choose, store } = useSelectedDraft();
  const [from, setFrom] = useState(draft?.graph.nodes[0]?.id || ""),
    [to, setTo] = useState(draft?.graph.nodes.at(-1)?.id || "");
  const [path, setPath] = useState<PathResult | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [pick, setPick] = useState("from");
  useEffect(() => {
    setFrom(draft?.graph.nodes[0]?.id || "");
    setTo(draft?.graph.nodes.at(-1)?.id || "");
    setPath(null);
  }, [selected, draft?.graph.scenario_id]);
  async function compute() {
    if (!draft) return;
    if (from === to) {
      setError("Choose different origin and destination nodes.");
      return;
    }
    setBusy(true);
    setError("");
    setPath(null);
    try {
      const result = await postJson<PathResult>(
        "/api/workspace/path",
        { ...requestBody(draft), source: from, target: to },
        SOLVE_TIMEOUT_MS,
      );
      setPath(result);
      if (!result.feasible)
        setError(
          result.error || "No directed path is available under these closures.",
        );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  if (!draft)
    return (
      <Empty
        title="A network is needed first"
        action={<Link to="/app/scenarios/new">Create scenario</Link>}
      >
        Select a road network before finding a path.
      </Empty>
    );
  return (
    <>
      <PageTitle
        eyebrow="POINT-TO-POINT ROUTING"
        title="Shortest path"
        description="Find the least-travel-time directed path between two nodes with Dijkstra’s algorithm."
        action={
          <select
            aria-label="Path scenario"
            disabled={busy}
            value={selected}
            onChange={(e) => choose(e.target.value)}
          >
            {store.drafts.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        }
      />
      {error && <Notice tone="red">{error}</Notice>}
      <div className="w-path-layout">
        <Panel
          title="Trip configuration"
          subtitle="Choose endpoints or select nodes on the schematic map."
        >
          <div className="w-form-body">
            <Field label="Origin">
              <select
                value={from}
                onChange={(e) => {
                  setFrom(e.target.value);
                  setPath(null);
                }}
              >
                {draft.graph.nodes.map((n) => (
                  <option key={n.id}>{n.id}</option>
                ))}
              </select>
            </Field>
            <Button
              variant="ghost"
              onClick={() => {
                setFrom(to);
                setTo(from);
                setPath(null);
              }}
            >
              <ArrowLeftRight size={16} />
              Swap endpoints
            </Button>
            <Field label="Destination">
              <select
                value={to}
                onChange={(e) => {
                  setTo(e.target.value);
                  setPath(null);
                }}
              >
                {draft.graph.nodes.map((n) => (
                  <option key={n.id}>{n.id}</option>
                ))}
              </select>
            </Field>
            <Field label="Map selection">
              <select value={pick} onChange={(e) => setPick(e.target.value)}>
                <option value="from">Set origin</option>
                <option value="to">Set destination</option>
              </select>
            </Field>
            <Button disabled={busy || !store.canEdit} onClick={compute}>
              <Route size={16} />
              {busy ? "Finding path…" : "Find shortest path"}
            </Button>
            <div className="w-summary-row">
              <span>Active closures</span>
              <strong>{draft.closures.length}</strong>
            </div>
            {path?.feasible && (
              <>
                <Tag tone="green">Exact directed shortest path</Tag>
                <div className="w-path-stats">
                  <strong>
                    {((path.distance_m || 0) / 1000).toFixed(2)} km
                  </strong>
                  <span>{duration(path.time_s || 0)} travel time</span>
                  <small>{path.edge_ids?.length} road segments</small>
                </div>
                <Button
                  variant="secondary"
                  onClick={() => {
                    const data = JSON.stringify(path, null, 2);
                    download("quanta-shortest-path.json", data);
                  }}
                >
                  <Download size={16} />
                  Export path
                </Button>
              </>
            )}
          </div>
        </Panel>
        <Panel
          title={`${from} → ${to}`}
          subtitle="The highlighted path respects road direction and selected closures."
          action={<Tag tone="blue">Dijkstra</Tag>}
        >
          <NetworkView
            graph={draft.graph}
            closed={draft.closures}
            pathEdges={path?.edge_ids || []}
            onNode={(id) => {
              if (pick === "from") {
                setFrom(id);
                setPick("to");
              } else setTo(id);
              setPath(null);
            }}
          />
          {path?.feasible && (
            <div className="w-form-body">
              <h3>Node sequence</h3>
              <div className="w-node-sequence">
                {path.node_ids?.map((n, i) => (
                  <span key={`${n}-${i}`}>
                    {n}
                    {i < (path.node_ids?.length || 0) - 1 && (
                      <ArrowRight size={13} />
                    )}
                  </span>
                ))}
              </div>
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}
