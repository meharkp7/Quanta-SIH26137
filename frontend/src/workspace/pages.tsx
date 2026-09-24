import { useEffect, useRef, useState } from "react";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Building2,
  Check,
  CheckCircle2,
  Clock3,
  Cpu,
  Database,
  Download,
  ExternalLink,
  FileText,
  FlaskConical,
  MapPin,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Route,
  Save,
  Search,
  ShieldCheck,
  TrafficCone,
  Truck,
  Users,
  Waypoints,
  X,
} from "lucide-react";
import {
  api,
  Evidence,
  postJson,
  ReplayResult,
  SOLVE_TIMEOUT_MS,
} from "../api";
import { useWorkspace } from "./Store";
import { useRunner } from "./Runner";
import {
  clone,
  Comparison,
  createDraft,
  dateLabel,
  download,
  duration,
  exportRun,
  methods,
  requestBody,
  Run,
  seedDrafts,
} from "./data";
import {
  Brand,
  Button,
  Chart,
  colors,
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
import { supabase } from "./supabase";
import { useCatalog } from "./scenarios";

export function Dashboard() {
  const store = useWorkspace(),
    navigate = useNavigate();
  const [selected, setSelected] = useState(store.drafts[0]?.id || "");
  const draft = store.drafts.find((d) => d.id === selected) || store.drafts[0];
  const latest = store.runs.find((r) => r.draft.id === draft?.id);
  return (
    <>
      <PageTitle
        eyebrow="WORKSPACE OVERVIEW"
        title="Your network, at a glance."
        description={`Welcome to ${store.company?.name}. A clear view of your fleet, scenarios and routing decisions.`}
        action={
          <Button
            disabled={!store.canEdit}
            onClick={() => navigate("/app/scenarios/new")}
          >
            <Plus size={17} />
            New scenario
          </Button>
        }
      />
      <div className="w-metrics">
        <Metric
          label="Saved scenarios"
          value={store.drafts.length.toString().padStart(2, "0")}
          detail="Ready to configure and compare"
          icon={<Waypoints size={18} />}
        />
        <Metric
          label="Configured vehicles"
          value={(draft?.graph.fleet.length || 0).toString().padStart(2, "0")}
          detail={draft?.name || "Create a scenario to start"}
          icon={<Truck size={18} />}
          tone="teal"
        />
        <Metric
          label="Delivery stops"
          value={(draft?.graph.requests.length || 0)
            .toString()
            .padStart(2, "0")}
          detail="In the selected scenario"
          icon={<MapPin size={18} />}
          tone="amber"
        />
        <Metric
          label="Completed runs"
          value={store.runs.length.toString().padStart(2, "0")}
          detail="Saved routing computations"
          icon={<BarChart3 size={18} />}
          tone="purple"
        />
      </div>
      {draft ? (
        <>
          <div className="w-dashboard-grid">
            <Panel
              title="Network overview"
              subtitle="Inspect the selected scenario and its latest computed routes."
              action={
                <select
                  aria-label="Dashboard scenario"
                  value={draft.id}
                  onChange={(e) => setSelected(e.target.value)}
                >
                  {store.drafts.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name}
                    </option>
                  ))}
                </select>
              }
              className="w-map-panel"
            >
              <NetworkView
                graph={draft.graph}
                result={latest?.result}
                closed={draft.closures}
              />
              <div className="w-map-overview-foot">
                <span>
                  <i className="w-status-dot" />
                  {draft.graph.nodes.length} road nodes ·{" "}
                  {draft.graph.edges.length} directed roads
                </span>
                <Link to={`/app/live?scenario=${draft.id}`}>
                  Open live operations <ArrowUpRight size={15} />
                </Link>
              </div>
            </Panel>
            <div className="w-dashboard-side">
              <Panel
                title="Recent runs"
                action={
                  <Link className="w-text-link" to="/app/results">
                    View all <ArrowRight size={14} />
                  </Link>
                }
              >
                {store.runs.length ? (
                  <div className="w-recent-runs">
                    {store.runs.slice(0, 4).map((r) => (
                      <Link key={r.id} to={`/app/results/${r.id}`}>
                        <span
                          className={`w-recent-icon ${r.result.evaluation.feasible ? "" : "amber"}`}
                        >
                          <Route size={18} />
                        </span>
                        <div>
                          <strong>{r.draft.name}</strong>
                          <span>
                            {r.result.method} · {r.draft.graph.fleet.length}{" "}
                            vehicles
                          </span>
                          <small>{dateLabel(r.createdAt)}</small>
                        </div>
                        <Chevron />
                      </Link>
                    ))}
                  </div>
                ) : (
                  <Empty
                    title="Ready for the first run"
                    action={
                      <Link
                        className="w-text-link"
                        to={`/app/live?scenario=${draft.id}`}
                      >
                        Optimise routes <ArrowRight size={14} />
                      </Link>
                    }
                  >
                    Results from your routing computations will appear here.
                  </Empty>
                )}
              </Panel>
              <div className="w-insight-card">
                <span className="w-insight-icon">
                  <TrafficCone size={20} />
                </span>
                <Tag tone="amber">SCENARIO CONDITIONS</Tag>
                <h3>
                  {draft.closures.length
                    ? `${draft.closures.length} directed road closures`
                    : "Ready for what changes next."}
                </h3>
                <p>
                  {draft.closures.length
                    ? "Selected roads are excluded from legal routing. Recompute to inspect the impact."
                    : "Introduce a road closure and watch the next plan adapt to the new network."}
                </p>
                <Link to={`/app/live?scenario=${draft.id}`}>
                  Explore incident response <ArrowRight size={16} />
                </Link>
              </div>
            </div>
          </div>
          <div className="w-dashboard-bottom">
            <Panel
              title="Fleet capacity"
              subtitle="Capacity configured for the selected scenario."
              action={
                <Link className="w-text-link" to="/app/fleet">
                  Manage fleet <ArrowRight size={14} />
                </Link>
              }
            >
              <div className="w-capacity-list">
                {draft.graph.fleet.slice(0, 5).map((v, i) => (
                  <div key={v.id}>
                    <span className="w-capacity-vehicle">
                      <Truck size={17} />
                      {v.id}
                    </span>
                    <div>
                      <span
                        style={{
                          width: `${(v.capacity / Math.max(...draft.graph.fleet.map((v) => v.capacity))) * 100}%`,
                          background: colors[i % colors.length],
                        }}
                      />
                    </div>
                    <strong>{v.capacity} units</strong>
                  </div>
                ))}
              </div>
            </Panel>
            <Panel
              title="Plan the next scenario"
              subtitle="Repeatable inputs for meaningful decisions."
            >
              <div className="w-scenario-shortcuts">
                {store.drafts.slice(0, 3).map((d, i) => (
                  <Link key={d.id} to={`/app/scenarios/${d.id}`}>
                    <span>0{i + 1}</span>
                    <div>
                      <strong>{d.name}</strong>
                      <small>
                        {d.graph.requests.length} deliveries ·{" "}
                        {d.method.toUpperCase()}
                      </small>
                    </div>
                    <ArrowUpRight size={17} />
                  </Link>
                ))}
              </div>
            </Panel>
          </div>
        </>
      ) : (
        <Empty
          title="Your company workspace is ready"
          action={
            <div className="w-title-actions">
              <Button onClick={() => navigate("/app/scenarios/new")}>
                Create a scenario
              </Button>
              <Button
                variant="secondary"
                disabled={!store.canEdit}
                onClick={async () => {
                  try {
                    for (const draft of seedDrafts())
                      await store.saveDraft(draft);
                  } catch (e) {
                    store.setError((e as Error).message);
                  }
                }}
              >
                Load sample scenarios
              </Button>
            </div>
          }
        >
          Start with your own network or load editable sample scenarios to
          explore the workflow.
        </Empty>
      )}
    </>
  );
}
function Chevron() {
  return <ArrowUpRight size={16} className="w-muted" />;
}

export function RunPage() {
  const { job, cancel, start } = useRunner();
  const store = useWorkspace();
  const [clock, setClock] = useState(Date.now());
  useEffect(() => {
    if (job?.status !== "running") return;
    const timer = setInterval(() => setClock(Date.now()), 250);
    return () => clearInterval(timer);
  }, [job?.status]);
  if (!job)
    return (
      <Empty
        title="No computation in progress"
        action={
          <Link className="w-button primary" to="/app/scenarios">
            Choose a scenario
          </Link>
        }
      >
        Configure a scenario and start an optimisation run.
      </Empty>
    );
  const elapsed = Math.max(0, (clock - job.started) / 1000);
  return (
    <>
      <PageTitle
        eyebrow="OPTIMISATION RUN"
        title={
          job.status === "running"
            ? "Finding a route forward."
            : job.status === "complete"
              ? "Your route computation is ready."
              : "This run needs attention."
        }
        description={`${job.draft.name} · ${job.draft.method.toUpperCase()} · ${job.draft.graph.fleet.length} vehicles · ${job.draft.graph.requests.length} deliveries`}
        action={
          job.status === "running" ? (
            <Button variant="secondary" onClick={cancel}>
              Cancel request
            </Button>
          ) : job.run ? (
            <Link
              className="w-button primary"
              to={`/app/results/${job.run.id}`}
            >
              Explore results <ArrowRight size={16} />
            </Link>
          ) : (
            <Button onClick={() => start(job.draft)}>Retry run</Button>
          )
        }
      />
      {job.error && (
        <Notice tone={job.status === "complete" ? "amber" : "red"}>
          {job.error}
          {job.run && (
            <Button
              variant="ghost"
              onClick={() =>
                store.saveRun(job.run!).catch((e) => store.setError(e.message))
              }
            >
              Retry saving
            </Button>
          )}
        </Notice>
      )}
      <div className="w-run-layout">
        <Panel
          title="Run status"
          action={
            <Tag
              tone={
                job.status === "complete"
                  ? "green"
                  : job.status === "running"
                    ? "blue"
                    : "amber"
              }
            >
              {job.status}
            </Tag>
          }
        >
          <div className="w-form-body">
            <div
              className={`w-run-orbit ${job.status === "running" ? "running" : ""}`}
            >
              {job.status === "complete" ? (
                <CheckCircle2 size={36} />
              ) : (
                <Cpu size={36} />
              )}
            </div>
            <div className="w-run-time">
              {job.run ? duration(job.run.result.elapsed_s) : duration(elapsed)}
              <small>
                {job.run ? "Reported solver time" : "Elapsed request time"}
              </small>
            </div>
            <div className="w-summary-row">
              <span>Population</span>
              <strong>{job.draft.particles}</strong>
            </div>
            <div className="w-summary-row">
              <span>Evaluation budget</span>
              <strong>{job.draft.evaluations}</strong>
            </div>
            <div className="w-summary-row">
              <span>Seed</span>
              <strong>{job.draft.seed}</strong>
            </div>
            <div className="w-summary-row">
              <span>Closures</span>
              <strong>{job.draft.closures.length}</strong>
            </div>
            {job.status === "running" && (
              <div className="w-indeterminate">
                <span />
              </div>
            )}
            <p className="w-help">
              The solver returns its complete trace when the computation
              finishes. Elapsed time is shown while it works.
            </p>
          </div>
        </Panel>
        <Panel
          title="Convergence"
          subtitle="Actual search trace; lower objective is better."
        >
          {job.run ? (
            <Chart
              series={[
                {
                  name: job.run.result.method,
                  values: job.run.result.trace?.best || [],
                },
              ]}
            />
          ) : (
            <div className="w-running-empty">
              <div className="w-spinner large" />
              <h3>Computing on your scenario</h3>
              <p>Evaluating candidate routes and checking hard constraints.</p>
              <span>Waiting for the backend result</span>
            </div>
          )}
          <div className="w-run-logs">
            <span>RUN LOG</span>
            <p>
              <CheckCircle2 size={14} />
              Snapshot prepared · {job.draft.graph.nodes.length} nodes,{" "}
              {job.draft.graph.edges.length} directed roads
            </p>
            <p>
              <CheckCircle2 size={14} />
              Fleet, demands, service windows and traffic factor submitted
            </p>
            <p>
              <Activity size={14} />
              {job.status === "running"
                ? "Solver request in progress"
                : `Request ${job.status}`}
            </p>
            {job.run && (
              <p>
                <ShieldCheck size={14} />
                {job.run.result.evaluation.feasible
                  ? "Independent validation passed"
                  : "Constraint violations reported"}
              </p>
            )}
          </div>
        </Panel>
      </div>
    </>
  );
}

export function History() {
  const store = useWorkspace(),
    runner = useRunner();
  const [query, setQuery] = useState(""),
    [filter, setFilter] = useState("all");
  const list = store.runs.filter(
    (r) =>
      r.draft.name.toLowerCase().includes(query.toLowerCase()) &&
      (filter === "all" || r.result.method.toLowerCase() === filter),
  );
  return (
    <>
      <PageTitle
        eyebrow="EXPERIMENT HISTORY"
        title="Simulations & runs"
        description="A record of completed computations with their original scenario snapshots."
        action={
          <Link className="w-button primary" to="/app/scenarios/new">
            <Plus size={16} />
            New experiment
          </Link>
        }
      />
      {runner.job?.status === "running" && (
        <Notice>
          A computation is running. <Link to="/app/run">View progress →</Link>
        </Notice>
      )}
      <Panel
        title="Run history"
        subtitle={`${store.runs.length} saved runs`}
        action={
          <div className="w-title-actions">
            <div className="w-table-search">
              <Search size={15} />
              <input
                aria-label="Search run history"
                value={query}
                placeholder="Search runs…"
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <select
              aria-label="Filter run algorithm"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            >
              <option value="all">All algorithms</option>
              {methods.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        }
      >
        {list.length ? (
          <RunTable runs={list} />
        ) : (
          <Empty
            title="No matching runs"
            action={
              <Link to="/app/live" className="w-text-link">
                Open live operations →
              </Link>
            }
          >
            Complete an optimisation to save its inputs and results here.
          </Empty>
        )}
      </Panel>
    </>
  );
}
function RunTable({ runs }: { runs: Run[] }) {
  return (
    <div className="w-table-scroll">
      <table className="w-table">
        <thead>
          <tr>
            <th>Scenario / run</th>
            <th>Method</th>
            <th>Objective</th>
            <th>Distance</th>
            <th>Validation</th>
            <th>Created</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id}>
              <td>
                <Link className="w-table-primary" to={`/app/results/${r.id}`}>
                  <span className="w-table-symbol">
                    <Route size={17} />
                  </span>
                  <span>
                    {r.draft.name}
                    <small>
                      {r.id.slice(0, 8)} · {r.draft.graph.requests.length}{" "}
                      deliveries
                    </small>
                  </span>
                </Link>
              </td>
              <td>
                <Tag tone="blue">{r.result.method}</Tag>
              </td>
              <td>{r.result.evaluation.objective.toFixed(1)}</td>
              <td>{(r.result.evaluation.distance_m / 1000).toFixed(2)} km</td>
              <td>
                <Tag tone={r.result.evaluation.feasible ? "green" : "red"}>
                  {r.result.evaluation.feasible ? "Passed" : "Review"}
                </Tag>
              </td>
              <td>{dateLabel(r.createdAt)}</td>
              <td>
                <Link
                  className="w-icon-button"
                  aria-label={`View run ${r.id.slice(0, 8)}`}
                  to={`/app/results/${r.id}`}
                >
                  <ArrowUpRight size={16} />
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Results() {
  const { id } = useParams();
  const store = useWorkspace(),
    runner = useRunner();
  const activeId = useRef(id);
  activeId.current = id;
  const run = id
    ? store.runs.find((r) => r.id === id) ||
      (runner.job?.run?.id === id ? runner.job.run : null)
    : null;
  const [vehicle, setVehicle] = useState(""),
    [tab, setTab] = useState("routes"),
    [replay, setReplay] = useState<ReplayResult | null>(null),
    [frame, setFrame] = useState(0),
    [playing, setPlaying] = useState(false),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false);
  useEffect(() => {
    setVehicle("");
    setReplay(null);
    setPlaying(false);
    setFrame(0);
  }, [id]);
  useEffect(() => {
    if (!playing || !replay) return;
    const timer = setInterval(
      () =>
        setFrame((n) => {
          if (n >= replay.frames.length - 1) {
            setPlaying(false);
            return n;
          }
          return n + 1;
        }),
      80,
    );
    return () => clearInterval(timer);
  }, [playing, replay]);
  if (!id)
    return (
      <>
        <PageTitle
          eyebrow="ROUTING OUTCOMES"
          title="Results"
          description="Inspect, compare and export the outcome of each saved optimisation."
          action={
            <Link className="w-button secondary" to="/app/comparisons">
              Compare algorithms <ArrowRight size={16} />
            </Link>
          }
        />
        <Panel
          title="Saved results"
          subtitle="Every result includes the exact scenario used for its computation."
        >
          {store.runs.length ? (
            <RunTable runs={store.runs} />
          ) : (
            <Empty
              title="Results start with a run"
              action={
                <Link className="w-button primary" to="/app/live">
                  Open live operations
                </Link>
              }
            >
              Optimise a configured scenario to inspect routes and validation.
            </Empty>
          )}
        </Panel>
      </>
    );
  if (!run)
    return (
      <Empty
        title="Result not found"
        action={<Link to="/app/results">Back to results</Link>}
      >
        This run is not available in the selected company workspace.
      </Empty>
    );
  const result = run.result,
    evaluation = result.evaluation;
  async function playback() {
    if (!run) return;
    if (replay) {
      setPlaying(!playing);
      return;
    }
    setLoading(true);
    setError("");
    const requestedId = id;
    try {
      const data = await postJson<ReplayResult>(
        "/api/workspace/replay",
        { ...requestBody(run.draft), plan: run.result.plan },
        SOLVE_TIMEOUT_MS,
      );
      if (activeId.current !== requestedId) return;
      setReplay(data);
      setPlaying(true);
      setFrame(0);
    } catch (e) {
      if (activeId.current === requestedId) setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }
  return (
    <>
      <PageTitle
        eyebrow={`RESULT · ${run.id.slice(0, 8).toUpperCase()}`}
        title={run.draft.name}
        description={`${result.method} · ${dateLabel(run.createdAt)} · Computed from a saved scenario snapshot`}
        action={
          <>
            <Button variant="secondary" onClick={() => exportRun(run)}>
              <Download size={16} />
              Export CSV
            </Button>
            <Button
              variant="secondary"
              onClick={() =>
                download(
                  `quanta-run-${run.id.slice(0, 8)}.json`,
                  JSON.stringify(run, null, 2),
                )
              }
            >
              JSON
            </Button>
            <Link
              className="w-button primary"
              to={`/app/live?scenario=${run.draft.id}`}
            >
              Open operations <ArrowRight size={16} />
            </Link>
          </>
        }
      />
      {error && <Notice tone="red">{error}</Notice>}
      {result.budget_note && <Notice tone="amber">{result.budget_note}</Notice>}
      {result.status === "unavailable" && (
        <Notice tone="amber">
          The requested solver was unavailable. This is the returned fallback
          plan. {result.error}
        </Notice>
      )}
      <div className="w-metrics">
        <Metric
          label="Objective"
          value={evaluation.objective.toFixed(1)}
          detail="Routing objective · lower is better"
          icon={<BarChart3 size={18} />}
        />
        <Metric
          label="Total distance"
          value={`${(evaluation.distance_m / 1000).toFixed(2)} km`}
          detail="Planned fleet distance"
          icon={<Route size={18} />}
          tone="teal"
        />
        <Metric
          label="Fleet travel time"
          value={duration(evaluation.time_s)}
          detail="Aggregate route evaluation"
          icon={<Clock3 size={18} />}
          tone="amber"
        />
        <Metric
          label="Solver time"
          value={`${result.elapsed_s.toFixed(2)} s`}
          detail={`${result.evaluations ?? "—"} evaluations`}
          icon={<Cpu size={18} />}
          tone="purple"
        />
      </div>
      <div className="w-operation-grid">
        <Panel
          title="Optimised routes"
          subtitle="Select a vehicle to isolate its road path."
          action={
            <Tag tone={evaluation.feasible ? "green" : "red"}>
              {evaluation.feasible ? "Validation passed" : "Review constraints"}
            </Tag>
          }
        >
          <NetworkView
            graph={run.draft.graph}
            result={result}
            closed={replay?.frames[frame]?.closed || run.draft.closures}
            movers={replay?.frames[frame]?.vehicles || []}
            selected={vehicle}
            onSelect={setVehicle}
          />
          <div className="w-playback">
            <Button
              variant="secondary"
              disabled={!evaluation.feasible || loading || !store.canEdit}
              onClick={playback}
            >
              {playing ? <Pause size={15} /> : <Play size={15} />}
              {loading
                ? "Loading…"
                : replay
                  ? playing
                    ? "Pause"
                    : "Play"
                  : "Replay simulation"}
            </Button>
            <input
              aria-label="Result replay timeline"
              type="range"
              min={0}
              max={Math.max(0, (replay?.frames.length || 1) - 1)}
              disabled={!replay}
              value={frame}
              onChange={(e) => {
                setPlaying(false);
                setFrame(Number(e.target.value));
              }}
            />
            <small>
              {replay
                ? duration(replay.frames[frame]?.t || 0)
                : "Kinematic simulation"}
            </small>
          </div>
        </Panel>
        <Panel
          title="Vehicle breakdown"
          subtitle={`${evaluation.vehicles.length} vehicles · ${run.draft.graph.requests.length} deliveries`}
        >
          <RouteList result={result} selected={vehicle} onSelect={setVehicle} />
          <div className="w-form-body">
            <Tag tone="blue">Computed result</Tag>
            <p className="w-help">
              Routes are computed by the backend and checked independently.
              Playback simulates movement along those routes.
            </p>
          </div>
        </Panel>
      </div>
      <div className="w-content-tabs">
        {[
          ["routes", "Route details"],
          ["validation", "Constraint checks"],
          ["convergence", "Convergence"],
          ["configuration", "Run configuration"],
        ].map(([key, label]) => (
          <button
            key={key}
            className={tab === key ? "active" : ""}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "routes" && (
        <Panel
          title="Delivery sequence"
          subtitle="Planned service times are measured from the scenario start."
        >
          <div className="w-table-scroll">
            <table className="w-table">
              <thead>
                <tr>
                  <th>Vehicle</th>
                  <th>Customer order</th>
                  <th>Load / capacity</th>
                  <th>Elapsed</th>
                  <th>Road segments</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {evaluation.vehicles
                  .filter((v) => !vehicle || v.id === vehicle)
                  .map((v, i) => (
                    <tr key={v.id}>
                      <td>
                        <span
                          className="w-colour-dot"
                          style={{ background: colors[i % colors.length] }}
                        />
                        {v.id}
                      </td>
                      <td>
                        {v.order.join(" → ") || "No assigned stops"}
                        <small>
                          {v.stops
                            .map((s) => `${s.job}: ${duration(s.start_s)}`)
                            .join(" · ")}
                        </small>
                      </td>
                      <td>
                        {v.load} / {v.capacity}
                      </td>
                      <td>{duration(v.elapsed_s)}</td>
                      <td>{v.edge_ids.length}</td>
                      <td>
                        <Tag tone={v.feasible ? "green" : "red"}>
                          {v.feasible ? "Valid" : "Review"}
                        </Tag>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
      {tab === "validation" && (
        <Panel title="Independent validator report">
          <div className="w-validation-grid">
            {[
              ["Capacity", evaluation.capacity],
              ["Time windows", evaluation.windows],
              ["Connectivity", evaluation.connectivity],
              ["All deliveries served", evaluation.all_served],
            ].map(([name, pass]) => (
              <div key={String(name)}>
                <ShieldCheck size={22} />
                <strong>{name}</strong>
                <Tag tone={pass ? "green" : "red"}>
                  {pass ? "Passed" : "Failed"}
                </Tag>
              </div>
            ))}
          </div>
          {evaluation.violations?.length > 0 && (
            <div className="w-form-body">
              {evaluation.violations.map((v, i) => (
                <Notice key={i} tone="red">
                  {v.name}: {v.detail}
                </Notice>
              ))}
            </div>
          )}
        </Panel>
      )}
      {tab === "convergence" && (
        <Panel
          title="Search convergence"
          subtitle="Best and mean objective values returned by this run."
        >
          <Chart
            series={[
              { name: "Best objective", values: result.trace?.best || [] },
              {
                name: "Mean objective",
                values: result.trace?.mean || [],
                color: colors[1],
              },
            ]}
          />
        </Panel>
      )}
      {tab === "configuration" && (
        <Panel title="Immutable input snapshot">
          <div className="w-form-body w-config-grid">
            {[
              ["Network", run.draft.networkId],
              ["Method", result.method],
              ["Population", run.draft.particles],
              ["Evaluation budget", run.draft.evaluations],
              ["Seed", run.draft.seed],
              ["Speed factor", `${run.draft.traffic * 100}%`],
              ["Closures", run.draft.closures.join(", ") || "None"],
              ["Data source", run.draft.source],
            ].map(([label, value]) => (
              <div className="w-summary-row" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </>
  );
}

export function Comparisons() {
  const store = useWorkspace(),
    runner = useRunner(),
    [params] = useSearchParams();
  const [selected, setSelected] = useState(
    params.get("scenario") || store.drafts[0]?.id || "",
  );
  const draft = store.drafts.find((d) => d.id === selected);
  const saved = store.runs.find((r) => r.draft.id === selected && r.comparison);
  const [comparison, setComparison] = useState<Comparison | null>(
      saved?.comparison || null,
    ),
    [snapshot, setSnapshot] = useState(saved?.draft || null);
  const [selectedMethods, setSelectedMethods] = useState([
      "qpso",
      "pso",
      "alns",
    ]),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    setComparison(saved?.comparison || null);
    setSnapshot(saved?.draft || null);
  }, [selected]);
  async function compare() {
    if (!draft) return;
    if (selectedMethods.length < 2) {
      setError("Select at least two algorithms.");
      return;
    }
    setBusy(true);
    setError("");
    const input = clone(draft);
    try {
      const result = await postJson<Comparison>(
        "/api/workspace/compare",
        { ...requestBody(input), methods: selectedMethods },
        SOLVE_TIMEOUT_MS,
      );
      setComparison(result);
      setSnapshot(input);
      const run = await runner.start(input);
      if (run) await store.saveRun({ ...run, comparison: result });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const eligible =
    comparison?.rows.filter(
      (r) =>
        r.feasible &&
        !r.error &&
        r.status !== "unavailable" &&
        r.objective != null,
    ) || [];
  const best = eligible.length
    ? Math.min(...eligible.map((r) => r.objective!))
    : null;
  return (
    <>
      <PageTitle
        eyebrow="ALGORITHM BENCHMARKING"
        title="Compare the approaches."
        description="Evaluate route quality, feasibility and runtime against the same scenario and search budget."
        action={
          <Button
            disabled={
              busy ||
              runner.job?.status === "running" ||
              !draft ||
              !store.canEdit
            }
            onClick={compare}
          >
            <Play size={16} />
            {busy ? "Comparing…" : "Run comparison"}
          </Button>
        }
      />
      {error && <Notice tone="red">{error}</Notice>}
      <Panel
        title="Comparison setup"
        subtitle="Timing can vary between executions; the scenario and random seed remain fixed."
      >
        <div className="w-form-body">
          <div className="w-form-grid">
            <Field label="Scenario">
              <select
                disabled={busy}
                value={selected}
                onChange={(e) => setSelected(e.target.value)}
              >
                <option value="" disabled>
                  Choose a scenario
                </option>
                {store.drafts.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </Field>
            <div className="w-comparison-budget">
              <span>{draft?.particles || "—"} population</span>
              <span>{draft?.evaluations || "—"} evaluation budget</span>
              <span>Seed {draft?.seed || "—"}</span>
            </div>
          </div>
          <div className="w-method-checkboxes">
            {methods.map((m) => (
              <label key={m.id}>
                <input
                  type="checkbox"
                  disabled={
                    busy ||
                    (m.id === "milp" &&
                      (draft?.graph.requests.length || 0) > 12)
                  }
                  checked={selectedMethods.includes(m.id)}
                  onChange={(e) =>
                    setSelectedMethods(
                      e.target.checked
                        ? [...selectedMethods, m.id]
                        : selectedMethods.filter((id) => id !== m.id),
                    )
                  }
                />
                {m.label}
              </label>
            ))}
          </div>
        </div>
      </Panel>
      {comparison ? (
        <>
          <div className="w-section-heading">
            <div>
              <h2>Comparison results</h2>
              <p>
                {snapshot?.name} · {snapshot?.graph.requests.length} deliveries
                · {snapshot?.closures.length} closures
              </p>
            </div>
            <Button
              variant="secondary"
              onClick={() =>
                download(
                  "quanta-comparison.json",
                  JSON.stringify({ scenario: snapshot, comparison }, null, 2),
                )
              }
            >
              <Download size={15} />
              Export comparison
            </Button>
          </div>
          <div className="w-comparison-cards">
            {comparison.rows.map((row, i) => (
              <div
                key={row.method}
                className={
                  row.objective === best &&
                  !row.error &&
                  row.status !== "unavailable"
                    ? "best"
                    : ""
                }
              >
                <span>
                  {row.method}
                  <i style={{ background: colors[i % colors.length] }} />
                </span>
                <strong>
                  {row.error || row.status === "unavailable"
                    ? "Unavailable"
                    : (row.objective?.toFixed(1) ?? "—")}
                </strong>
                <small>
                  {row.objective === best &&
                  !row.error &&
                  row.status !== "unavailable"
                    ? "Best observed objective"
                    : row.error || row.status === "unavailable"
                      ? "No valid benchmark result"
                      : row.feasible
                        ? "Feasible solution"
                        : "No feasible solution"}
                </small>
              </div>
            ))}
          </div>
          <div className="w-two-columns">
            <Panel
              title="Convergence comparison"
              subtitle="Only algorithms returning a search trace are plotted."
            >
              <Chart
                series={Object.entries(comparison.traces || {}).map(
                  ([name, trace], i) => ({
                    name,
                    values: trace?.best || [],
                    color: colors[i],
                  }),
                )}
              />
            </Panel>
            <Panel
              title="Compute time"
              subtitle="Measured wall-clock solver time on this machine."
            >
              <div className="w-runtime-bars">
                {eligible.map((row, i) => (
                  <div key={row.method}>
                    <span>{row.method}</span>
                    <div>
                      <span
                        style={{
                          width: `${Math.max(2, ((row.elapsed_s || 0) / Math.max(...eligible.map((r) => r.elapsed_s || 0), 0.001)) * 100)}%`,
                          background: colors[i % colors.length],
                        }}
                      />
                    </div>
                    <strong>{row.elapsed_s?.toFixed(3)} s</strong>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
          <Panel title="Benchmark detail">
            <div className="w-table-scroll">
              <table className="w-table">
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>Objective</th>
                    <th>Travel time</th>
                    <th>Distance</th>
                    <th>Compute time</th>
                    <th>Feasibility / status</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.rows.map((r) => (
                    <tr key={r.method}>
                      <td>{r.method}</td>
                      <td>{r.objective?.toFixed(1) ?? "—"}</td>
                      <td>{duration(r.time_s || 0)}</td>
                      <td>{((r.distance_m || 0) / 1000).toFixed(2)} km</td>
                      <td>{r.elapsed_s?.toFixed(3)} s</td>
                      <td>
                        <Tag
                          tone={
                            r.error || r.status === "unavailable"
                              ? "amber"
                              : r.feasible
                                ? "green"
                                : "red"
                          }
                        >
                          {r.error ||
                            r.status ||
                            (r.feasible ? "Feasible" : "Infeasible")}
                        </Tag>
                        {r.certified && <small>Certified optimum</small>}
                        {r.gap != null && (
                          <small>Gap {(r.gap * 100).toFixed(2)}%</small>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      ) : (
        <Panel>
          <Empty
            title="Compare on equal terms"
            action={busy ? <span className="w-spinner" /> : undefined}
          >
            Choose two or more algorithms and run the comparison. Cost and
            runtime charts use actual returned measurements.
          </Empty>
        </Panel>
      )}
    </>
  );
}

export function DataModels() {
  const store = useWorkspace(),
    { catalog, error: catalogError } = useCatalog();
  const [evidence, setEvidence] = useState<Evidence | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    api<Evidence>("/api/evidence")
      .then(setEvidence)
      .catch((e) => setError(e.message));
  }, []);
  return (
    <>
      <PageTitle
        eyebrow="DATA & MODEL OBSERVABILITY"
        title="Know what powers the decision."
        description="Inspect network availability, model evidence and the distinction between computation and simulation."
        action={
          <Link className="w-button secondary" to="/app/documentation">
            <FileText size={16} />
            Read the guide
          </Link>
        }
      />
      {(error || catalogError) && (
        <Notice tone="amber">{error || catalogError}</Notice>
      )}
      <div className="w-two-columns">
        <Panel
          title="Capability status"
          subtitle="Current execution paths in the platform."
        >
          <div className="w-status-list">
            {[
              ["Directed routing & validation", "Computed", "green"],
              ["QPSO / PSO / ALNS", "Computed", "green"],
              ["Vehicle movement", "Simulation", "amber"],
              ["Neural traffic forecasting", "Offline evidence", "blue"],
              ["DRL scope selection", "Demo / research", "amber"],
              ["Live GPS telemetry", "Not connected", "neutral"],
            ].map(([label, status, tone]) => (
              <div key={label}>
                <span>{label}</span>
                <Tag tone={tone as any}>{status}</Tag>
              </div>
            ))}
          </div>
        </Panel>
        <Panel
          title="Forecast evidence"
          subtitle={
            evidence?.demo
              ? "Static demo evidence returned by the API."
              : "Saved evaluation artifacts; not live road predictions."
          }
          action={
            <Tag tone="blue">
              {evidence?.demo ? "Demo values" : "Offline evaluation"}
            </Tag>
          }
        >
          <div className="w-form-body">
            {evidence?.available ? (
              <>
                <div className="w-evidence-number">
                  <span>GNN–Transformer MAE</span>
                  <strong>{evidence.model_mae?.toFixed(4) ?? "—"}</strong>
                </div>
                <div className="w-summary-row">
                  <span>Temporal baseline MAE</span>
                  <strong>
                    {evidence.temporal_baseline_mae?.toFixed(4) ?? "—"}
                  </strong>
                </div>
                <div className="w-summary-row">
                  <span>Held-out interval coverage</span>
                  <strong>
                    {evidence.uncertainty?.test?.actual_coverage != null
                      ? `${(evidence.uncertainty.test.actual_coverage * 100).toFixed(1)}%`
                      : "—"}
                  </strong>
                </div>
                <p className="w-help">
                  Lower MAE is better. These measurements do not imply that the
                  neural model beats the baseline or is serving predictions to
                  the live routing loop.
                </p>
              </>
            ) : (
              <p>Forecast artifacts are not available to this API process.</p>
            )}
          </div>
        </Panel>
      </div>
      <Panel
        title="Network catalog"
        subtitle="Existing file references are preserved. Add the missing Delhi files, then refresh the page."
      >
        <div className="w-table-scroll">
          <table className="w-table">
            <thead>
              <tr>
                <th>Network</th>
                <th>Identifier</th>
                <th>Purpose</th>
                <th>Availability</th>
              </tr>
            </thead>
            <tbody>
              {catalog.map((c) => (
                <tr key={c.id}>
                  <td>{c.label}</td>
                  <td>
                    <code>{c.id}</code>
                  </td>
                  <td>{c.role}</td>
                  <td>
                    <Tag tone={c.available ? "green" : "amber"}>
                      {c.available ? "Ready" : "Data unavailable"}
                    </Tag>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <Panel
        title="Workspace data"
        subtitle="Download editable network and fleet snapshots."
      >
        <div className="w-status-list">
          {store.drafts.map((d) => (
            <div key={d.id}>
              <span>
                <strong>{d.name}</strong>
                <small>
                  {d.graph.nodes.length} nodes · {d.graph.fleet.length} vehicles
                  · {d.graph.requests.length} deliveries
                </small>
              </span>
              <Button
                variant="secondary"
                onClick={() =>
                  download(
                    `${d.name.replace(/[^a-z0-9]/gi, "-")}.json`,
                    JSON.stringify(d.graph, null, 2),
                  )
                }
              >
                <Download size={15} />
                Export network
              </Button>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}

export function SettingsPage() {
  const store = useWorkspace();
  const [name, setName] = useState(store.company?.name || ""),
    [industry, setIndustry] = useState(store.company?.industry || "");
  const [email, setEmail] = useState(""),
    [role, setRole] = useState("dispatcher"),
    [message, setMessage] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [reset, setReset] = useState(false);
  const [members, setMembers] = useState<any[]>([]),
    [invites, setInvites] = useState<any[]>([]);
  const owner = store.company?.role === "owner";
  async function loadTeam() {
    if (store.demo) {
      setMembers([
        {
          user_id: "demo-owner",
          display_name: "Demo workspace owner",
          role: "owner",
        },
      ]);
      setInvites(
        JSON.parse(localStorage.getItem("quanta.demo.invites") || "[]"),
      );
      return;
    }
    if (!supabase || !store.company) return;
    const { data, error } = await supabase
      .from("company_members")
      .select("*")
      .eq("company_id", store.company.id);
    if (error) throw error;
    setMembers(data || []);
    if (owner) {
      const { data, error } = await supabase
        .from("company_invites")
        .select("*")
        .eq("company_id", store.company.id);
      if (error) throw error;
      setInvites(data || []);
    }
  }
  useEffect(() => {
    loadTeam().catch((e) => setError(e.message));
  }, [store.company?.id]);
  async function saveCompany() {
    if (!supabase || !store.company || store.demo) return;
    setBusy(true);
    setError("");
    try {
      const { error } = await supabase
        .from("companies")
        .update({ name, industry })
        .eq("id", store.company.id);
      if (error) throw error;
      await store.reloadCompanies();
      setMessage("Company details updated.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function invite(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const address = email.trim().toLowerCase();
      if (store.demo) {
        if (invites.some((i) => i.email === address))
          throw new Error("This invitation already exists.");
        const next = [
          ...invites,
          { id: crypto.randomUUID(), email: address, role },
        ];
        localStorage.setItem("quanta.demo.invites", JSON.stringify(next));
        setInvites(next);
        setMessage(
          "Sample invitation added to this browser. No email was sent.",
        );
      } else {
        const { error } = await supabase!
          .from("company_invites")
          .insert({ company_id: store.company!.id, email: address, role });
        if (error) throw error;
        await loadTeam();
        setMessage(
          "Invitation saved. Ask this person to sign up or sign in with this email; they will join after verification. No invitation email is sent automatically.",
        );
      }
      setEmail("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="WORKSPACE ADMINISTRATION"
        title="Company settings"
        description="Manage your company profile, team access and workspace preferences."
      />
      {message && <Notice tone="green">{message}</Notice>}
      {error && <Notice tone="red">{error}</Notice>}
      <div className="w-settings-grid">
        <Panel
          title="Company profile"
          subtitle={
            store.demo
              ? "This is a sample company profile."
              : "Visible to the members of your workspace."
          }
        >
          <div className="w-form-body">
            <div className="w-company-profile">
              <span>{name.slice(0, 1)}</span>
              <div>
                <strong>{store.company?.name}</strong>
                <Tag tone="blue">{store.company?.role}</Tag>
              </div>
            </div>
            <Field label="Company name">
              <input
                value={name}
                minLength={2}
                maxLength={120}
                disabled={store.demo || !owner}
                onChange={(e) => setName(e.target.value)}
              />
            </Field>
            <Field label="Industry">
              <input
                value={industry}
                disabled={store.demo || !owner}
                onChange={(e) => setIndustry(e.target.value)}
              />
            </Field>
            <Button
              disabled={busy || store.demo || !owner || name.trim().length < 2}
              onClick={saveCompany}
            >
              <Save size={15} />
              Save company details
            </Button>
            {store.companies.length > 1 && (
              <Field label="Switch workspace">
                <select
                  value={store.company?.id}
                  onChange={(e) => store.selectCompany(e.target.value)}
                >
                  {store.companies.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              </Field>
            )}
          </div>
        </Panel>
        <Panel
          title="Account & access"
          subtitle="Authentication and storage status."
        >
          <div className="w-form-body">
            <div className="w-summary-row">
              <span>Email</span>
              <strong>
                {store.demo ? "Demo session" : store.session?.user.email}
              </strong>
            </div>
            <div className="w-summary-row">
              <span>Authentication</span>
              <Tag tone={supabase ? "green" : "amber"}>
                {supabase ? "Supabase configured" : "Setup required"}
              </Tag>
            </div>
            <div className="w-summary-row">
              <span>Storage</span>
              <strong>
                {store.demo ? "This browser" : "Company database"}
              </strong>
            </div>
            <div className="w-summary-row">
              <span>Your role</span>
              <strong>{store.company?.role}</strong>
            </div>
            <p className="w-help">
              Owners manage company details and team access. Dispatchers edit
              scenarios and run computations. Viewers inspect saved data and
              results.
            </p>
            {!store.demo && (
              <Link className="w-button secondary" to="/forgot-password">
                Reset password <ArrowRight size={15} />
              </Link>
            )}
            {store.demo && (
              <Link className="w-button secondary" to="/signup">
                Create your own company <ArrowRight size={15} />
              </Link>
            )}
          </div>
        </Panel>
      </div>
      <Panel
        title="Team members"
        subtitle="Owners retain ownership. Other members can be dispatchers or viewers."
      >
        <div className="w-table-scroll">
          <table className="w-table">
            <thead>
              <tr>
                <th>Member</th>
                <th>Role</th>
                <th>Access</th>
              </tr>
            </thead>
            <tbody>
              {members.map((m) => (
                <tr key={m.user_id}>
                  <td>
                    <span className="w-member">
                      <span className="w-avatar">
                        {(m.display_name || "M")[0]}
                      </span>
                      {m.display_name || "Team member"}
                    </span>
                  </td>
                  <td>
                    {m.role === "owner" || !owner || store.demo ? (
                      <Tag>{m.role}</Tag>
                    ) : (
                      <select
                        aria-label={`Role for ${m.display_name}`}
                        value={m.role}
                        onChange={async (e) => {
                          try {
                            const { error } = await supabase!.rpc(
                              "set_member_role",
                              {
                                target_company: store.company!.id,
                                target_user: m.user_id,
                                new_role: e.target.value,
                              },
                            );
                            if (error) throw error;
                            await loadTeam();
                          } catch (e) {
                            setError((e as Error).message);
                          }
                        }}
                      >
                        <option value="dispatcher">Dispatcher</option>
                        <option value="viewer">Viewer</option>
                      </select>
                    )}
                  </td>
                  <td>
                    <Tag tone="green">Active</Tag>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      {owner && (
        <Panel
          title="Invite a team member"
          subtitle="An invitation is accepted when the matching verified account signs in."
        >
          <form className="w-invite-form" onSubmit={invite}>
            <Field label="Email address">
              <input
                required
                type="email"
                value={email}
                placeholder="colleague@company.com"
                onChange={(e) => setEmail(e.target.value)}
              />
            </Field>
            <Field label="Role">
              <select value={role} onChange={(e) => setRole(e.target.value)}>
                <option value="dispatcher">Dispatcher</option>
                <option value="viewer">Viewer</option>
              </select>
            </Field>
            <Button disabled={busy}>
              <Plus size={16} />
              Add invitation
            </Button>
          </form>
          {invites.length > 0 && (
            <div className="w-status-list">
              {invites.map((i) => (
                <div key={i.id}>
                  <span>
                    {i.email}
                    <small>{i.role} · Pending sign-in</small>
                  </span>
                  <Button
                    variant="ghost"
                    onClick={async () => {
                      try {
                        if (store.demo) {
                          const next = invites.filter((x) => x.id !== i.id);
                          setInvites(next);
                          localStorage.setItem(
                            "quanta.demo.invites",
                            JSON.stringify(next),
                          );
                        } else {
                          const { error } = await supabase!
                            .from("company_invites")
                            .delete()
                            .eq("id", i.id)
                            .eq("company_id", store.company!.id);
                          if (error) throw error;
                          await loadTeam();
                        }
                      } catch (e) {
                        setError((e as Error).message);
                      }
                    }}
                  >
                    Revoke
                  </Button>
                </div>
              ))}
            </div>
          )}
        </Panel>
      )}
      {store.demo && (
        <Panel
          title="Demo workspace"
          subtitle="Reset only the sample scenarios and run history stored in this browser."
        >
          <div className="w-form-body">
            <Button variant="secondary" onClick={() => setReset(true)}>
              <RefreshCw size={15} />
              Reset demo data
            </Button>
          </div>
        </Panel>
      )}
      {reset && (
        <div className="w-modal-backdrop">
          <div
            className="w-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Reset demo"
          >
            <h2>Reset the demo workspace?</h2>
            <p>
              This clears local demo runs, edits and sample invitations. Company
              database records are unaffected.
            </p>
            <div className="w-modal-actions">
              <Button variant="secondary" onClick={() => setReset(false)}>
                Keep changes
              </Button>
              <Button
                variant="danger"
                onClick={() => {
                  store.resetDemo();
                  localStorage.removeItem("quanta.demo.invites");
                  setInvites([]);
                  setReset(false);
                  setMessage("Demo workspace reset.");
                }}
              >
                Reset demo
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export function Documentation({
  publicPage = false,
}: {
  publicPage?: boolean;
}) {
  return (
    <div className={publicPage ? "w-public-docs" : ""}>
      {publicPage && (
        <header>
          <Link to="/">
            <Brand />
          </Link>
          <Link className="w-button secondary" to="/">
            Back to website
          </Link>
        </header>
      )}
      <PageTitle
        eyebrow="PLATFORM GUIDE"
        title="A clear path through Quanta."
        description="Understand the workflow, the algorithms and the data behind each result."
      />
      <div className="w-doc-grid">
        {[
          [
            "01",
            "Configure a scenario",
            "Choose a project network, generate a synthetic network or import JSON / GraphML. Add fleet capacity, delivery demands and service windows. Save the draft before running.",
          ],
          [
            "02",
            "Compute and validate",
            "QPSO, PSO, ALNS and the constructive baseline compute real routes on your submitted snapshot. Capacity, time windows and directed connectivity are checked independently. MILP is restricted to small scenarios and reports availability or certification explicitly.",
          ],
          [
            "03",
            "Inspect live operations",
            "Select a scenario, optimise and inspect each vehicle’s route. Close a directed road to test a disruption. Auto replan computes a fresh route when a result already exists. Changed inputs are flagged until recomputed.",
          ],
          [
            "04",
            "Replay a simulation",
            "Workspace playback uses kinematic vehicle movement over validated road paths. It is a simulation, not a live GPS feed. The original small-fixture SUMO API remains available in the existing project.",
          ],
          [
            "05",
            "Compare fairly",
            "The comparison uses one scenario snapshot, random seed, traffic factor and requested budget. Inspect objective, runtime, feasibility and convergence. A best observed objective is not automatically a certified optimum.",
          ],
          [
            "06",
            "Share and reproduce",
            "Saved runs retain their original inputs. Export a CSV route summary or a full JSON run snapshot. Company records are visible only to members; demo records stay in the current browser.",
          ],
        ].map(([n, t, d]) => (
          <Panel key={n}>
            <div className="w-doc-card">
              <span>{n}</span>
              <h2>{t}</h2>
              <p>{d}</p>
            </div>
          </Panel>
        ))}
      </div>
      <Panel title="How to read data labels">
        <div className="w-status-list">
          <div>
            <Tag tone="green">Computed</Tag>
            <span>Returned by an actual backend routing computation.</span>
          </div>
          <div>
            <Tag tone="amber">Simulation</Tag>
            <span>Modelled movement or configured traffic conditions.</span>
          </div>
          <div>
            <Tag tone="blue">Offline evidence</Tag>
            <span>Saved research measurements, not live predictions.</span>
          </div>
          <div>
            <Tag>Data unavailable</Tag>
            <span>The configured local scenario file is not present yet.</span>
          </div>
        </div>
      </Panel>
      <Notice>
        Forecast and DRL research are presented with their current evidence
        status. Free-flow speed colours are not a real-time congestion feed.
        Traffic assumptions remain visible in the scenario configuration.
      </Notice>
    </div>
  );
}
