import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { Activity, AlertTriangle, ArrowRight, Bot, CarFront, CheckCircle2, ChevronRight, CircleAlert, Gauge, Layers3, MapPinned, Minus, Pause, Play, Route, ShieldCheck, Sparkles, TrafficCone, TrendingDown, TrendingUp, Trophy, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { api, Evidence, DrlDemo, Evaluation, Graph, ReplayFrame, ScenarioSummary, SolveResult, postJson } from "@/api";
import { useCountUp, useInView, usePrefersReducedMotion } from "@/hooks";

type Page = "home" | "simulation";

const SPEEDS = [0.5, 1, 2, 4];

function App() {
  const [page, setPage] = useState<Page>("home");
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [scenarioId, setScenarioId] = useState("S3_BASE");
  const [graph, setGraph] = useState<Graph | null>(null);
  const [closed, setClosed] = useState<string[]>([]);
  const [method, setMethod] = useState("qpso");
  const [solve, setSolve] = useState<SolveResult | null>(null);
  const [previousSolve, setPreviousSolve] = useState<SolveResult | null>(null);
  const [replay, setReplay] = useState<ReplayFrame[]>([]);
  const [frame, setFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("control");
  const [evidence, setEvidence] = useState<Evidence | null>(null);

  useEffect(() => {
    Promise.all([api<{ scenarios: ScenarioSummary[] }>("/api/scenarios"), api<Graph>(`/api/scenarios/${scenarioId}`)])
      .then(([catalog, nextGraph]) => { setScenarios(catalog.scenarios); setGraph(nextGraph); })
      .catch((err: Error) => setError(err.message));
    api<Evidence>("/api/evidence").then(setEvidence).catch(() => setEvidence({ available: false, reason: "Evidence endpoint unavailable" }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (page !== "simulation") return;
    api<Graph>(`/api/scenarios/${scenarioId}?closed=${encodeURIComponent(closed.join(","))}`)
      .then(setGraph).catch((err: Error) => setError(err.message));
  }, [scenarioId, closed, page]);

  useEffect(() => {
    if (!replay.length || !isPlaying) return;
    const timer = window.setInterval(() => setFrame((value) => {
      if (value >= replay.length - 1) {
        setIsPlaying(false);
        return value;
      }
      return value + 1;
    }), Math.max(30, 120 / speed));
    return () => window.clearInterval(timer);
    }, [replay, isPlaying, speed]);

  const activeFrame = replay[frame];
  const activeEdges = useMemo(() => new Set(solve?.evaluation.vehicles.flatMap((vehicle) => vehicle.edge_ids) || []), [solve]);

  async function runSolve() {
    setBusy("solve"); setError("");
    try {
      const result = await postJson<SolveResult>("/api/solve", { scenario_id: scenarioId, method, particles: 12, evaluations: 40, seed: 7, closed_edge_ids: closed });
      setPreviousSolve(solve);
      setSolve(result);
    } catch (err) { setError((err as Error).message); } finally { setBusy(""); }
  }

  async function runReplay() {
    setBusy("replay"); setError("");
    try { const result = await postJson<{ frames: ReplayFrame[] }>("/api/sumo/replay", { scenario_id: scenarioId, plan: solve?.plan || null, closed_edge_ids: closed }); setReplay(result.frames || []); setFrame(0); setIsPlaying(true); }
    catch (err) { setError((err as Error).message); } finally { setBusy(""); }
  }

  function toggleReplay() { if (!replay.length) { void runReplay(); return; } setIsPlaying((value) => !value); }

  // Spacebar play/pause shortcut (control tab only, never hijacks form fields).
  const toggleRef = useRef(toggleReplay);
  toggleRef.current = toggleReplay;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== "Space" || activeTab !== "control") return;
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(target.tagName)) return;
      event.preventDefault();
      toggleRef.current();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeTab]);

  if (page === "home") return <Home onStart={() => setPage("simulation")} />;
  return <div className="app-shell">
    <header className="topbar">
      <button className="brand-button" onClick={() => setPage("home")}><span className="brand-mark">✦</span><span>QUANTA</span><Badge variant="outline">SIH26137</Badge></button>
      <nav className="nav-tabs"><button className={activeTab === "control" ? "nav-active" : ""} onClick={() => setActiveTab("control")}>Live Control Room</button><button className={activeTab === "forecast" ? "nav-active" : ""} onClick={() => setActiveTab("forecast")}>Forecast</button><button className={activeTab === "replanning" ? "nav-active" : ""} onClick={() => setActiveTab("replanning")}>Replanning</button><button className={activeTab === "evidence" ? "nav-active" : ""} onClick={() => setActiveTab("evidence")}>Evidence</button></nav>
      <div className="top-status"><span className="live-dot" /> SYSTEM ONLINE <span className="muted">· SUMO READY</span></div>
    </header>
    {error && <div className="error-bar"><CircleAlert size={15} /> {error}<button onClick={() => setError("")}><X size={15} /></button></div>}
    {activeTab === "control" ? <main className="control-layout" key="control">
      <section className="control-main">
        <div className="eyebrow reveal"><Activity size={14} /> LIVE DECISION SYSTEM <span>•</span><span className="muted">{scenarioId}</span></div>
        <div className="headline-row reveal" style={{ "--d": "60ms" } as CSSProperties}><div><h1>See the decision.<br /><em>Trust the route.</em></h1><p className="lede">Quanta turns traffic disruption into a validated delivery plan in seconds.</p></div><div className="step-rail"><Step label="Observe" active /><Step label="Predict" /><Step label="Replan" /><Step label="Prove" /></div></div>
        <Card className="map-card reveal lift" style={{ "--d": "120ms" } as CSSProperties}>
          <CardHeader><div><CardTitle><MapPinned size={17} /> City operations map</CardTitle><CardDescription>Directed roads, delivery fleet, and the route selected by the optimizer.</CardDescription></div><Badge key={closed.length ? `closed-${closed.length}` : "clear"} variant={closed.length ? "warning" : "success"} className="badge-pulse">{closed.length ? `${closed.length} ROAD CLOSURE` : "NETWORK CLEAR"}</Badge></CardHeader>
          <CardContent><NetworkMap graph={graph} routeEdges={activeEdges} closed={closed} movers={activeFrame?.vehicles || []} replayPct={replay.length ? ((frame + 1) / replay.length) * 100 : 0} /><div className="replay-bar"><span className="replay-time">{activeFrame ? `t = ${activeFrame.t.toFixed(0)}s` : "No replay loaded"}</span><input aria-label="SUMO replay timeline" type="range" min="0" max={Math.max(0, replay.length - 1)} value={frame} onChange={(event) => { setIsPlaying(false); setFrame(Number(event.target.value)); }} disabled={!replay.length} /><span className="replay-count">{replay.length ? `${frame + 1} / ${replay.length}` : "—"}</span></div><div className="replay-sub"><button className="replay-toggle" onClick={toggleReplay} disabled={busy === "replay"} aria-label={isPlaying ? "Pause replay" : "Play replay"}>{isPlaying ? <Pause size={13} /> : <Play size={13} />}</button><div className="speed-ctl" role="group" aria-label="Playback speed">{SPEEDS.map((option) => <button key={option} className={speed === option ? "speed-active" : ""} onClick={() => setSpeed(option)}>{option}x</button>)}</div><span className="replay-vehicles">{activeFrame ? `${activeFrame.vehicles.length} vehicles` : "0 vehicles"}</span><span className="replay-hint">Space to play / pause</span></div></CardContent>
        </Card>
        <div className="kpi-grid"><CountKpi icon={<Gauge />} label="Travel time" value={solve ? solve.evaluation.time_s : null} format={(n) => `${n.toFixed(0)} s`} delay="160ms" /><CountKpi icon={<Route />} label="Distance" value={solve ? solve.evaluation.distance_m / 1000 : null} format={(n) => `${n.toFixed(2)} km`} delay="220ms" /><Kpi icon={<CarFront />} label="Completed delivery" value={solve ? `${solve.evaluation.vehicles.reduce((sum, v) => sum + v.order.length, 0)} / ${graph?.requests.length || 5}` : "—"} accent={solve?.evaluation.all_served ? "good" : ""} delay="280ms" /><CountKpi icon={<Activity />} label="Replanning latency" value={solve ? solve.elapsed_s : null} format={(n) => `${n.toFixed(2)} s`} delay="340ms" /></div>
      </section>
      <aside className="control-sidebar">
        <Card className="control-card reveal lift" style={{ "--d": "180ms" } as CSSProperties}><CardHeader><CardTitle>Run a scenario</CardTitle><CardDescription>Start with a disruption so the system has something meaningful to explain.</CardDescription></CardHeader><CardContent className="control-fields"><Field label="Scenario"><select value={scenarioId} onChange={(e) => { setScenarioId(e.target.value); setSolve(null); setPreviousSolve(null); }}><option value="S3_BASE">Five-job city fixture</option>{scenarios.filter((s) => s.id !== "S3_BASE" && s.available).slice(0, 5).map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}</select></Field><Field label="Optimizer"><select value={method} onChange={(e) => setMethod(e.target.value)}><option value="qpso">QPSO · quantum-inspired</option><option value="pso">PSO · classical comparator</option><option value="alns">ALNS · adaptive heuristic</option><option value="constructive">Constructive baseline</option></select></Field><Field label="Incident"><select value={closed[0] || ""} onChange={(e) => setClosed(e.target.value ? [e.target.value] : [])}><option value="">No active incident</option>{graph?.edges.map((edge) => <option key={edge.id} value={edge.id}>{edge.id} · {edge.from} → {edge.to}</option>)}</select></Field><div className="button-stack"><Button size="lg" onClick={runSolve} disabled={!!busy}><Sparkles size={16} /> {busy === "solve" ? "Solving…" : "Solve & validate"}</Button><Button size="lg" variant="outline" onClick={toggleReplay} disabled={busy === "replay"}>{isPlaying ? <Pause size={16} /> : <Play size={16} />} {busy === "replay" ? "Loading SUMO…" : replay.length ? (isPlaying ? "Pause SUMO replay" : "Resume SUMO replay") : "Load SUMO replay"}</Button></div></CardContent></Card>
        <Card className="why-card reveal lift" style={{ "--d": "240ms" } as CSSProperties}><CardHeader><CardTitle><Bot size={17} /> Why did it choose this?</CardTitle></CardHeader><CardContent>{solve ? <div className="decision-list"><Decision icon={<TrafficCone />} title={closed.length ? "Closure-aware" : "Traffic-aware"} body={closed.length ? "The blocked road was removed from legal route search." : "The solver used directed road time and service windows."} /><Decision icon={<ShieldCheck />} title="Validator-first" body={solve.evaluation.feasible ? "Every vehicle, customer, time window, and depot return passed." : "The independent validator found a constraint issue."} /><Decision icon={<Activity />} title="Search trace" body={`${solve.method} evaluated ${solve.evaluations || "multiple"} candidate plans.`} /></div> : <div className="empty-explain"><Bot size={30} /><p>Run a scenario to see the reasoning chain here.</p></div>}</CardContent></Card>
      </aside>
      <section className="bottom-grid"><CompareStrip current={solve} previous={previousSolve} /><Card className="reveal lift" style={{ "--d": "60ms" } as CSSProperties}><CardHeader><CardTitle>Route result</CardTitle><CardDescription>The exact customer order and road path returned by the backend.</CardDescription></CardHeader><CardContent><RouteTable result={solve} />{solve && <><Separator /><div className="route-subhead">Delivery timeline</div><VehicleGantt result={solve} /><div className="route-subhead">Validator report</div><ViolationsPanel evaluation={solve.evaluation} /></>}</CardContent></Card><Card className="reveal lift" style={{ "--d": "120ms" } as CSSProperties}><CardHeader><CardTitle>Convergence</CardTitle><CardDescription>Lower objective is better. This is the optimizer’s actual evaluation trace.</CardDescription></CardHeader><CardContent><Trace values={solve?.trace?.best || []} diversity={solve?.trace?.diversity} traceKey={solve ? `${solve.method}-${solve.evaluations || 0}-${(solve.trace?.best || []).length}` : "empty"} /></CardContent></Card><Card className="timeline-card reveal" style={{ "--d": "180ms" } as CSSProperties}><CardHeader><CardTitle>What happened</CardTitle><CardDescription>A short audit trail for the current run.</CardDescription></CardHeader><CardContent><Timeline solve={solve} replay={replay} /></CardContent></Card></section>
    </main> : activeTab === "forecast" ? <ForecastTab evidence={evidence} graph={graph} /> : activeTab === "replanning" ? <ReplanningTab current={solve} previous={previousSolve} /> : <EvidenceTab evidence={evidence} />}
  </div>;
}

function Home({ onStart }: { onStart: () => void }) { return <div className="home-shell"><header className="home-nav"><button className="brand-button"><span className="brand-mark">✦</span><span>QUANTA</span></button><Button variant="outline" size="sm" onClick={onStart}>Open control room <ArrowRight size={14} /></Button></header><main className="home-main"><div className="hero-copy reveal"><Badge variant="outline">SMART ROUTING FOR DYNAMIC CITIES</Badge><h1>When the city changes,<br /><em>the route adapts.</em></h1><p>Quanta is an adaptive dispatch system for delivery fleets. It watches a directed road network, forecasts near-term traffic, and searches for a route that is legal, fast, and ready to execute.</p><div className="hero-actions"><Button size="lg" onClick={onStart}>Launch live simulation <ArrowRight size={17} /></Button><a href="#method">See the method <ChevronRight size={15} /></a></div><div className="hero-proof"><span><CheckCircle2 size={15} /> Independent validation</span><span><CheckCircle2 size={15} /> SUMO execution</span><span><CheckCircle2 size={15} /> QPSO + baselines</span></div><div className="hero-stats"><HeroStat value={4} label="OPTIMIZERS LIVE" /><HeroStat value={3} label="FORECAST HORIZONS" /><HeroStat value={5} label="JOBS IN FIXTURE" /></div></div><div className="hero-visual" aria-hidden="true"><div className="orbital orbital-one" /><div className="orbital orbital-two" /><div className="city-core"><Layers3 size={42} /><span>LIVE<br />NETWORK</span></div><div className="hero-node node-a" /><div className="hero-node node-b" /><div className="hero-node node-c" /><div className="hero-line line-a" /><div className="hero-line line-b" /><div className="hero-line line-c" /><div className="hero-float float-one"><span className="live-dot" /> FORECAST v1</div><div className="hero-float float-two">QPSO <strong>READY</strong></div></div></main><section id="method" className="home-method"><div><span className="eyebrow">THE DECISION LOOP</span><h2>From signal to safe action.</h2></div><div className="method-grid"><ScrollReveal delay="0ms"><Method number="01" icon={<Activity />} title="Observe" text="Traffic, closures, fleet position, and delivery windows enter one causal state." /></ScrollReveal><ScrollReveal delay="90ms"><Method number="02" icon={<Bot />} title="Predict" text="The graph forecaster estimates speed by road and horizon with measurable uncertainty." /></ScrollReveal><ScrollReveal delay="180ms"><Method number="03" icon={<Sparkles />} title="Optimize" text="QPSO searches a route plan, while an independent validator protects feasibility." /></ScrollReveal><ScrollReveal delay="270ms"><Method number="04" icon={<ShieldCheck />} title="Prove" text="SUMO executes the plan so the result can be replayed and inspected." /></ScrollReveal></div></section></div> }

function HeroStat({ value, label }: { value: number; label: string }) {
  const animated = useCountUp(value, 1100);
  return <div className="hero-stat"><strong>{Number.isFinite(value) ? Math.round(animated).toString() : "—"}</strong><span>{label}</span></div>;
}

function ScrollReveal({ children, delay = "0ms" }: { children: ReactNode; delay?: string }) {
  const { ref, inView } = useInView<HTMLDivElement>(0.15);
  return <div ref={ref} className={`scroll-reveal${inView ? " in-view" : ""}`} style={{ "--d": delay } as CSSProperties}>{children}</div>;
}

function EvidenceTab({ evidence }: { evidence: Evidence | null }) { const validation = (evidence?.uncertainty?.validation?.actual_coverage ?? 0) * 100; const test = (evidence?.uncertainty?.test?.actual_coverage ?? 0) * 100; const modelMae = evidence?.model_mae ?? 0; const baselineMae = evidence?.temporal_baseline_mae ?? 0; return <main className="evidence-page" key="evidence"><div className="eyebrow reveal"><Trophy size={14} /> RESEARCH EVIDENCE {evidence?.demo === true && <Badge variant="warning" className="badge-pulse">DEMO VALUES</Badge>}</div>{evidence?.demo === true && evidence?.provenance && <p className="lede reveal" style={{ "--d": "90ms" } as CSSProperties}><small>{evidence.provenance}</small></p>}<h1 className="reveal" style={{ "--d": "60ms" } as CSSProperties}>What the current system has measured.</h1><p className="lede reveal" style={{ "--d": "120ms" } as CSSProperties}>The live control room is backed by real SUMO episodes and saved Step 13–14 artifacts. Metrics stay visible even when a model is not yet the winner.</p>{!evidence?.available && <div className="evidence-note"><CircleAlert size={15} /> Saved forecaster artifacts are unavailable to this API process. Build or mount `artifacts/step13_forecaster_v1` to display measured values.</div>}<div className="evidence-grid"><Card className="reveal lift" style={{ "--d": "160ms" } as CSSProperties}><CardHeader><CardTitle>Forecast uncertainty</CardTitle><CardDescription>Residual intervals calibrated on validation data.</CardDescription></CardHeader><CardContent><MetricBar label="Validation coverage" value={validation} target={80} /><MetricBar label="Held-out test coverage" value={test} target={80} warning /><div className="evidence-note"><CircleAlert size={15} /> Test coverage is below nominal; the model remains experimental.</div></CardContent></Card><Card className="reveal lift" style={{ "--d": "220ms" } as CSSProperties}><CardHeader><CardTitle>Forecast comparison</CardTitle><CardDescription>Held-out speed-ratio aggregate.</CardDescription></CardHeader><CardContent><div className="compare-row"><span>GNN–Transformer</span><strong>{modelMae ? `${modelMae.toFixed(4)} MAE` : "Unavailable"}</strong></div><div className="compare-row"><span>Temporal-only baseline</span><strong className="good-text">{baselineMae ? `${baselineMae.toFixed(4)} MAE` : "Unavailable"}</strong></div><Separator /><p className="small-copy">This is an honest comparison: the current neural model has not yet beaten the baseline.</p></CardContent></Card><Card className="evidence-wide reveal" style={{ "--d": "280ms" } as CSSProperties}><CardHeader><CardTitle>Architecture status</CardTitle><CardDescription>What runs now and what is being integrated next.</CardDescription></CardHeader><CardContent><div className="status-grid"><StatusLine label="Directed graph + constraints" state="Live" /><StatusLine label="QPSO + PSO + ALNS references" state="Live" /><StatusLine label="SUMO replay" state="Live" /><StatusLine label="GNN–Transformer training" state="Measured" /><StatusLine label="Forecast-aware route cost" state="Measured" /><StatusLine label="Neural forecast in live loop" state="Next" /></div></CardContent></Card></div></main> }

function ForecastTab({ evidence, graph }: { evidence: Evidence | null; graph: Graph | null }) {
  const horizons = [["5 min", "near-term", "Fast reaction window"], ["10 min", "decision", "Primary replanning window"], ["15 min", "extended", "Longer look-ahead"]];
  const coverage = evidence?.uncertainty?.test?.actual_coverage;
  return <main className="evidence-page forecast-page" key="forecast"><div className="eyebrow reveal"><Gauge size={14} /> FORECAST LAYER</div><h1 className="reveal" style={{ "--d": "60ms" } as CSSProperties}>Traffic before it becomes a delay.</h1><p className="lede reveal" style={{ "--d": "120ms" } as CSSProperties}>The forecaster is evaluated by road and horizon. This view keeps the three decision horizons visible and labels what is already measured versus what is still being connected to live serving.</p><div className="pipeline-cue reveal" style={{ "--d": "160ms" } as CSSProperties}><span>OBSERVE</span><ArrowRight size={14} /><span>PREDICT</span><ArrowRight size={14} /><span>REPLAN</span></div><div className="horizon-grid pipeline">{horizons.map(([label, kind, detail], index) => <Card className="horizon-card pipeline-step reveal" style={{ "--d": `${180 + index * 90}ms` } as CSSProperties} key={label}><span className="horizon-step">{`0${index + 1}`}</span><span className="horizon-label">{label}</span><Badge variant="outline">{kind}</Badge><strong>{detail}</strong><small>Road-level speed ratio target</small></Card>)}</div><div className="evidence-grid"><Card className="reveal lift" style={{ "--d": "200ms" } as CSSProperties}><CardHeader><CardTitle>Road forecast preview</CardTitle><CardDescription>{graph ? `${graph.edges.length} legal directed roads in ${graph.scenario_id}` : "Loading graph…"}</CardDescription></CardHeader><CardContent>{graph ? <div className="road-forecast-list">{graph.edges.slice(0, 7).map((edge, index) => <div className="road-forecast road-row-in" style={{ "--d": `${index * 70}ms` } as CSSProperties} key={edge.id}><div><span>{edge.id}</span><small>{edge.from} → {edge.to}</small></div><Progress value={Math.min(100, Math.max(12, edge.speed_mps * 4))} /><em>{edge.speed_mps.toFixed(1)} m/s free flow</em></div>)}</div> : <div className="road-forecast-list"><div className="skeleton skeleton-row" /><div className="skeleton skeleton-row" /><div className="skeleton skeleton-row" /></div>}<p className="small-copy">This is the live network profile used by the planner. Neural road predictions are measured offline in the evidence artifact and are not presented here as fake live values.</p></CardContent></Card><Card className="reveal lift" style={{ "--d": "260ms" } as CSSProperties}><CardHeader><CardTitle>Uncertainty contract</CardTitle><CardDescription>Versioned forecast metadata for planner integration.</CardDescription></CardHeader><CardContent><div className="compare-row"><span>Forecast version</span><strong>forecaster_v1</strong></div><div className="compare-row"><span>Issue time</span><strong>Per forecast request</strong></div><div className="compare-row"><span>Held-out coverage</span><strong className={coverage && coverage >= 0.8 ? "good-text" : "warning-text"}>{coverage == null ? "Unavailable" : `${(coverage * 100).toFixed(1)}%`}</strong></div><Separator /><p className="small-copy">Known closures remain hard constraints. Beyond the served horizon, the planner uses the declared current/time-of-day extension.</p></CardContent></Card></div></main>
}

function ReplanningTab({ current, previous }: { current: SolveResult | null; previous: SolveResult | null }) {
  const planEntries = (plan: Record<string, string[]> | undefined) => Object.entries(plan || {});
  const [drl, setDrl] = useState<DrlDemo | null>(null);
  useEffect(() => { api<DrlDemo>("/api/demo/drl").then(setDrl).catch(() => setDrl(null)); }, []);
  return <main className="evidence-page replanning-page" key="replanning"><div className="eyebrow reveal"><Sparkles size={14} /> REPLANNING TRACE</div><h1 className="reveal" style={{ "--d": "60ms" } as CSSProperties}>See what changed, and why.</h1><p className="lede reveal" style={{ "--d": "120ms" } as CSSProperties}>Run the same scenario again after changing an incident to compare the previous plan with the current validated plan.</p><CompareStrip current={current} previous={previous} /><div className="plan-compare"><Card className="plan-box reveal lift" style={{ "--d": "180ms" } as CSSProperties}><CardHeader><CardTitle>Before</CardTitle><CardDescription>{previous ? "Previous solve in this session" : "No previous solve captured"}</CardDescription></CardHeader><CardContent>{previous ? planEntries(previous.plan).map(([vehicle, stops]) => <div className="plan-line" key={vehicle}><strong>{vehicle}</strong><span>{stops.join(" → ") || "No stops"}</span></div>) : <div className="empty-table"><Route size={22} /><span>Run solve twice to create a before/after comparison.</span></div>}</CardContent></Card><Card className="plan-box reveal lift" style={{ "--d": "240ms" } as CSSProperties}><CardHeader><CardTitle>After</CardTitle><CardDescription>{current ? `${current.status} · ${current.evaluation.time_s.toFixed(0)}s` : "Current plan appears here"}</CardDescription></CardHeader><CardContent>{current ? planEntries(current.plan).map(([vehicle, stops]) => <div className="plan-line" key={vehicle}><strong>{vehicle}</strong><span>{stops.join(" → ") || "No stops"}</span></div>) : <div className="empty-table"><Sparkles size={22} /><span>Run a scenario from Live Control Room.</span></div>}</CardContent></Card></div><div className="evidence-grid"><Card className="reveal lift" style={{ "--d": "300ms" } as CSSProperties}><CardHeader><CardTitle>Change driver</CardTitle><CardDescription>The event that can force a new route.</CardDescription></CardHeader><CardContent><div className="route-change"><CircleAlert size={20} /><div><strong>{current?.closed_edge_ids.length ? current.closed_edge_ids.join(", ") : "No closure selected"}</strong><p>{current?.closed_edge_ids.length ? "Closed roads were removed from legal search." : "Change an incident to create a controlled comparison."}</p></div></div></CardContent></Card><Card className="reveal lift" style={{ "--d": "360ms" } as CSSProperties}><CardHeader><CardTitle>QPSO convergence</CardTitle><CardDescription>Search effort and validator outcome.</CardDescription></CardHeader><CardContent><div className="compare-row"><span>Evaluations</span><strong>{current?.evaluations ?? "—"}</strong></div><div className="compare-row"><span>Best objective</span><strong>{current?.trace?.best?.at(-1)?.toFixed(1) ?? "—"}</strong></div><div className="compare-row"><span>Validator</span><strong className={current?.evaluation.feasible ? "good-text" : "warning-text"}>{current ? (current.evaluation.feasible ? "PASS" : "REPAIR") : "—"}</strong></div></CardContent></Card>{drl && <Card className="reveal lift" style={{ "--d": "420ms" } as CSSProperties}><CardHeader><CardTitle>Scope decision</CardTitle><CardDescription>Rule-baseline replanning scope {drl.demo === true && <Badge variant="warning">DEMO VALUES</Badge>}</CardDescription></CardHeader><CardContent><div className="plan-line"><strong>{drl.action}</strong><span>{drl.scope === "job" ? `job ${drl.job_id} · vehicle ${drl.vehicle_id}` : drl.scope || "—"}</span></div><p className="small-copy">{drl.reason}</p></CardContent></Card>}</div></main>
}

function CompareStrip({ current, previous }: { current: SolveResult | null; previous: SolveResult | null }) {
  const objective = current?.evaluation.objective ?? NaN;
  const animatedObjective = useCountUp(Number.isFinite(objective) ? objective : 0);
  if (!current) return <Card className="compare-strip reveal" style={{ "--d": "0ms" } as CSSProperties}><CardContent><div className="empty-table"><Gauge size={22} /><span>Solve a route to compare optimizer runs.</span></div></CardContent></Card>;
  const delta = previous ? previous.evaluation.objective - current.evaluation.objective : null;
  const improved = delta != null && delta > 0.05;
  const regressed = delta != null && delta < -0.05;
  return <Card className="compare-strip reveal" style={{ "--d": "0ms" } as CSSProperties}>
    <CardContent>
      <div className="compare-stats">
        <div className="compare-stat"><span>Method</span><strong>{current.method}</strong></div>
        <div className="compare-stat"><span>Evaluations</span><strong>{current.evaluations ?? "—"}</strong></div>
        <div className="compare-stat"><span>Solve time</span><strong>{current.elapsed_s.toFixed(2)} s</strong></div>
        <div className="compare-stat"><span>Objective</span><strong>{Number.isFinite(objective) ? animatedObjective.toFixed(1) : "—"}</strong></div>
      </div>
      {delta == null
        ? <Badge variant="secondary" className="delta-badge"><Minus size={12} /> FIRST SOLVE</Badge>
        : improved
          ? <Badge variant="success" className="delta-badge badge-pulse"><TrendingDown size={12} /> IMPROVED −{delta.toFixed(1)} vs previous</Badge>
          : regressed
            ? <Badge variant="warning" className="delta-badge badge-pulse"><TrendingUp size={12} /> REGRESSED +{Math.abs(delta).toFixed(1)} vs previous</Badge>
            : <Badge variant="secondary" className="delta-badge"><Minus size={12} /> TIED ±{Math.abs(delta).toFixed(1)} vs previous</Badge>}
    </CardContent>
  </Card>;
}

function NetworkMap({ graph, routeEdges, closed, movers, replayPct }: { graph: Graph | null; routeEdges: Set<string>; closed: string[]; movers: ReplayFrame["vehicles"]; replayPct: number }) {
  const reduced = usePrefersReducedMotion();
  const posRef = useRef<Record<string, { x: number; y: number }>>({});
  const [, setTick] = useState(0);

  // Smooth vehicle interpolation: ease displayed positions toward each new
  // replay frame with a short rAF lerp instead of jump-cutting between frames.
  useEffect(() => {
    const ids = new Set(movers.map((m) => m.id));
    for (const key of Object.keys(posRef.current)) {
      if (!ids.has(key)) delete posRef.current[key];
    }
    for (const m of movers) {
      if (!posRef.current[m.id]) posRef.current[m.id] = { x: m.x, y: m.y };
    }
    if (reduced || movers.length === 0) {
      const snap: Record<string, { x: number; y: number }> = {};
      for (const m of movers) snap[m.id] = { x: m.x, y: m.y };
      posRef.current = snap;
      setTick((t) => t + 1);
      return;
    }
    let raf = 0;
    const started = performance.now();
    const step = () => {
      let moving = false;
      for (const m of movers) {
        const p = posRef.current[m.id] ?? { x: m.x, y: m.y };
        const nx = p.x + (m.x - p.x) * 0.35;
        const ny = p.y + (m.y - p.y) * 0.35;
        if (Math.abs(nx - m.x) > 0.02 || Math.abs(ny - m.y) > 0.02) {
          moving = true;
          posRef.current[m.id] = { x: nx, y: ny };
        } else {
          posRef.current[m.id] = { x: m.x, y: m.y };
        }
      }
      setTick((t) => t + 1);
      if (moving && performance.now() - started < 800) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [movers, reduced]);

  if (!graph) return <div className="map-wrap"><div className="map-loading"><div className="skeleton skeleton-map" /><div className="skeleton skeleton-line" /></div></div>;
  const xs = graph.nodes.map((n) => n.x), ys = graph.nodes.map((n) => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const point = (n: { x: number; y: number }) => ({ x: 40 + ((n.x - minX) / Math.max(1, maxX - minX)) * 720, y: 360 - ((n.y - minY) / Math.max(1, maxY - minY)) * 300 });
  const nodes = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
  return <div className="map-wrap"><svg viewBox="0 0 800 400" role="img" aria-label="Directed city road network"><defs><filter id="glow"><feGaussianBlur stdDeviation="4" result="coloredBlur" /><feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge></filter><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9" fill="none" stroke="#5b6a58" strokeWidth="1.6" /></marker></defs>{graph.edges.map((edge) => { const a = point(nodes[edge.from]), b = point(nodes[edge.to]); const isRoute = routeEdges.has(edge.id); const isClosed = closed.includes(edge.id) || !edge.open; return <line key={edge.id} x1={a.x} y1={a.y} x2={b.x} y2={b.y} className={`map-edge ${isRoute ? "map-route" : ""} ${isClosed ? "map-closed" : ""}`} markerEnd="url(#arrow)" />; })}{graph.edges.filter((edge) => closed.includes(edge.id) || !edge.open).map((edge) => { const a = point(nodes[edge.from]), b = point(nodes[edge.to]); return <g key={`closure-${edge.id}`} className="closure-marker"><circle cx={(a.x + b.x) / 2} cy={(a.y + b.y) / 2} r="5" className="closure-core" /><circle cx={(a.x + b.x) / 2} cy={(a.y + b.y) / 2} r="5" className="closure-ping" /></g>; })}{graph.nodes.map((node) => { const p = point(node); const isDepot = node.kind.includes("depot"); return <g key={node.id}>{isDepot && <circle cx={p.x} cy={p.y} r={16} className="depot-halo" />}<circle cx={p.x} cy={p.y} r={isDepot ? 10 : 6} className={isDepot ? "map-depot" : "map-node"} /><text x={p.x + 9} y={p.y - 9} className="map-label">{node.id}</text></g>; })}{graph.requests.map((job) => { const p = point(nodes[job.node]); return <g key={job.id}><circle cx={p.x} cy={p.y} r="3" className="map-job" /><text x={p.x + 9} y={p.y + 15} className="map-job-label">{job.id}</text></g>; })}{movers.map((mover) => { const raw = posRef.current[mover.id] ?? { x: mover.x, y: mover.y }; const p = point(raw); return <circle key={mover.id} cx={p.x} cy={p.y} r={mover.kind === "delivery" ? 8 : 4} className={mover.kind === "delivery" ? "map-mover delivery" : "map-mover traffic"} />; })}</svg><div className="replay-progress" aria-hidden="true"><span style={{ width: `${Math.min(100, Math.max(0, replayPct))}%` }} /></div><div className="map-legend"><span><i className="legend-line route" />Selected route</span><span><i className="legend-line closed" />Closure</span><span><i className="legend-dot car" />SUMO vehicle</span><span><i className="legend-dot job" />Delivery</span></div></div> }

function Step({ label, active }: { label: string; active?: boolean }) { return <div className={`step ${active ? "step-active" : ""}`}><span className="step-dot" />{label}</div> }
function Kpi({ icon, label, value, accent = "", delay = "0ms" }: { icon: ReactNode; label: string; value: string; accent?: string; delay?: string }) { return <Card className="kpi lift reveal" style={{ "--d": delay } as CSSProperties}><CardContent><span className="kpi-icon">{icon}</span><div><span className="kpi-label">{label}</span><strong className={accent}>{value}</strong></div></CardContent></Card> }
function CountKpi({ icon, label, value, format, delay = "0ms" }: { icon: ReactNode; label: string; value: number | null; format: (n: number) => string; delay?: string }) {
  const animated = useCountUp(value ?? 0);
  const display = value == null || !Number.isFinite(value) ? "—" : format(animated);
  return <Card className="kpi lift reveal" style={{ "--d": delay } as CSSProperties}><CardContent><span className="kpi-icon">{icon}</span><div><span className="kpi-label">{label}</span><strong>{display}</strong></div></CardContent></Card>;
}
function Field({ label, children }: { label: string; children: React.ReactNode }) { return <label className="field"><span>{label}</span>{children}</label> }
function Decision({ icon, title, body }: { icon: ReactNode; title: string; body: string }) { return <div className="decision"><span className="decision-icon">{icon}</span><div><strong>{title}</strong><p>{body}</p></div></div> }
function Method({ number, icon, title, text }: { number: string; icon: ReactNode; title: string; text: string }) { return <Card className="method-card lift"><span className="method-number">{number}</span><span className="method-icon">{icon}</span><CardTitle>{title}</CardTitle><CardDescription>{text}</CardDescription></Card> }
function RouteTable({ result }: { result: SolveResult | null }) { if (!result) return <div className="empty-table"><Route size={24} /><span>Run the solver to see the route explanation.</span></div>; return <div className="route-table">{result.evaluation.vehicles.map((v) => <div className="route-row" key={v.id}><span className={`vehicle-dot ${v.id === "V2" ? "blue" : "lime"}`} /> <strong>{v.id}</strong><span>{v.order.join(" → ") || "No stops"}</span><small>{v.elapsed_s.toFixed(0)}s · {v.edge_ids.length} roads</small></div>)}</div> }

function VehicleGantt({ result }: { result: SolveResult }) {
  const vehicles = result.evaluation.vehicles;
  if (!vehicles.length) return <div className="empty-table"><span>No vehicles in this plan.</span></div>;
  const maxT = Math.max(1, ...vehicles.map((v) => v.elapsed_s));
  return <div className="gantt" role="img" aria-label="Per-vehicle delivery timeline">{vehicles.map((v, index) => {
    const stops = [...(v.stops || [])].sort((a, b) => a.start_s - b.start_s);
    return <div className="gantt-row" key={v.id}>
      <span className="gantt-label"><span className={`vehicle-dot ${v.id === "V2" ? "blue" : "lime"}`} />{v.id} <em>· {v.order.length} stops</em></span>
      <div className="gantt-track">
        <div className="gantt-fill" style={{ width: `${Math.min(100, (v.elapsed_s / maxT) * 100)}%`, animationDelay: `${index * 90}ms` }} />
        {stops.map((stop) => <i key={`${v.id}-${stop.job}`} className="gantt-stop" style={{ left: `${Math.min(100, (stop.start_s / maxT) * 100)}%` }} title={`${stop.job} @ ${stop.start_s.toFixed(0)}s`} />)}
      </div>
      <small>{v.elapsed_s.toFixed(0)}s</small>
    </div>;
  })}<div className="gantt-scale"><span>0s</span><span>{(maxT / 2).toFixed(0)}s</span><span>{maxT.toFixed(0)}s</span></div></div>;
}

function ViolationsPanel({ evaluation }: { evaluation: Evaluation }) {
  const violations = evaluation.violations || [];
  if (!violations.length) return <div className="violations violations-none"><CheckCircle2 size={14} /><span>No violations — the validator passed every hard constraint.</span></div>;
  return <div className="violations">{violations.map((violation, index) => <div className="violation-row" key={`${violation.name}-${index}`}><AlertTriangle size={14} /><div><strong>{violation.name}</strong><p>{violation.detail}</p></div></div>)}</div>;
}

function Timeline({ solve, replay }: { solve: SolveResult | null; replay: ReplayFrame[] }) { const events = solve ? [{ label: "Plan validated", detail: solve.evaluation.feasible ? "All hard constraints pass" : "Validator requested repair" }, { label: "Route selected", detail: `${solve.evaluation.vehicles.length} delivery vehicles assigned` }, { label: "SUMO replay", detail: replay.length ? `${replay.length} frames ready` : "Replay not loaded" }] : [{ label: "Waiting for scenario", detail: "Choose an incident and solve to begin" }]; return <div className="timeline">{events.map((event, index) => <div className="timeline-item cascade" style={{ "--d": `${index * 90}ms` } as CSSProperties} key={event.label}><span className={`timeline-dot ${index === 0 && solve ? "active" : ""}`} /><div><strong>{event.label}</strong><small>{event.detail}</small></div></div>)}</div> }

function Trace({ values, diversity, traceKey }: { values: number[]; diversity?: number[]; traceKey: string }) {
  const last = values.length ? values[values.length - 1] : NaN;
  const animatedLast = useCountUp(Number.isFinite(last) ? last : 0);
  if (values.length < 2) return <div className="empty-table"><Activity size={24} /><span>Solve a route to reveal the search trace.</span></div>;
  const min = Math.min(...values), max = Math.max(...values);
  const project = (v: number) => 92 - ((v - min) / Math.max(1e-6, max - min)) * 72;
  const points = values.map((v, i) => `${(i / (values.length - 1)) * 100},${project(v).toFixed(1)}`).join(" ");
  const headX = 100, headY = project(last);
  const hasDiversity = !!diversity && diversity.length >= 2;
  const dMin = hasDiversity ? Math.min(...diversity!) : 0;
  const dMax = hasDiversity ? Math.max(...diversity!) : 1;
  const dPoints = hasDiversity ? diversity!.map((v, i) => `${(i / (diversity!.length - 1)) * 100},${(92 - ((v - dMin) / Math.max(1e-6, dMax - dMin)) * 72).toFixed(1)}`).join(" ") : "";
  const improvement = values[0] - last;
  return <div className="trace-chart"><svg key={traceKey} viewBox="0 0 100 100" preserveAspectRatio="none">{hasDiversity && <polyline points={dPoints} fill="none" className="trace-diversity" vectorEffect="non-scaling-stroke" />}<polyline points={points} fill="none" stroke="currentColor" strokeWidth="2.5" vectorEffect="non-scaling-stroke" pathLength={100} className="trace-line" /><circle cx={headX} cy={headY} r={3} className="trace-head" /></svg><div className="trace-legend"><span className="trace-legend-item"><i className="legend-line route" />Best</span>{hasDiversity && <span className="trace-legend-item"><i className="legend-line diversity" />Diversity</span>}{Number.isFinite(improvement) && improvement > 0 && <span className="trace-legend-item good">−{improvement.toFixed(1)} from start</span>}</div><div className="trace-caption"><span>Best objective</span><strong>{Number.isFinite(last) ? animatedLast.toFixed(1) : "—"}</strong></div></div>;
}
function MetricBar({ label, value, target, warning }: { label: string; value: number; target: number; warning?: boolean }) { const animated = useCountUp(Number.isFinite(value) ? value : 0); return <div className="metric-bar"><div><span>{label}</span><strong className={warning ? "warning-text" : ""}>{Number.isFinite(value) ? `${animated.toFixed(1)}%` : "—"}</strong></div><Progress value={value} /><small>Nominal target {target}%</small></div> }
function StatusLine({ label, state }: { label: string; state: string }) { return <div className="status-line"><span>{label}</span><Badge variant={state === "Live" ? "success" : state === "Measured" ? "warning" : "secondary"}>{state}</Badge></div> }

export default App;
