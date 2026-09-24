import { useEffect, useRef, useState } from "react";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  Copy,
  Download,
  FileUp,
  Map,
  Plus,
  Save,
  Search,
  ShieldCheck,
  Trash2,
  Truck,
  Upload,
  Waypoints,
  X,
} from "lucide-react";
import { api, postJson, ScenarioSummary } from "../api";
import { useWorkspace } from "./Store";
import { useRunner } from "./Runner";
import {
  clone,
  createDraft,
  dateLabel,
  Delivery,
  download,
  Draft,
  generateNetwork,
  methods,
  Network,
  parseCsv,
  requestBody,
  uid,
  validateDraft,
} from "./data";
import {
  Button,
  Empty,
  Field,
  NetworkView,
  Notice,
  PageTitle,
  Panel,
  Steps,
  Tag,
} from "./ui";

export function useCatalog() {
  const [catalog, setCatalog] = useState<ScenarioSummary[]>([]),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api<{ scenarios: ScenarioSummary[] }>("/api/scenarios")
      .then((r) => {
        if (active) setCatalog(r.scenarios);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, []);
  return { catalog, error };
}
export function ScenarioList() {
  const store = useWorkspace(),
    navigate = useNavigate();
  const { catalog, error } = useCatalog();
  const [query, setQuery] = useState(""),
    [message, setMessage] = useState(""),
    [remove, setRemove] = useState<string | null>(null);
  async function load(item: ScenarioSummary) {
    try {
      const graph = await api<Network>(`/api/scenarios/${item.id}`);
      const draft = {
        ...createDraft(item.label, normalizeNetwork(graph)),
        networkId: item.id,
        source: "catalog" as const,
      };
      await store.saveDraft(draft);
      navigate(`/app/scenarios/${draft.id}`);
    } catch (e) {
      setMessage((e as Error).message);
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="PLANNING WORKSPACE"
        title="Scenarios"
        description="Build, organise and rerun the conditions your fleet operates in."
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
      {message && <Notice tone="red">{message}</Notice>}
      <Panel
        title="Your scenarios"
        subtitle={`${store.drafts.length} saved configurations`}
        action={
          <div className="w-table-search">
            <Search size={16} />
            <input
              aria-label="Filter scenarios"
              placeholder="Find a scenario…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
        }
      >
        {store.drafts.length ? (
          <div className="w-table-scroll">
            <table className="w-table">
              <thead>
                <tr>
                  <th>Scenario</th>
                  <th>Network</th>
                  <th>Fleet / deliveries</th>
                  <th>Conditions</th>
                  <th>Updated</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {store.drafts
                  .filter((d) =>
                    d.name.toLowerCase().includes(query.toLowerCase()),
                  )
                  .map((d) => (
                    <tr key={d.id}>
                      <td>
                        <Link
                          className="w-table-primary"
                          to={`/app/scenarios/${d.id}`}
                        >
                          <span className="w-table-symbol">
                            <Waypoints size={18} />
                          </span>
                          <span>
                            {d.name}
                            <small>{d.description}</small>
                          </span>
                        </Link>
                      </td>
                      <td>
                        {d.graph.nodes.length} nodes
                        <small>
                          {d.source === "generated"
                            ? "Synthetic network"
                            : d.networkId}
                        </small>
                      </td>
                      <td>
                        {d.graph.fleet.length} vehicles
                        <small>{d.graph.requests.length} deliveries</small>
                      </td>
                      <td>
                        <Tag tone={d.closures.length ? "amber" : "blue"}>
                          {d.closures.length
                            ? `${d.closures.length} closures`
                            : d.traffic < 1
                              ? "Reduced speeds"
                              : "Normal"}
                        </Tag>
                      </td>
                      <td>{dateLabel(d.updatedAt)}</td>
                      <td>
                        <div className="w-row-actions">
                          <button
                            aria-label={`Duplicate ${d.name}`}
                            disabled={!store.canEdit}
                            onClick={async () => {
                              try {
                                await store.saveDraft({
                                  ...clone(d),
                                  id: uid(),
                                  name: `${d.name} (copy)`,
                                  updatedAt: new Date().toISOString(),
                                });
                              } catch (e) {
                                setMessage((e as Error).message);
                              }
                            }}
                          >
                            <Copy size={16} />
                          </button>
                          <button
                            aria-label={`Delete ${d.name}`}
                            disabled={!store.canEdit}
                            onClick={() => setRemove(d.id)}
                          >
                            <Trash2 size={16} />
                          </button>
                          <Link
                            aria-label={`Edit ${d.name}`}
                            to={`/app/scenarios/${d.id}`}
                          >
                            <ChevronRight size={17} />
                          </Link>
                        </div>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty
            title="Your first scenario starts here"
            action={
              <Button onClick={() => navigate("/app/scenarios/new")}>
                Create scenario
              </Button>
            }
          >
            Define a network, add vehicles and deliveries, then run an
            optimiser.
          </Empty>
        )}
      </Panel>
      <div className="w-section-heading">
        <div>
          <h2>Network library</h2>
          <p>
            Existing project scenarios. Delhi data is detected automatically
            when its files are added.
          </p>
        </div>
        <Tag>{catalog.filter((c) => c.available).length} available</Tag>
      </div>
      {error && (
        <Notice tone="amber">
          The API is unavailable. Saved and generated scenarios can still be
          edited.
        </Notice>
      )}
      <div className="w-library-grid">
        {catalog.map((item) => (
          <article
            className={`w-library-card ${!item.available ? "unavailable" : ""}`}
            key={item.id}
          >
            <div>
              <Map size={22} />
              <Tag tone={item.available ? "green" : "amber"}>
                {item.available ? "Available" : "Data unavailable"}
              </Tag>
            </div>
            <h3>{item.label}</h3>
            <p>{item.description}</p>
            <Button
              variant="secondary"
              disabled={!item.available || !store.canEdit}
              onClick={() => load(item)}
            >
              {item.available ? "Use network" : "Waiting for local files"}
              <ArrowRight size={15} />
            </Button>
          </article>
        ))}
      </div>
      {remove && (
        <div className="w-modal-backdrop">
          <div
            className="w-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Delete scenario"
          >
            <h2>Delete this scenario?</h2>
            <p>Saved run snapshots remain available in Results.</p>
            <div className="w-modal-actions">
              <Button variant="secondary" onClick={() => setRemove(null)}>
                Keep scenario
              </Button>
              <Button
                variant="danger"
                onClick={async () => {
                  try {
                    await store.deleteDraft(remove);
                    setRemove(null);
                  } catch (e) {
                    setMessage((e as Error).message);
                    setRemove(null);
                  }
                }}
              >
                Delete scenario
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export function normalizeNetwork(raw: any): Network {
  // Accept both a UI graph export and the repository's canonical scenario format.
  if (raw.nodes?.[0]?.node_id)
    return {
      scenario_id: raw.scenario_id || "UPLOADED",
      graph_version: raw.graph_version || "upload-v1",
      reference_plan: {},
      nodes: raw.nodes.map((n: any) => ({
        id: n.node_id,
        x: n.x_m,
        y: n.y_m,
        kind: n.kind,
        zone: n.zone_id,
      })),
      edges: raw.edges.map((e: any) => ({
        id: e.edge_id,
        from: e.from_node,
        to: e.to_node,
        length_m: e.length_m,
        speed_mps: e.speed_limit_mps,
        road_class: e.road_class,
        open: e.open_by_default,
        free_flow_s: e.length_m / e.speed_limit_mps,
      })),
      requests: raw.requests.map((j: any) => ({
        id: j.request_id,
        node: j.access_node_id,
        demand: j.demand,
        earliest_s: j.earliest_service_start_s,
        latest_s: j.latest_service_start_s,
        service_s: j.service_duration_s,
      })),
      fleet: raw.fleet.map((v: any) => ({
        id: v.vehicle_id,
        capacity: v.capacity,
        depot: v.depot_node_id,
      })),
    };
  if (
    !Array.isArray(raw.nodes) ||
    !Array.isArray(raw.edges) ||
    !raw.nodes.length
  )
    throw new Error("Network JSON needs nodes and edges arrays.");
  return {
    ...raw,
    scenario_id: raw.scenario_id || "UPLOADED",
    graph_version: raw.graph_version || "upload-v1",
    reference_plan: {},
    fleet: raw.fleet?.length
      ? raw.fleet
      : [{ id: "V1", capacity: 100, depot: raw.nodes[0].id }],
    requests: (raw.requests?.length
      ? raw.requests
      : [{ id: "D001", node: raw.nodes[1]?.id, demand: 10 }]
    ).map((j: any) => ({
      ...j,
      earliest_s: j.earliest_s ?? 0,
      latest_s: j.latest_s ?? 14400,
      service_s: j.service_s ?? 30,
    })),
  };
}

function graphmlNetwork(text: string): Network {
  const document = new DOMParser().parseFromString(text, "application/xml");
  if (document.querySelector("parsererror"))
    throw new Error("This GraphML file is not valid XML.");
  const keys = new globalThis.Map(
    [...document.querySelectorAll("key")].map((k) => [
      k.getAttribute("id"),
      k.getAttribute("attr.name"),
    ]),
  );
  const values = (element: Element) =>
    Object.fromEntries(
      [...element.querySelectorAll(":scope > data")].map((d) => [
        keys.get(d.getAttribute("key")) || d.getAttribute("key"),
        d.textContent,
      ]),
    );
  const sourceNodes = [...document.querySelectorAll("node")];
  if (sourceNodes.length < 2)
    throw new Error("The network must contain at least two nodes.");
  const nodes = sourceNodes.map((n, i) => {
    const data = values(n);
    return {
      id: n.getAttribute("id")!,
      x: Number(data.x ?? (i % 6) * 300),
      y: Number(data.y ?? Math.floor(i / 6) * 300),
      kind: i === 0 ? "depot" : "customer_access",
      zone: "Z1",
    };
  });
  const nodesById = new globalThis.Map(nodes.map((n) => [n.id, n]));
  const edges = [...document.querySelectorAll("edge")].map((e, i) => {
    const d = values(e),
      a = nodesById.get(e.getAttribute("source")!),
      b = nodesById.get(e.getAttribute("target")!);
    if (!a || !b) throw new Error("A road references an unknown node.");
    return {
      id: `E${i}`,
      from: a.id,
      to: b.id,
      length_m: Math.max(
        1,
        Number(d.length ?? Math.hypot(a.x - b.x, a.y - b.y)),
      ),
      speed_mps: Number(d.speed_mps || 10),
      road_class: "local",
      open: true,
      free_flow_s: 0,
    };
  });
  if (
    document.querySelector("graph")?.getAttribute("edgedefault") !== "directed"
  )
    edges.push(
      ...edges.map((e) => ({
        ...e,
        id: `${e.id}-reverse`,
        from: e.to,
        to: e.from,
      })),
    );
  return normalizeNetwork({ nodes, edges });
}

export function ScenarioWizard() {
  const { id } = useParams();
  const store = useWorkspace(),
    runner = useRunner(),
    navigate = useNavigate();
  const { catalog } = useCatalog();
  const existing = store.drafts.find((d) => d.id === id);
  const [draft, setDraft] = useState<Draft>(() =>
    clone(existing || createDraft()),
  );
  const [step, setStep] = useState(0),
    [error, setError] = useState(""),
    [saved, setSaved] = useState(false),
    [busy, setBusy] = useState(false);
  const [source, setSource] = useState("generated"),
    [size, setSize] = useState(24),
    [seed, setSeed] = useState(7);
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (existing) setDraft(clone(existing));
    else if (!id) setDraft(createDraft());
  }, [id]);
  function update(patch: Partial<Draft>) {
    setDraft((d) => ({ ...d, ...patch }));
    setSaved(false);
    setError("");
  }
  async function load(id: string) {
    setBusy(true);
    try {
      const graph = normalizeNetwork(
        await api<Network>(`/api/scenarios/${id}`),
      );
      update({ graph, source: "catalog", networkId: id, closures: [] });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function upload(file?: File) {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      if (file.size > 12 * 1024 * 1024)
        throw new Error("Use a network file smaller than 12 MB.");
      const text = await file.text();
      const graph = file.name.toLowerCase().endsWith(".graphml")
        ? graphmlNetwork(text)
        : normalizeNetwork(JSON.parse(text));
      const next = {
        ...draft,
        graph,
        source: "upload" as const,
        networkId: file.name,
        closures: [],
      };
      await postJson("/api/workspace/validate", requestBody(next));
      update(next);
    } catch (e) {
      setError(`Upload failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }
  async function save(run = false) {
    const invalid = validateDraft(draft);
    if (invalid) {
      setError(invalid);
      return;
    }
    setBusy(true);
    setError("");
    try {
      const next = { ...draft, updatedAt: new Date().toISOString() };
      await store.saveDraft(next);
      setSaved(true);
      if (run) {
        runner.start(next).catch((e) => store.setError(e.message));
        navigate("/app/run");
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  if (id && !existing)
    return (
      <Empty
        title="Scenario not found"
        action={<Link to="/app/scenarios">Back to scenarios</Link>}
      >
        This scenario is not available in the selected workspace.
      </Empty>
    );
  return (
    <>
      <PageTitle
        eyebrow="SCENARIO BUILDER"
        title={id ? draft.name : "Create a new scenario"}
        description="Bring the network, fleet and operating conditions into one repeatable configuration."
        action={
          <Button
            variant="secondary"
            disabled={busy || !store.canEdit}
            onClick={() => save()}
          >
            <Save size={16} />
            {saved ? "Draft saved" : "Save draft"}
          </Button>
        }
      />
      {!store.canEdit && (
        <Notice tone="amber">
          You have viewer access. You can inspect this configuration; saving and
          running require dispatcher access.
        </Notice>
      )}
      <Steps
        labels={[
          "Road network",
          "Fleet & demands",
          "Traffic conditions",
          "Algorithm & run",
        ]}
        current={step}
        onSelect={setStep}
      />
      {error && <Notice tone="red">{error}</Notice>}
      {saved && (
        <Notice tone="green">
          Scenario saved to{" "}
          {store.demo ? "this demo workspace" : "your company workspace"}.
        </Notice>
      )}
      <div className="w-builder-layout">
        <div className="w-builder-main">
          {step === 0 && (
            <>
              <Panel
                title="Scenario details"
                subtitle="A clear name helps your team find and compare this scenario."
              >
                <div className="w-form-body w-form-grid">
                  <Field label="Scenario name">
                    <input
                      value={draft.name}
                      onChange={(e) => update({ name: e.target.value })}
                      maxLength={120}
                    />
                  </Field>
                  <Field label="Description">
                    <input
                      value={draft.description}
                      onChange={(e) => update({ description: e.target.value })}
                      placeholder="What are you testing?"
                    />
                  </Field>
                </div>
              </Panel>
              <Panel
                title="Choose a road network"
                subtitle="Start from a sample, an existing network or your own data."
              >
                <div className="w-source-tabs">
                  {[
                    ["generated", "Generate network"],
                    ["catalog", "Use project network"],
                    ["upload", "Upload network"],
                  ].map(([value, label]) => (
                    <button
                      key={value}
                      className={source === value ? "active" : ""}
                      onClick={() => setSource(value)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <div className="w-form-body">
                  {source === "generated" && (
                    <>
                      <Notice>
                        A synthetic network is useful for testing the full
                        workflow without external map files.
                      </Notice>
                      <div className="w-form-grid">
                        <Field label="Road nodes">
                          <select
                            value={size}
                            onChange={(e) => setSize(Number(e.target.value))}
                          >
                            {[12, 24, 40, 60].map((n) => (
                              <option key={n} value={n}>
                                {n} nodes
                              </option>
                            ))}
                          </select>
                        </Field>
                        <Field label="Network seed">
                          <input
                            type="number"
                            min={1}
                            max={9999}
                            value={seed}
                            onChange={(e) => setSeed(Number(e.target.value))}
                          />
                        </Field>
                      </div>
                      <Button
                        variant="secondary"
                        onClick={() =>
                          update({
                            graph: generateNetwork(
                              size,
                              Math.min(12, size - 1),
                              4,
                              seed,
                            ),
                            source: "generated",
                            networkId: `SYNTHETIC_${seed}`,
                            closures: [],
                            seed,
                          })
                        }
                      >
                        <Waypoints size={16} />
                        Generate network
                      </Button>
                    </>
                  )}
                  {source === "catalog" && (
                    <div className="w-network-options">
                      {catalog.map((item) => (
                        <button
                          key={item.id}
                          disabled={!item.available || busy}
                          className={
                            draft.networkId === item.id ? "selected" : ""
                          }
                          onClick={() => load(item.id)}
                        >
                          <Map size={19} />
                          <span>
                            <strong>{item.label}</strong>
                            <small>
                              {item.available
                                ? item.role
                                : "Data unavailable · add the existing local files to enable"}
                            </small>
                          </span>
                          <Tag tone={item.available ? "green" : "amber"}>
                            {item.available ? "Available" : "Waiting for data"}
                          </Tag>
                        </button>
                      ))}
                    </div>
                  )}
                  {source === "upload" && (
                    <>
                      <button
                        className="w-upload"
                        disabled={busy}
                        onClick={() => fileRef.current?.click()}
                      >
                        <Upload size={30} />
                        <strong>
                          {busy
                            ? "Validating network…"
                            : "Choose a network file"}
                        </strong>
                        <span>
                          JSON graph, Quanta scenario JSON or GraphML · up to 12
                          MB
                        </span>
                      </button>
                      <input
                        hidden
                        ref={fileRef}
                        type="file"
                        accept=".json,.graphml"
                        onChange={(e) => upload(e.target.files?.[0])}
                      />
                      <p className="w-help">
                        GraphML roads use their supplied length and speed_mps;
                        absent speeds default to 10 m/s. Customer and fleet data
                        can be edited in the next step.
                      </p>
                      <Button
                        variant="ghost"
                        onClick={() =>
                          download(
                            "quanta-network-template.json",
                            JSON.stringify(generateNetwork(12, 5, 2), null, 2),
                          )
                        }
                      >
                        <Download size={15} />
                        Download JSON template
                      </Button>
                    </>
                  )}
                </div>
              </Panel>
              <Panel
                title="Network preview"
                action={
                  <Tag tone="blue">
                    {draft.source === "generated"
                      ? "Synthetic network"
                      : draft.networkId}
                  </Tag>
                }
              >
                <NetworkView graph={draft.graph} compact />
              </Panel>
            </>
          )}
          {step === 1 && <FleetEditor draft={draft} onChange={update} />}
          {step === 2 && (
            <>
              <Panel
                title="Traffic conditions"
                subtitle="A scenario-wide speed factor simulates congestion. It is not a live traffic feed."
              >
                <div className="w-form-body">
                  <div className="w-choice-grid">
                    {[
                      [1, "Normal flow", "100% of configured road speeds"],
                      [0.7, "Busy period", "70% of configured road speeds"],
                      [0.45, "Rush hour", "45% of configured road speeds"],
                    ].map(([factor, label, detail]) => (
                      <button
                        key={factor}
                        className={`w-choice ${draft.traffic === factor ? "selected" : ""}`}
                        onClick={() => update({ traffic: factor as number })}
                      >
                        <span className="w-radio" />
                        <strong>{label}</strong>
                        <small>{detail}</small>
                      </button>
                    ))}
                  </div>
                  <Notice tone="amber">
                    Closures below are active from the start of the run and are
                    enforced as hard constraints.
                  </Notice>
                </div>
              </Panel>
              <Panel
                title="Road incidents"
                subtitle="Select a road on the map or choose its directed edge below."
                action={
                  <Tag tone={draft.closures.length ? "amber" : "green"}>
                    {draft.closures.length} closures
                  </Tag>
                }
              >
                <NetworkView
                  graph={draft.graph}
                  closed={draft.closures}
                  compact
                  onCloseRoad={(edge) =>
                    update({
                      closures: draft.closures.includes(edge)
                        ? draft.closures.filter((x) => x !== edge)
                        : [...draft.closures, edge],
                    })
                  }
                />
                <div className="w-form-body">
                  <Field label="Add a directed road closure">
                    <select
                      value=""
                      onChange={(e) => {
                        if (
                          e.target.value &&
                          !draft.closures.includes(e.target.value)
                        )
                          update({
                            closures: [...draft.closures, e.target.value],
                          });
                      }}
                    >
                      <option value="">Select a road…</option>
                      {draft.graph.edges.map((e) => (
                        <option key={e.id} value={e.id}>
                          {e.id} · {e.from} → {e.to}
                        </option>
                      ))}
                    </select>
                  </Field>
                  <div className="w-chips">
                    {draft.closures.map((edge) => (
                      <button
                        key={edge}
                        onClick={() =>
                          update({
                            closures: draft.closures.filter((x) => x !== edge),
                          })
                        }
                      >
                        {edge}
                        <X size={13} />
                      </button>
                    ))}
                  </div>
                </div>
              </Panel>
            </>
          )}
          {step === 3 && (
            <>
              <Panel
                title="Optimisation algorithm"
                subtitle="Each algorithm uses the same independently validated routing constraints."
              >
                <div className="w-form-body w-algorithm-grid">
                  {methods.map((m) => (
                    <button
                      key={m.id}
                      disabled={
                        m.id === "milp" && draft.graph.requests.length > 12
                      }
                      className={`w-algorithm ${draft.method === m.id ? "selected" : ""}`}
                      onClick={() => update({ method: m.id })}
                    >
                      <span className="w-radio" />
                      <div>
                        <strong>
                          {m.label}
                          {m.id === "qpso" && <Tag tone="blue">Default</Tag>}
                        </strong>
                        <p>{m.detail}</p>
                      </div>
                    </button>
                  ))}
                </div>
              </Panel>
              <Panel
                title="Search budget"
                subtitle="The backend reports any large-network caps with the result."
              >
                <div className="w-form-body w-form-grid">
                  <Field
                    label="Population size"
                    hint="2–40 particles; applies to QPSO / PSO."
                  >
                    <input
                      type="number"
                      min={2}
                      max={40}
                      value={draft.particles}
                      onChange={(e) =>
                        update({
                          particles: Math.min(
                            40,
                            Math.max(2, Number(e.target.value)),
                          ),
                        })
                      }
                    />
                  </Field>
                  <Field
                    label="Evaluation budget"
                    hint="4–400 evaluations; at least the population size."
                  >
                    <input
                      type="number"
                      min={4}
                      max={400}
                      value={draft.evaluations}
                      onChange={(e) =>
                        update({
                          evaluations: Math.min(
                            400,
                            Math.max(4, Number(e.target.value)),
                          ),
                        })
                      }
                    />
                  </Field>
                  <Field
                    label="Random seed"
                    hint="Repeatable inputs for algorithm comparisons."
                  >
                    <input
                      type="number"
                      min={0}
                      value={draft.seed}
                      onChange={(e) => update({ seed: Number(e.target.value) })}
                    />
                  </Field>
                  <div className="w-budget-explainer">
                    <ShieldCheck size={22} />
                    <strong>Validation is always on</strong>
                    <p>
                      Vehicle capacity, customer service windows, directed
                      connectivity and depot return remain hard constraints.
                    </p>
                  </div>
                </div>
              </Panel>
              <Notice>
                Optimisation computes on the submitted scenario. Movement
                playback is a kinematic simulation of its validated routes.
              </Notice>
            </>
          )}
          <div className="w-builder-footer">
            <Button
              variant="secondary"
              disabled={step === 0}
              onClick={() => setStep(step - 1)}
            >
              <ArrowLeft size={16} />
              Back
            </Button>
            <span>Step {step + 1} of 4</span>
            {step < 3 ? (
              <Button onClick={() => setStep(step + 1)}>
                Continue <ArrowRight size={16} />
              </Button>
            ) : (
              <Button
                disabled={
                  busy || runner.job?.status === "running" || !store.canEdit
                }
                onClick={() => save(true)}
              >
                Save & run optimisation <ArrowRight size={16} />
              </Button>
            )}
          </div>
        </div>
        <aside className="w-builder-summary">
          <Panel title="Scenario summary">
            <div className="w-summary-body">
              <span className="w-summary-icon">
                <Waypoints size={26} />
              </span>
              <h3>{draft.name}</h3>
              <p>{draft.description}</p>
              {[
                [
                  "Network",
                  draft.source === "generated"
                    ? "Synthetic city"
                    : draft.networkId,
                ],
                ["Road nodes", draft.graph.nodes.length],
                ["Directed roads", draft.graph.edges.length],
                ["Vehicles", draft.graph.fleet.length],
                ["Deliveries", draft.graph.requests.length],
                [
                  "Demand",
                  `${draft.graph.requests.reduce((s, j) => s + j.demand, 0)} units`,
                ],
                ["Speed factor", `${Math.round(draft.traffic * 100)}%`],
                ["Closures", draft.closures.length],
                ["Algorithm", draft.method.toUpperCase()],
              ].map(([label, value]) => (
                <div className="w-summary-row" key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
              <div className="w-summary-note">
                <ShieldCheck size={17} />
                Changes are included in the next computation.
              </div>
            </div>
          </Panel>
        </aside>
      </div>
    </>
  );
}

export function FleetEditor({
  draft,
  onChange,
}: {
  draft: Draft;
  onChange: (patch: Partial<Draft>) => void;
}) {
  const [error, setError] = useState("");
  const uploadRef = useRef<HTMLInputElement>(null);
  const graph = draft.graph;
  const nodes = graph.nodes;
  function updateJob(index: number, patch: Partial<Delivery>) {
    onChange({
      graph: {
        ...graph,
        requests: graph.requests.map((j, i) =>
          i === index ? { ...j, ...patch } : j,
        ),
      },
    });
  }
  function updateVehicle(
    index: number,
    patch: Partial<Network["fleet"][number]>,
  ) {
    onChange({
      graph: {
        ...graph,
        fleet: graph.fleet.map((v, i) =>
          i === index ? { ...v, ...patch } : v,
        ),
      },
    });
  }
  async function importJobs(file?: File) {
    if (!file) return;
    try {
      if (file.size > 1024 * 1024)
        throw new Error("Use a CSV smaller than 1 MB.");
      const rows = parseCsv(await file.text());
      if (rows.length > 200)
        throw new Error("The demo supports up to 200 deliveries.");
      const requests = rows.map((r) => {
        for (const field of ["id", "node", "demand"])
          if (!r[field]) throw new Error(`Missing ${field} column or value.`);
        return {
          id: r.id,
          node: r.node,
          demand: Number(r.demand),
          earliest_s: Number(r.earliest_s || 0),
          latest_s: Number(r.latest_s || 14400),
          service_s: Number(r.service_s || 30),
        };
      });
      const next = { ...draft, graph: { ...graph, requests } };
      const invalid = validateDraft(next);
      if (invalid) throw new Error(invalid);
      onChange({ graph: next.graph });
      setError("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      if (uploadRef.current) uploadRef.current.value = "";
    }
  }
  return (
    <>
      {error && <Notice tone="red">{error}</Notice>}
      <Panel
        title="Vehicle fleet"
        subtitle="Capacity uses the same load units as delivery demand."
        action={
          <Button
            variant="secondary"
            disabled={graph.fleet.length >= 50}
            onClick={() =>
              onChange({
                graph: {
                  ...graph,
                  fleet: [
                    ...graph.fleet,
                    {
                      id: `V-${uid().slice(0, 4)}`,
                      capacity: 100,
                      depot: nodes[0].id,
                    },
                  ],
                },
              })
            }
          >
            <Plus size={15} />
            Add vehicle
          </Button>
        }
      >
        <div className="w-table-scroll">
          <table className="w-table w-edit-table">
            <thead>
              <tr>
                <th>Vehicle</th>
                <th>Capacity (units)</th>
                <th>Depot / start node</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {graph.fleet.map((v, i) => (
                <tr key={i}>
                  <td>
                    <div className="w-inline-input">
                      <Truck size={17} />
                      <input
                        aria-label={`Vehicle ${i + 1} ID`}
                        value={v.id}
                        onChange={(e) =>
                          updateVehicle(i, { id: e.target.value })
                        }
                      />
                    </div>
                  </td>
                  <td>
                    <input
                      aria-label={`${v.id} capacity`}
                      type="number"
                      min={1}
                      value={v.capacity}
                      onChange={(e) =>
                        updateVehicle(i, { capacity: Number(e.target.value) })
                      }
                    />
                  </td>
                  <td>
                    <select
                      aria-label={`${v.id} depot`}
                      value={v.depot}
                      onChange={(e) =>
                        updateVehicle(i, { depot: e.target.value })
                      }
                    >
                      {nodes.map((n) => (
                        <option key={n.id} value={n.id}>
                          {n.id}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td>
                    <button
                      className="w-icon-button"
                      aria-label={`Remove vehicle ${v.id}`}
                      disabled={graph.fleet.length <= 1}
                      onClick={() =>
                        onChange({
                          graph: {
                            ...graph,
                            fleet: graph.fleet.filter((_, n) => i !== n),
                          },
                        })
                      }
                    >
                      <Trash2 size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <Panel
        title="Customer deliveries"
        subtitle="Service windows and durations are in seconds from the scenario start."
        action={
          <div className="w-row-actions">
            <Button
              variant="ghost"
              onClick={() =>
                download(
                  "deliveries-template.csv",
                  `id,node,demand,earliest_s,latest_s,service_s\nD001,${nodes[1]?.id},10,0,14400,30\n`,
                  "text/csv",
                )
              }
            >
              <Download size={15} />
              Template
            </Button>
            <Button
              variant="secondary"
              onClick={() => uploadRef.current?.click()}
            >
              <FileUp size={15} />
              Import CSV
            </Button>
          </div>
        }
      >
        <input
          hidden
          ref={uploadRef}
          type="file"
          accept=".csv"
          onChange={(e) => importJobs(e.target.files?.[0])}
        />
        <div className="w-table-scroll">
          <table className="w-table w-edit-table w-delivery-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Road node</th>
                <th>Demand</th>
                <th>Window start</th>
                <th>Window end</th>
                <th>Service</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {graph.requests.map((j, i) => (
                <tr key={i}>
                  <td>
                    <input
                      aria-label={`Delivery ${i + 1} ID`}
                      value={j.id}
                      onChange={(e) => updateJob(i, { id: e.target.value })}
                    />
                  </td>
                  <td>
                    <select
                      aria-label={`${j.id} road node`}
                      value={j.node}
                      onChange={(e) => updateJob(i, { node: e.target.value })}
                    >
                      {nodes.map((n) => (
                        <option key={n.id}>{n.id}</option>
                      ))}
                    </select>
                  </td>
                  {(
                    ["demand", "earliest_s", "latest_s", "service_s"] as const
                  ).map((key) => (
                    <td key={key}>
                      <input
                        aria-label={`${j.id} ${key}`}
                        type="number"
                        min={0}
                        value={j[key]}
                        onChange={(e) =>
                          updateJob(i, { [key]: Number(e.target.value) })
                        }
                      />
                    </td>
                  ))}
                  <td>
                    <button
                      className="w-icon-button"
                      aria-label={`Remove delivery ${j.id}`}
                      disabled={graph.requests.length <= 1}
                      onClick={() =>
                        onChange({
                          graph: {
                            ...graph,
                            requests: graph.requests.filter((_, n) => n !== i),
                          },
                        })
                      }
                    >
                      <Trash2 size={15} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="w-panel-foot">
          <Button
            variant="ghost"
            disabled={graph.requests.length >= 200}
            onClick={() =>
              onChange({
                graph: {
                  ...graph,
                  requests: [
                    ...graph.requests,
                    {
                      id: `D-${uid().slice(0, 4)}`,
                      node: nodes[1]?.id || nodes[0].id,
                      demand: 10,
                      earliest_s: 0,
                      latest_s: 14400,
                      service_s: 30,
                    },
                  ],
                },
              })
            }
          >
            <Plus size={16} />
            Add delivery
          </Button>
          <span>
            {graph.requests.reduce((s, j) => s + j.demand, 0)} total demand
            units
          </span>
        </div>
      </Panel>
    </>
  );
}

export function FleetPage() {
  const store = useWorkspace();
  const [id, setId] = useState(store.drafts[0]?.id || "");
  const [draft, setDraft] = useState<Draft | null>(() =>
    clone(store.drafts.find((d) => d.id === id) || null),
  );
  const [message, setMessage] = useState(""),
    [error, setError] = useState("");
  useEffect(() => {
    setDraft(clone(store.drafts.find((d) => d.id === id) || null));
  }, [id]);
  return (
    <>
      <PageTitle
        eyebrow="FLEET PLANNING"
        title="Fleet & deliveries"
        description="Manage the vehicles and delivery commitments assigned to each scenario."
        action={
          <select
            aria-label="Fleet scenario"
            value={id}
            onChange={(e) => {
              setId(e.target.value);
              setMessage("");
            }}
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
        }
      />
      {error && <Notice tone="red">{error}</Notice>}
      {message && <Notice tone="green">{message}</Notice>}
      {draft ? (
        <>
          <FleetEditor
            draft={draft}
            onChange={(patch) => {
              setDraft({ ...draft, ...patch });
              setMessage("");
            }}
          />
          <div className="w-builder-footer">
            <span>
              Updates affect future runs. Existing result snapshots stay
              unchanged.
            </span>
            <Button
              disabled={!store.canEdit}
              onClick={async () => {
                const invalid = validateDraft(draft);
                if (invalid) {
                  setError(invalid);
                  return;
                }
                try {
                  await store.saveDraft({
                    ...draft,
                    updatedAt: new Date().toISOString(),
                  });
                  setMessage("Fleet and deliveries saved.");
                  setError("");
                } catch (e) {
                  setError((e as Error).message);
                }
              }}
            >
              <Save size={16} />
              Save changes
            </Button>
          </div>
        </>
      ) : (
        <Empty
          title="Choose a scenario to manage its fleet"
          action={
            <Link className="w-button primary" to="/app/scenarios/new">
              Create scenario
            </Link>
          }
        >
          Vehicle capacity and delivery demand are saved together with the road
          network.
        </Empty>
      )}
    </>
  );
}
