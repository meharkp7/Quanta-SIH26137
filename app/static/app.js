const state = {
  scenarioId: "S3_BASE",
  graph: null,
  closed: "",
  lastSolve: null,
  replay: { frames: [], index: 0, timer: null },
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}, timeoutMs = 20000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...options,
    });
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`No response from ${path} after ${timeoutMs / 1000}s — is the backend running?`);
    }
    throw new Error(`Could not reach ${path} — ${error.message}`);
  } finally {
    clearTimeout(timer);
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || response.statusText);
  }
  return data;
}

function setStatus(text) {
  $("run-status").textContent = text;
}

function setBusy(button, isBusy) {
  if (!button) return;
  button.classList.toggle("is-busy", isBusy);
  button.disabled = isBusy;
}

async function withBusy(button, task) {
  setBusy(button, true);
  try {
    return await task();
  } finally {
    setBusy(button, false);
  }
}

function closedIds() {
  return state.closed ? [state.closed] : [];
}

function solveBody(extra = {}) {
  return {
    scenario_id: state.scenarioId,
    method: $("method").value,
    particles: Number($("particles").value),
    evaluations: Number($("evaluations").value),
    seed: 7,
    closed_edge_ids: closedIds(),
    ...extra,
  };
}

function projectPoint(node, graph) {
  const xs = graph.nodes.map((n) => n.x);
  const ys = graph.nodes.map((n) => n.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = 48;
  const width = 800 - pad * 2;
  const height = 420 - pad * 2;
  const x = pad + ((node.x - minX) / Math.max(1, maxX - minX)) * width;
  const y = pad + ((maxY - node.y) / Math.max(1, maxY - minY)) * height;
  return { x, y };
}

function drawGraph(graph, routes = [], pathEdges = [], movers = [], closedIds = []) {
  const svg = $("network");
  svg.innerHTML = "";
  const byId = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
  const routeSet = new Map();
  routes.forEach((route, index) => {
    (route.edge_ids || []).forEach((id) => routeSet.set(id, index));
  });
  const pathSet = new Set(pathEdges);
  const closedSet = new Set(closedIds);

  graph.edges.forEach((edge) => {
    const a = projectPoint(byId[edge.from], graph);
    const b = projectPoint(byId[edge.to], graph);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y);
    const blocked = !edge.open || closedSet.has(edge.id);
    line.setAttribute("class", `road${blocked ? " closed" : ""}`);
    if (routeSet.has(edge.id) && !blocked) {
      line.setAttribute("class", "route");
      line.setAttribute("stroke", routeSet.get(edge.id) === 0 ? "#5c9a6f" : "#4c86bd");
    }
    if (pathSet.has(edge.id)) {
      line.setAttribute("stroke", "#b6873f");
      line.setAttribute("stroke-width", "6");
    }
    svg.appendChild(line);
  });

  graph.nodes.forEach((node) => {
    const p = projectPoint(node, graph);
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", p.x);
    circle.setAttribute("cy", p.y);
    circle.setAttribute("r", node.kind.includes("depot") ? 8 : 6);
    circle.setAttribute("class", "node");
    circle.setAttribute("fill", node.kind.includes("depot") ? "#b6873f" : "#e6e8eb");
    svg.appendChild(circle);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", p.x + 8);
    label.setAttribute("y", p.y - 8);
    label.setAttribute("class", "label");
    label.textContent = node.id;
    svg.appendChild(label);
  });

  graph.requests.forEach((job) => {
    const node = byId[job.node];
    if (!node) return;
    const p = projectPoint(node, graph);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", p.x + 8);
    label.setAttribute("y", p.y + 14);
    label.setAttribute("class", "label");
    label.setAttribute("fill", "#5ee1a8");
    label.textContent = job.id;
    svg.appendChild(label);
  });

  movers.forEach((mover) => {
    const p = projectPoint({ x: mover.x, y: mover.y, kind: "" }, graph);
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    const delivery = mover.kind === "delivery";
    dot.setAttribute("cx", p.x);
    dot.setAttribute("cy", p.y);
    dot.setAttribute("r", delivery ? 8 : 4);
    dot.setAttribute("class", `mover ${delivery ? (mover.id === "V2" ? "v2" : "v1") : "bg"}`);
    if (mover.stopped) dot.setAttribute("stroke", "#f5c16c");
    svg.appendChild(dot);
    if (delivery) {
      const name = document.createElementNS("http://www.w3.org/2000/svg", "text");
      name.setAttribute("x", p.x + 10);
      name.setAttribute("y", p.y + 4);
      name.setAttribute("class", "label mover-label");
      name.textContent = mover.stopped ? `${mover.id} stop` : mover.id;
      svg.appendChild(name);
    }
  });
}

function drawTrace(trace) {
  const canvas = $("trace");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!trace || !trace.best || trace.best.length < 2) {
    ctx.fillStyle = "#8d96a1";
    ctx.fillText("Solve with QPSO or PSO to see the search curve.", 16, 24);
    return;
  }
  const values = trace.best;
  const min = Math.min(...values);
  const max = Math.max(...values);
  ctx.strokeStyle = "#2a3038";
  ctx.beginPath();
  ctx.moveTo(20, 10);
  ctx.lineTo(20, 160);
  ctx.lineTo(620, 160);
  ctx.stroke();
  ctx.strokeStyle = "#4c86bd";
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = 20 + (index / (values.length - 1)) * 600;
    const y = 160 - ((value - min) / Math.max(1e-6, max - min)) * 140;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function chips(evaluation) {
  const items = [
    ["Feasible", evaluation.feasible],
    ["Capacity", evaluation.capacity],
    ["Windows", evaluation.windows],
    ["Roads", evaluation.connectivity],
    ["Depot", evaluation.depot !== false],
    ["All served", evaluation.all_served],
  ];
  $("constraint-chips").innerHTML = items
    .map(([name, ok]) => `<span class="chip ${ok ? "ok" : "bad"}">${name}</span>`)
    .join("");
}

function showSolve(result) {
  state.lastSolve = result;
  const ev = result.evaluation || {};
  $("sc-method").textContent = result.method || "—";
  $("sc-time").textContent = ev.time_s != null ? `${ev.time_s.toFixed(1)} s` : "—";
  $("sc-dist").textContent = ev.distance_m != null ? `${ev.distance_m.toFixed(0)} m` : "—";
  $("sc-cong").textContent = ev.congestion_s != null ? `${ev.congestion_s.toFixed(1)} s` : "—";
  $("sc-elapsed").textContent = result.elapsed_s != null ? `${result.elapsed_s.toFixed(2)} s` : "—";
  chips(ev);
  const rows = (ev.vehicles || [])
    .map((veh) => {
      const stops = (veh.stops || [])
        .map((stop) => `${stop.job} @ ${stop.start_s.toFixed(0)}s`)
        .join(" → ");
      return `<tr><td>${veh.id}</td><td>${(veh.order || []).join(" → ")}</td><td>${veh.load}/${veh.capacity}</td><td>${veh.elapsed_s.toFixed(1)}</td><td>${stops}</td></tr>`;
    })
    .join("");
  $("route-table").innerHTML = `
    <table>
      <thead><tr><th>Vehicle</th><th>Order</th><th>Load</th><th>Elapsed s</th><th>Service</th></tr></thead>
      <tbody>${rows || "<tr><td colspan=5>No routes yet</td></tr>"}</tbody>
    </table>`;
  drawGraph(state.graph, ev.vehicles || []);
  drawTrace(result.trace);
}

async function loadGraph() {
  const overlay = $("graph-loading");
  overlay.hidden = false;
  const closed = closedIds().join(",");
  state.graph = await api(`/api/scenarios/${state.scenarioId}?closed=${encodeURIComponent(closed)}`).finally(
    () => { overlay.hidden = true; },
  );
  $("graph-caption").textContent = `${state.graph.scenario_id} — ${state.graph.nodes.length} nodes, ${state.graph.edges.length} directed roads`;
  $("closed-edge").innerHTML = `<option value="">None</option>` + state.graph.edges
    .map((edge) => `<option value="${edge.id}" ${edge.id === state.closed ? "selected" : ""}>${edge.id} ${edge.from}→${edge.to}</option>`)
    .join("");
  const nodeOptions = state.graph.nodes.map((n) => `<option value="${n.id}">${n.id}</option>`).join("");
  $("path-from").innerHTML = nodeOptions;
  $("path-to").innerHTML = nodeOptions;
  if (state.graph.nodes.length > 1) $("path-to").selectedIndex = 1;
  drawGraph(state.graph, state.lastSolve?.evaluation?.vehicles || []);
}

function revealApp() {
  $("app-shell").classList.add("ready");
  const loader = $("boot-loader");
  loader.classList.add("hidden");
  setTimeout(() => { loader.style.display = "none"; }, 300);
}

async function boot() {
  const meta = await api("/api/meta");
  $("ps-name").textContent = `${meta.ps_id} · ${meta.ps_name}`;
  const fill = (id, items) => {
    $(id).innerHTML = items.map((item) => `<li>${item}</li>`).join("");
  };
  fill("ps-wants", meta.what_sih_wants);
  fill("ps-build", meta.what_we_build);
  fill("ps-live", meta.what_is_live);
  fill("ps-base", meta.what_is_labelled_baseline);

  const catalog = await api("/api/scenarios");
  $("scenario").innerHTML = catalog.scenarios
    .filter((item) => item.available)
    .map((item) => `<option value="${item.id}">${item.label}</option>`)
    .join("");
  await loadGraph();
  const checked = await api("/api/validate", {
    method: "POST",
    body: JSON.stringify({ scenario_id: state.scenarioId, plan: state.graph.reference_plan }),
  });
  chips(checked.validation);
  revealApp();
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    $(`view-${button.dataset.tab}`).classList.add("active");
  });
});

$("scenario").addEventListener("change", async (event) => {
  state.scenarioId = event.target.value;
  state.lastSolve = null;
  await loadGraph();
});

$("closed-edge").addEventListener("change", async (event) => {
  state.closed = event.target.value;
  await loadGraph();
});

$("btn-solve").addEventListener("click", async (event) => {
  setStatus("Solving…");
  try {
    await withBusy(event.currentTarget, async () => {
      const result = await api("/api/solve", { method: "POST", body: JSON.stringify(solveBody()) });
      showSolve(result);
      $("loop-notes").innerHTML = `<li>${result.method} finished with status ${result.status}.</li>`;
    });
    setStatus("Solved");
  } catch (error) {
    setStatus(error.message);
  }
});

$("btn-loop").addEventListener("click", async (event) => {
  setStatus("Running loop…");
  try {
    await withBusy(event.currentTarget, async () => {
      const result = await api("/api/loop", { method: "POST", body: JSON.stringify(solveBody()) });
      showSolve(result.solve);
      $("loop-notes").innerHTML = result.notes.map((note) => `<li>${note}</li>`).join("");
      setStatus(`Loop — ${result.scope_action}, ${result.forecast_mode}`);
    });
  } catch (error) {
    setStatus(error.message);
  }
});

$("btn-bad").addEventListener("click", async () => {
  const plan = { V1: ["J1", "J2", "J3", "J4", "J5"], V2: [] };
  const checked = await api("/api/validate", {
    method: "POST",
    body: JSON.stringify({ scenario_id: state.scenarioId, plan, closed_edge_ids: closedIds() }),
  });
  chips(checked.validation);
  $("loop-notes").innerHTML = "<li>Overloaded V1 is illegal. The checker rejected it without trusting any solver.</li>";
  setStatus("Illegal plan rejected");
});

function stopReplay() {
  if (state.replay.timer) {
    clearInterval(state.replay.timer);
    state.replay.timer = null;
  }
}

function showReplayFrame(index) {
  const frames = state.replay.frames;
  if (!frames.length || !state.graph) return;
  const frame = frames[Math.max(0, Math.min(index, frames.length - 1))];
  state.replay.index = index;
  $("sim-clock").textContent = `t = ${frame.t.toFixed(0)}s, ${frame.vehicles.length} vehicles`;
  $("sim-scrub").value = String(index);
  drawGraph(
    state.graph,
    state.lastSolve?.evaluation?.vehicles || [],
    [],
    frame.vehicles,
    frame.closed || [],
  );
}

function playLoadedReplay() {
  stopReplay();
  if (!state.replay.frames.length) return;
  state.replay.timer = setInterval(() => {
    const next = state.replay.index + 1;
    if (next >= state.replay.frames.length) {
      stopReplay();
      setStatus("SUMO replay finished");
      return;
    }
    showReplayFrame(next);
  }, 90);
}

async function playSumoInUi(event) {
  stopReplay();
  setStatus("Running SUMO…");
  $("sim-clock").textContent = "SUMO running…";
  try {
    const result = await withBusy(event?.currentTarget, () => api("/api/sumo/replay", { method: "POST" }));
    state.replay.frames = result.frames || [];
    $("sim-scrub").max = String(Math.max(0, state.replay.frames.length - 1));
    const episode = result.episode || {};
    $("loop-notes").innerHTML = `
      <li>${result.message}</li>
      <li>Delivered ${episode.delivered_count ?? "?"} / 5 · teleports ${episode.teleport_events ?? "?"}</li>
    `;
    showReplayFrame(0);
    playLoadedReplay();
    setStatus("Playing SUMO on the map");
  } catch (error) {
    $("sim-clock").textContent = "SUMO failed";
    setStatus(error.message);
  }
}

$("btn-replay").addEventListener("click", playSumoInUi);
$("btn-replay-side").addEventListener("click", playSumoInUi);
$("sim-scrub").addEventListener("input", (event) => {
  stopReplay();
  showReplayFrame(Number(event.target.value));
});

$("btn-sumo").addEventListener("click", async (event) => {
  setStatus("Launching SUMO-GUI…");
  try {
    const result = await withBusy(event.currentTarget, () => api("/api/sumo", { method: "POST", body: JSON.stringify({ gui: true }) }));
    $("loop-notes").innerHTML = `<li>${result.message || "SUMO-GUI started."}</li>`;
    setStatus(result.mode === "gui" ? `SUMO-GUI pid ${result.pid}` : "SUMO finished");
  } catch (error) {
    setStatus(error.message);
  }
});

$("btn-compare").addEventListener("click", async (event) => {
  setStatus("Comparing solvers…");
  const body = $("compare-table").querySelector("tbody");
  body.innerHTML = "<tr><td colspan=8>Running…</td></tr>";
  try {
    const result = await withBusy(event.currentTarget, () => api("/api/compare", {
      method: "POST",
      body: JSON.stringify(solveBody({ methods: ["constructive", "qpso", "pso", "alns", "milp"] })),
    }));
    body.innerHTML = result.rows.map((row) => `
      <tr>
        <td>${row.method || ""}</td>
        <td>${row.error ? "error" : row.feasible ? "yes" : "no"}</td>
        <td>${row.objective != null ? row.objective.toFixed(1) : "—"}</td>
        <td>${row.time_s != null ? row.time_s.toFixed(1) : "—"}</td>
        <td>${row.distance_m != null ? row.distance_m.toFixed(0) : "—"}</td>
        <td>${row.elapsed_s != null ? row.elapsed_s.toFixed(2) : "—"}</td>
        <td>${row.evaluations ?? "—"}</td>
        <td>${row.error || row.status || ""}</td>
      </tr>`).join("");
    setStatus("Comparison ready");
  } catch (error) {
    body.innerHTML = `<tr><td colspan=8>${error.message}</td></tr>`;
    setStatus(error.message);
  }
});

$("btn-path").addEventListener("click", async (event) => {
  try {
    const result = await withBusy(event.currentTarget, () => api("/api/path", {
      method: "POST",
      body: JSON.stringify({
        scenario_id: state.scenarioId,
        source: $("path-from").value,
        target: $("path-to").value,
        closed_edge_ids: closedIds(),
      }),
    }));
    $("path-result").textContent = JSON.stringify(result, null, 2);
    if (result.feasible) drawGraph(state.graph, [], result.edge_ids);
  } catch (error) {
    $("path-result").textContent = error.message;
  }
});

boot().catch((error) => {
  setStatus(error.message);
  $("boot-loader-text").textContent = `Could not reach the backend: ${error.message}`;
  $("boot-ring").style.display = "none";
});