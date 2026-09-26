/**
 * Forecasting section end-to-end against the live backend (127.0.0.1:8765):
 *   /app/forecasting boots the shell onto section 5 → the recorded-corpus zone
 *   picker lists every zone → the episode picker auto-selects an episode →
 *   issued forecasts (GET /api/episodes/{id}/forecasts) render the speed
 *   forecast chart with its P10–P90 band → switching horizon (5/10/15 min)
 *   re-reads the same recorded issue → the predicted-speed heatmap projects
 *   onto the network → the "How the AI model works" card carries corpus +
 *   artifact facts → Model Performance shows both the browser-recomputed
 *   episode MAE and the saved held-out test metrics → no runtime errors.
 *
 * react-leaflet is mocked so MapPanel never constructs real Leaflet in jsdom.
 */
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createRoot, type Root } from "react-dom/client";
import WorkspaceApp from "./workspace/WorkspaceApp";
import { ToastProvider } from "./components/ui/Toast";

vi.mock("react-leaflet", async () => {
  const { createElement: h, Fragment } = await import("react");
  const MapContainer = ({ children }: { children?: React.ReactNode }) => h("div", { "data-map-container": "1" }, children);
  const TileLayer = () => null;
  const LayersControl = ({ children }: { children?: React.ReactNode }) => h(Fragment, null, children);
  LayersControl.BaseLayer = ({ children }: { children?: React.ReactNode }) => h(Fragment, null, children);
  LayersControl.Overlay = ({ children }: { children?: React.ReactNode }) => h(Fragment, null, children);
  const Polyline = ({ children, className, pathOptions }: { children?: React.ReactNode; className?: string; pathOptions?: Record<string, unknown> }) =>
    h("div", { className, "data-color": pathOptions?.color, "data-weight": pathOptions?.weight }, children);
  const Marker = ({ children, icon }: { children?: React.ReactNode; icon?: { options?: { className?: string } } }) =>
    h("div", { "data-marker": icon?.options?.className ?? "" }, children);
  const CircleMarker = ({ children, radius, pathOptions }: { children?: React.ReactNode; radius?: number; pathOptions?: Record<string, unknown> }) =>
    h("div", { "data-circle": pathOptions?.className ?? "", "data-radius": radius }, children);
  const Tooltip = ({ children }: { children?: React.ReactNode }) => h("span", { "data-tip": "1" }, children);
  return { MapContainer, TileLayer, LayersControl, Polyline, Marker, CircleMarker, Tooltip };
});

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = false;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function waitFor(cond: () => boolean, timeoutMs: number, label: string) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (cond()) return;
    await sleep(100);
  }
  throw new Error(`timeout after ${timeoutMs}ms waiting for: ${label}`);
}

const $$ = (sel: string) => Array.from(document.querySelectorAll<HTMLElement>(sel));
const text = () => document.body.textContent || "";
const findButton = (value: string | RegExp) =>
  $$("button").find((b) => (typeof value === "string" ? (b.textContent || "").includes(value) : value.test(b.textContent || "")));

function click(el: Element | null | undefined) {
  if (!el) throw new Error("click target missing");
  el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
}

function setRange(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
  setter.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

describe("forecasting section — recorded episode forecasts (live backend)", () => {
  const runtimeErrors: string[] = [];
  let root: Root | null = null;
  let container: HTMLDivElement | null = null;
  const origError = console.error;

  beforeAll(() => {
    console.error = (...args: unknown[]) => {
      const value = args.map(String).join(" ");
      if (!/not wrapped in act|ReactDOM.render|deprecated/i.test(value)) runtimeErrors.push(value);
      origError.apply(console, args);
    };
    window.addEventListener("unhandledrejection", (e) => runtimeErrors.push(`unhandled: ${String((e as PromiseRejectionEvent).reason)}`));
    window.addEventListener("error", (e) => runtimeErrors.push(`error: ${(e as ErrorEvent).message}`));
    sessionStorage.setItem("quanta.demo", "true");
    window.history.pushState({}, "", "/app/forecasting");
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    root.render(
      <ToastProvider>
        <WorkspaceApp />
      </ToastProvider>,
    );
  });

  afterEach(() => {
    console.error = origError;
    root?.unmount();
    container?.remove();
  });

  it("zone → episode → forecast chart → heatmap → model card → metrics", async () => {
    try {
      // ── Section 5 boots with its numbered heading ─────────────────────
      await waitFor(() => text().includes("FORECASTING"), 60000, "section 5 heading");
      await waitFor(() => !!document.querySelector('select[aria-label="Zone"]'), 60000, "zone select");

      // ── Recorded zones from the corpus index ──────────────────────────
      const zone = document.querySelector('select[aria-label="Zone"]') as HTMLSelectElement;
      await waitFor(() => zone.options.length > 0, 60000, "zones listed");
      const zoneValues = Array.from(zone.options).map((o) => o.value);
      expect(zoneValues).toContain("connaught_place-map000");

      // ── Episode picker auto-selects an episode for that zone ──────────
      await waitFor(
        () => (document.querySelector('select[aria-label="Episode"]') as HTMLSelectElement)?.options.length > 0,
        60000,
        "episodes listed",
      );
      const episode = document.querySelector('select[aria-label="Episode"]') as HTMLSelectElement;
      expect(episode.value).toBeTruthy();

      // ── Recorded issues render the forecast chart ─────────────────────
      await waitFor(() => text().includes("Speed forecast"), 60000, "forecast panel");
      await waitFor(() => !!document.querySelector('input[aria-label="Forecast issue time"]'), 90000, "issue time slider");
      const issueSlider = document.querySelector('input[aria-label="Forecast issue time"]') as HTMLInputElement;
      expect(Number(issueSlider.max)).toBeGreaterThan(0);

      // History + forecast polylines drawn from real recorded values.
      await waitFor(() => !!document.querySelector(".q-hist-line"), 60000, "historical line");
      await waitFor(() => !!document.querySelector(".q-fc-line"), 60000, "forecast line");
      await waitFor(() => !!document.querySelector(".q-band"), 60000, "P10–P90 band");

      // ── The recorded issue model is labelled honestly ─────────────────
      await waitFor(() => /persistence-v0/.test(text()), 60000, "forecast version label");
      expect(text()).toContain("persistence baseline");

      // ── Horizon radios re-read the same issue at +10 min ──────────────
      const horizonRadios = $$('[role="radio"]');
      expect(horizonRadios.length).toBe(3);
      const badgeBefore = document.querySelector(".q-badge")?.textContent || "";
      click(horizonRadios[1]);
      await waitFor(() => (document.querySelector(".q-badge")?.textContent || "").includes("+10 min"), 20000, "horizon → 10 min");
      expect(badgeBefore).toContain("+5 min");
      click(horizonRadios[2]);
      await waitFor(() => (document.querySelector(".q-badge")?.textContent || "").includes("+15 min"), 20000, "horizon → 15 min");

      // ── Scrub the issue time → a different recorded issue is charted ──
      setRange(issueSlider, "0");
      await waitFor(() => Number(issueSlider.value) === 0, 10000, "issue scrubbed to first");
      await waitFor(() => !!document.querySelector(".q-fc-line"), 20000, "forecast redrawn");

      // ── Predicted-speed heatmap projected onto the network ────────────
      await waitFor(() => text().includes("Traffic congestion map"), 60000, "heatmap panel");
      await waitFor(() => $$("[data-map-container]").length >= 1, 60000, "network map");
      await waitFor(() => !!document.querySelector(".q-lg-scale"), 60000, "heatmap scale legend");

      // ── Model explainer card carries real corpus + artifact facts ─────
      await waitFor(() => text().includes("How the AI model works"), 60000, "model explainer card");
      await waitFor(() => /episodes \(/.test(text()), 60000, "corpus episode counts");
      expect(text()).toContain("GNN — reads the map");
      expect(text()).toContain("Transformer — reads the recent past");
      expect(text()).toContain("Map-disjoint split");

      // ── Model Performance: recomputed + saved held-out metrics ────────
      await waitFor(() => text().includes("Model Performance"), 60000, "performance section");
      expect(text()).toContain("Selected episode · live recomputation");
      await waitFor(() => text().includes("Corpus test split · saved artifact"), 60000, "saved artifact metrics");
      const perfRows = $$(".q-perf-grid .q-table tbody tr");
      expect(perfRows.length).toBe(6); // 3 horizons × 2 tables
      expect(text()).toContain("speed ratio");
      expect(text()).toContain("GET /api/evidence");

      // ── Scanning another zone swaps the episode list ──────────────────
      const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")!.set!;
      setter.call(zone, "okhla-map007");
      zone.dispatchEvent(new Event("change", { bubbles: true }));
      await waitFor(
        () => (document.querySelector('select[aria-label="Episode"]') as HTMLSelectElement)?.value.startsWith("ep-"),
        60000,
        "episodes reloaded for the new zone",
      );

      // ── No runtime errors ─────────────────────────────────────────────
      expect(runtimeErrors).toEqual([]);
    } catch (err) {
      console.log("DIAG pathname:", window.location.pathname);
      console.log("DIAG zones:", $$('select[aria-label="Zone"] option').map((o) => (o as HTMLOptionElement).value));
      console.log("DIAG episodes:", ($$('select[aria-label="Episode"]')[0] as unknown as HTMLSelectElement | undefined)?.options.length);
      console.log("DIAG chart lines:", $$(".q-hist-line").length, $$(".q-fc-line").length, "| band:", $$(".q-band").length);
      console.log("DIAG tables:", $$(".q-table").length, "| perf rows:", $$(".q-perf-grid .q-table tbody tr").length);
      console.log("DIAG badges:", $$(".q-badge").map((b) => b.textContent));
      console.log("DIAG runtime-errors:", JSON.stringify(runtimeErrors.slice(0, 6)));
      throw err;
    }
  }, 300000);
});
