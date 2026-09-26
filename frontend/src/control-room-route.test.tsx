/**
 * Entry + navigation integration (live backend 127.0.0.1:8765): the redesign's
 * five numbered sections are what the router serves.
 *
 * Boots WorkspaceApp in demo mode at /app and asserts:
 *   1. the shell renders the image's chrome — top bar (QUANTA | Fleet Routing
 *      Intelligence + tagline), sidebar with sections 2–5, bottom step rail;
 *   2. section 2 DASHBOARD / LIVE OPERATIONS loads the real scenario catalog,
 *      the map, the optimization controls and the KPI row;
 *   3. sections 3, 4 and 5 are reachable from the sidebar and each renders its
 *      own numbered heading and controls;
 *   4. the removed legacy workspace navigation (w-sidebar / w-nav-section) and
 *      the old standalone control room UI are NOT what renders.
 *
 * react-leaflet is mocked — MapPanel must not construct real Leaflet in jsdom.
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

describe("quanta sections (workspace entry, live backend)", () => {
  let root: Root | null = null;
  let container: HTMLDivElement | null = null;

  beforeAll(() => {
    sessionStorage.setItem("quanta.demo", "true");
    window.history.pushState({}, "", "/app");
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
    root?.unmount();
    container?.remove();
  });

  it("shell chrome + all five numbered sections are live", async () => {
    try {
      // ── Image chrome: top bar, tagline, sidebar sections 2–5, step rail ─
      await waitFor(() => !!document.querySelector(".q-shell"), 60000, "quanta shell");
      expect(document.querySelector(".q-brand-word")?.textContent).toBe("QUANTA");
      expect(document.querySelector(".q-brand-sub")?.textContent).toContain("Fleet Routing Intelligence");
      expect(document.querySelector(".q-tagline")?.textContent).toContain("Smarter Routes. Greener Cities.");

      const navLinks = () => $$(".q-nav a");
      await waitFor(() => navLinks().length === 4, 30000, "four sidebar sections");
      const labels = navLinks().map((a) => (a.textContent || "").trim());
      expect(labels.join("|")).toContain("Dashboard");
      expect(labels.join("|")).toContain("What-If Scenarios");
      expect(labels.join("|")).toContain("Route Lab");
      expect(labels.join("|")).toContain("Forecasting");

      const rail = $$(".q-rail-step");
      expect(rail.length).toBe(6);
      expect((document.querySelector(".q-rail-steps")?.textContent || "")).toContain("Configure");

      // ── Section 2: DASHBOARD / LIVE OPERATIONS ─────────────────────────
      await waitFor(() => text().includes("DASHBOARD / LIVE OPERATIONS"), 60000, "section 2 heading");
      await waitFor(() => !!document.querySelector('select[aria-label="Scenario"]'), 60000, "scenario select");
      const scenario = document.querySelector('select[aria-label="Scenario"]') as HTMLSelectElement;
      expect(Array.from(scenario.options).some((o) => o.value === "DELHI_CNP")).toBe(true);
      await waitFor(() => text().includes("Optimization Controls"), 30000, "optimization controls");
      await waitFor(() => text().includes("Vehicles deployed"), 30000, "KPI row");
      expect(!!findButton("Solve Routes")).toBe(true);
      expect(!!findButton("Download PDF Action Plan")).toBe(true);
      // Map panel renders (mocked Leaflet container for the geo graph).
      await waitFor(() => $$("[data-map-container]").length >= 1, 60000, "city map");

      // ── Section 3: WHAT-IF SCENARIOS ───────────────────────────────────
      click(navLinks()[1]);
      await waitFor(() => text().includes("WHAT-IF SCENARIOS"), 30000, "section 3 heading");
      await waitFor(() => text().includes("Disruption Controls"), 30000, "disruption controls");
      expect(!!findButton("Apply Scenario")).toBe(true);
      expect(!!findButton("Replan with Scenario")).toBe(true);
      expect(!!document.querySelector('input[aria-label="Traffic increase"]')).toBe(true);
      expect(!!document.querySelector('input[aria-label="Vehicle delay per stop"]')).toBe(true);

      // ── Section 4: ROUTE LAB ───────────────────────────────────────────
      click(navLinks()[2]);
      await waitFor(() => text().includes("ROUTE LAB"), 30000, "section 4 heading");
      await waitFor(() => text().includes("Shortest Path Finder"), 30000, "shortest path finder");
      expect(!!document.querySelector('select[aria-label="Source node"]')).toBe(true);
      expect(!!document.querySelector('select[aria-label="Destination node"]')).toBe(true);
      expect(!!document.querySelector('select[aria-label="Cost type"]')).toBe(true);
      expect(!!findButton("Find Path")).toBe(true);
      expect(text()).toContain("Algorithm Comparison");

      // ── Section 5: FORECASTING ─────────────────────────────────────────
      click(navLinks()[3]);
      await waitFor(() => text().includes("FORECASTING"), 60000, "section 5 heading");
      await waitFor(() => !!document.querySelector('select[aria-label="Zone"]'), 60000, "zone select");
      // The episode list is fetched after the zone lands — wait for its options,
      // not just for the select shell to mount.
      await waitFor(
        () =>
          ((document.querySelector('select[aria-label="Episode"]') as HTMLSelectElement | null)?.options.length || 0) > 0,
        60000,
        "episode select with episodes",
      );
      await waitFor(() => text().includes("Model Performance"), 60000, "model performance table");
      await waitFor(
        () => !!document.querySelector('input[aria-label="Forecast issue time"]'),
        90000,
        "forecast issue time slider",
      );

      // ── Removed legacy UI is NOT what renders ──────────────────────────
      expect(document.querySelector(".w-sidebar")).toBeNull();
      expect(document.querySelector(".w-nav-section")).toBeNull();
      expect(document.querySelector(".home-shell")).toBeNull();
      expect(document.querySelector(".episode-card")).toBeNull();
      expect(document.querySelector("a.nav-tab-link")).toBeNull();
      expect(findButton("Open control room")).toBeUndefined();
    } catch (err) {
      console.log("DIAG pathname:", window.location.pathname, "| nav:", $$(".q-nav a").length, "| rail:", $$(".q-rail-step").length);
      console.log("DIAG headings:", $$(".q-section-head h1").map((h) => h.textContent));
      console.log("DIAG selects:", $$("select").map((s) => s.getAttribute("aria-label")));
      console.log("DIAG buttons:", $$("button").map((b) => (b.textContent || "").trim()).filter(Boolean).slice(0, 12));
      console.log("DIAG map containers:", $$("[data-map-container]").length, "| w-sidebar:", $$(".w-sidebar").length);
      throw err;
    }
  }, 240000);
});
