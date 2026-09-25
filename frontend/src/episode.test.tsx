/**
 * Episode → simulation-map end-to-end against the live backend (127.0.0.1:8765):
 *   /control-room (the NEW frontend's control room) boots the workspace
 *   Shell onto a recorded Delhi map (DELHI_CP) → the Operations simulation
 *   map offers all 10 Delhi recorded maps in its scenario picker → the
 *   episode feed auto-selects an episode and auto-plays fleet movement →
 *   1929 roads painted from real frame truth (observed speed bands vs gray
 *   no-data) → recorded fleet cars + background dots on the Leaflet map →
 *   legend switches to episode semantics → tooltip carries "% of free flow
 *   at t=" → scrub the timeline (frame refetches) → play advances the
 *   clock → the old control room UI is NOT what /control-room renders →
 *   no runtime errors.
 *
 * react-leaflet is mocked so real Leaflet map construction never runs in
 * jsdom; the mock exposes pathOptions/icon identity as data attributes so
 * assertions inspect exactly what GeoMap asked Leaflet to draw.
 */
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createRoot, type Root } from "react-dom/client";
import WorkspaceApp from "./workspace/WorkspaceApp";
import { ToastProvider } from "./components/ui/Toast";

// Mock react-leaflet primitives — GeoMap's usage is declarative, so we
// render each layer as a plain DOM node carrying the props we assert on.
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
const findButton = (text: string | RegExp) =>
  $$("button").find((b) => (typeof text === "string" ? (b.textContent || "").includes(text) : text.test(b.textContent || "")));

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

describe("episode feed on the control-room simulation map (live backend)", () => {
  const runtimeErrors: string[] = [];
  let root: Root | null = null;
  let container: HTMLDivElement | null = null;
  const origError = console.error;

  beforeAll(() => {
    console.error = (...args: unknown[]) => {
      const text = args.map(String).join(" ");
      if (!/not wrapped in act|ReactDOM.render|deprecated/i.test(text)) runtimeErrors.push(text);
      origError.apply(console, args);
    };
    window.addEventListener("unhandledrejection", (e) => runtimeErrors.push(`unhandled: ${String((e as PromiseRejectionEvent).reason)}`));
    window.addEventListener("error", (e) => runtimeErrors.push(`error: ${(e as ErrorEvent).message}`));
    sessionStorage.setItem("quanta.demo", "true");
    window.history.pushState({}, "", "/control-room");
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

  it("control room = workspace simulation map fed by recorded episodes", async () => {
    try {
      // ── Workspace Shell boots at /control-room ────────────────────────
      await waitFor(() => $$(".w-nav-section").length >= 3, 60000, "workspace sidebar");
      await waitFor(() => !!document.querySelector('select[aria-label="Operation scenario"]'), 60000, "operations scenario select");
      const breadcrumb = document.querySelector(".w-breadcrumb")?.textContent || "";
      expect(breadcrumb).toContain("Control room");

      // ── Old control-room UI is NOT what renders here ──────────────────
      expect(document.querySelector(".home-shell")).toBeNull();
      expect(document.querySelector(".episode-card")).toBeNull();
      expect(document.querySelector("a.home-link")).toBeNull();
      expect(document.querySelector("a.nav-tab-link")).toBeNull();
      expect(findButton("Open control room")).toBeUndefined();

      // ── Delhi recorded maps sit in the scenario picker ────────────────
      const scenario = document.querySelector('select[aria-label="Operation scenario"]') as HTMLSelectElement;
      await waitFor(
        () => Array.from(scenario.options).some((o) => o.value === "DELHI_CP"),
        30000,
        "Delhi maps listed in scenario picker",
      );
      const delhiOptions = Array.from(scenario.options).filter((o) => o.value.startsWith("DELHI_"));
      expect(delhiOptions.length).toBe(10);
      // Boot default for /control-room is a recorded Delhi map.
      expect(scenario.value).toBe("DELHI_CP");

      // ── Geo map renders (mocked react-leaflet, all 1929 dwarka roads) ─
      await waitFor(() => $$(".road-hit").length >= 1000, 120000, "Delhi road polylines");

      // ── Episode feed: 175 episodes, first auto-selected, auto-playing ─
      await waitFor(() => !!document.querySelector('select[aria-label="Recorded episode"]'), 60000, "recorded episode select");
      const episodeSel = document.querySelector('select[aria-label="Recorded episode"]') as HTMLSelectElement;
      expect(episodeSel.options.length).toBe(175);
      // Auto-play started (button reads "Pause episode") — fleet movement
      // is on screen without any click. Stop it for deterministic asserts.
      await waitFor(() => !!findButton("Pause episode"), 60000, "episode auto-play started");
      click(findButton("Pause episode"));
      await waitFor(() => !!findButton("Play episode"), 10000, "episode paused");

      const timeText = () => document.querySelector(".w-episode")?.textContent || "";
      await waitFor(() => /t=\d+s \/ \d+s/.test(timeText()), 60000, "episode clock rendered");
      // Frame stats from real recorded truth arrive for the default playhead.
      await waitFor(() => /\/1929 roads/.test(document.body.textContent || ""), 60000, "frame stats loaded");

      // ── Roads painted from recorded truth ────────────────────────────
      const roads = $$(".road-hit");
      expect(roads.length).toBeGreaterThanOrEqual(1900);
      const band = new Set(["#22c55e", "#a3e635", "#f59e0b", "#ef4444"]);
      const observed = roads.filter((r) => band.has(r.getAttribute("data-color") || "")).length;
      const missing = roads.filter((r) => r.getAttribute("data-color") === "#64748b").length;
      expect(observed).toBeGreaterThanOrEqual(100); // real observed speeds
      expect(missing).toBeGreaterThanOrEqual(1000); // sparse-by-design gaps

      // ── Recorded vehicles: fleet cars (blue) + background dots ───────
      await waitFor(() => $$('[data-marker="fleet-car-icon"]').length > 0, 30000, "recorded fleet cars");
      expect($$('[data-marker="fleet-car-icon"]').length).toBeGreaterThanOrEqual(1);
      await waitFor(() => $$('[data-circle="ep-dot"]').length > 0, 30000, "background traffic dots");

      // ── Legend switched to episode semantics ─────────────────────────
      await waitFor(() => (document.body.textContent || "").includes("Recorded ≥80% free flow"), 30000, "episode legend");
      const legend = document.querySelector(".map-legend")?.textContent || "";
      expect(legend).toContain("No observation at t");
      expect(legend).toContain("Recorded background traffic");
      expect(legend).toMatch(/episode t=\d+s/);

      // ── Tooltips carry observed % of free flow at t ──────────────────
      expect(document.body.textContent).toMatch(/% of free flow at t=\d+s/);
      expect(document.body.textContent).toMatch(/no observation at t=\d+s/);

      // ── Scrub to t=900 → frame refetches (map re-keys on new t) ──────
      const range = document.querySelector('input[aria-label="Episode timeline"]') as HTMLInputElement;
      expect(range).toBeTruthy();
      setRange(range, "900");
      await waitFor(() => /t=900s \/ \d+s/.test(timeText()), 10000, "scrubbed to t=900");
      await waitFor(() => (document.body.textContent || "").includes("% of free flow at t=900s"), 60000, "frame at t=900 applied");
      expect(document.querySelector(".map-legend")?.textContent || "").toContain("episode t=900s");
      // Scrubbing pauses playback.
      expect(findButton("Play episode")).toBeTruthy();

      // ── Scrub to t=1800 → another real frame ─────────────────────────
      setRange(range, "1800");
      await waitFor(() => (document.body.textContent || "").includes("% of free flow at t=1800s"), 60000, "frame at t=1800 applied");

      // ── Play advances the recorded clock; pause stops it ─────────────
      click(findButton("Play episode"));
      await waitFor(() => /t=1860s \/ \d+s/.test(timeText()), 20000, "playback advanced to 1860");
      click(findButton("Pause episode"));
      await waitFor(() => !!findButton("Play episode"), 10000, "playback paused");

      // ── No runtime errors ────────────────────────────────────────────
      expect(runtimeErrors).toEqual([]);
    } catch (err) {
      // Dump state so failures are diagnosable from the log alone.
      const colors = $$(".road-hit").map((r) => r.getAttribute("data-color") || "");
      console.log("DIAG roads:", colors.length, "| observed:", colors.filter((c) => ["#22c55e", "#a3e635", "#f59e0b", "#ef4444"].includes(c)).length, "| gray:", colors.filter((c) => c === "#64748b").length);
      console.log("DIAG pathname:", window.location.pathname, "| episode-select:", $$('select[aria-label="Recorded episode"]').length, "| w-episode:", document.querySelector(".w-episode")?.textContent);
      console.log("DIAG fleet:", $$('[data-marker="fleet-car-icon"]').length, "| dots:", $$('[data-circle="ep-dot"]').length);
      console.log("DIAG legend:", document.querySelector(".map-legend")?.textContent);
      console.log("DIAG runtime-errors:", JSON.stringify(runtimeErrors.slice(0, 6)));
      throw err;
    }
  }, 300000);
});
