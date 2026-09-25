/**
 * Entry integration (live backend): the PR #25 workspace remains the default
 * entry, and /control-room opens the NEW frontend's control room — the
 * workspace Shell landing on the Operations simulation map (Leaflet fleet
 * map), with the recorded Delhi episode feed attached to it. The old
 * standalone control room UI (home/hero/topbar links) is not what renders.
 *
 * Boots WorkspaceApp in demo mode (sessionStorage) at /app, clicks the
 * sidebar "Control room" entry, and asserts the router swapped in the
 * simulation map with its episode feed.
 */
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { createRoot, type Root } from "react-dom/client";
import WorkspaceApp from "./workspace/WorkspaceApp";
import { ToastProvider } from "./components/ui/Toast";

// Mock react-leaflet — the control-room map is a real Leaflet/GeoMap graph
// (DELHI_CP), which must not construct a real map inside jsdom.
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

describe("control-room route (workspace entry)", () => {
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

  it("sidebar entry opens the control room = workspace simulation map", async () => {
    try {
      // ── Demo workspace shell renders with the sidebar navigation ─────
      await waitFor(() => $$(".w-nav-section").length >= 3, 60000, "workspace sidebar sections");
      const controlLink = $$(".w-sidebar nav a").find((a) => (a.textContent || "").includes("Control room"));
      expect(controlLink).toBeTruthy();

      // ── Click it → URL /control-room, workspace Shell stays mounted ──
      click(controlLink);
      await waitFor(() => window.location.pathname === "/control-room", 20000, "URL is /control-room");
      await waitFor(
        () => !!document.querySelector('select[aria-label="Operation scenario"]'),
        60000,
        "operations scenario select (simulation map page)",
      );
      const breadcrumb = document.querySelector(".w-breadcrumb")?.textContent || "";
      expect(breadcrumb).toContain("Control room");

      // ── The simulation map itself: Fleet operations map panel ─────────
      await waitFor(
        () => (document.body.textContent || "").includes("Fleet operations map"),
        30000,
        "fleet operations map panel",
      );

      // ── Recorded episodes are pushed onto that map ────────────────────
      await waitFor(
        () => !!document.querySelector('select[aria-label="Recorded episode"]'),
        90000,
        "recorded episode feed on the simulation map",
      );

      // ── The old standalone control room UI is not what renders ────────
      expect(document.querySelector(".home-shell")).toBeNull();
      expect(document.querySelector("a.home-link")).toBeNull();
      expect(document.querySelector("a.nav-tab-link")).toBeNull();
      expect(findButton("Open control room")).toBeUndefined();
      expect(window.location.pathname).toBe("/control-room");
    } catch (err) {
      console.log("DIAG pathname:", window.location.pathname, "| sidebar-sections:", $$(".w-nav-section").length, "| nav-links:", $$(".w-sidebar nav a").length);
      console.log("DIAG scenario-select:", $$('select[aria-label="Operation scenario"]').length, "| episode-select:", $$('select[aria-label="Recorded episode"]').length, "| road-hit:", $$(".road-hit").length);
      console.log("DIAG buttons:", $$("button").map((b) => (b.textContent || "").slice(0, 30)).slice(0, 8));
      throw err;
    }
  }, 180000);
});
