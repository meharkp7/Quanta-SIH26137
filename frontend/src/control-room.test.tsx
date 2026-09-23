/**
 * End-to-end control-room flow against the live backend (127.0.0.1:8765):
 *   home → control room → load replay → fleet cars render (blue, rotated)
 *   → click a car → pick destination → per-vehicle re-route (magenta path)
 *   → click a road (E12) → auto re-plan fires → "Fleet re-routed"
 *   → unblock → fleet re-plans back, with no console/runtime errors.
 *
 * jsdom dispatches real bubbling MouseEvents, so React's delegated handlers
 * on the SVG <g> car and hit-lines exercise the same code a browser would.
 */
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { createRoot, type Root } from "react-dom/client";
import App from "./App";
import { ToastProvider } from "./components/ui/Toast";

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
// Historical union of every toast title ever observed (captured on poll).
const seenToasts = new Set<string>();
const toastTitles = () => {
  const titles = $$(".toast-title").map((e) => e.textContent || "");
  titles.forEach((t) => seenToasts.add(t));
  return titles;
};
const toastDescriptions = () => {
  const descs = $$(".toast-message").map((e) => e.textContent || "");
  descs.forEach((d) => seenToasts.add(`desc: ${d}`));
  return descs;
};
const findButton = (text: string | RegExp) =>
  $$("button").find((b) => (typeof text === "string" ? (b.textContent || "").includes(text) : text.test(b.textContent || "")));

function click(el: Element | null | undefined) {
  if (!el) throw new Error("click target missing");
  el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
}

function setSelect(sel: HTMLSelectElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")!.set!;
  setter.call(sel, value);
  sel.dispatchEvent(new Event("change", { bubbles: true }));
}

describe("control room e2e (live backend)", () => {
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
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    root.render(
      <ToastProvider>
        <App />
      </ToastProvider>,
    );
  });

  afterEach(() => {
    console.error = origError;
    root?.unmount();
    container?.remove();
  });

  it("fleet cars → per-vehicle re-route → auto re-plan on road closure", async () => {
    try {
    // ── Home → control room ────────────────────────────────────────────
    await waitFor(() => !!findButton("Open control room"), 30000, "home page");
    click(findButton("Open control room"));

    await waitFor(() => $$(".map-edge").length >= 10, 60000, "network map edges");
    await waitFor(() => $$("select").length >= 3, 30000, "scenario sidebar");

    // ── Load SUMO replay → fleet renders as blue cars ──────────────────
    click(findButton(/Load SUMO replay/));
    await waitFor(() => $$(".fleet-car-svg").length > 0, 120000, "fleet cars rendered");
    expect(document.body.textContent).toContain("Fleet car · click = reroute");
    const carCount = $$(".fleet-car-svg").length;
    expect(carCount).toBeGreaterThan(0);

    // ── Click a fleet car → selection panel + toast ────────────────────
    click(document.querySelector('.fleet-car-svg[data-vehicle]'));
    await waitFor(() => !!document.querySelector(".veh-dispatch"), 8000, "vehicle dispatch panel");
    await waitFor(() => toastTitles().some((t) => /selected/.test(t)), 8000, "vehicle-selected toast");

    // ── Pick destination → re-route → magenta path + car animates ──────
    const dest = document.querySelector('select[aria-label="Vehicle destination"]') as HTMLSelectElement | null;
    expect(dest).toBeTruthy();
    setSelect(dest!, "N1");
    await waitFor(() => {
      const b = findButton("Re-route vehicle");
      return !!b && !(b as HTMLButtonElement).disabled;
    }, 8000, "re-route button enabled");
    click(findButton("Re-route vehicle"));
    await waitFor(() => !!document.querySelector(".map-vtrip-line"), 20000, "vehicle re-route path line");
    await waitFor(() => toastTitles().some((t) => /rerouted/.test(t)), 20000, "vehicle rerouted toast");
    await waitFor(() => !!document.querySelector(".legend-line.vtrip"), 8000, "re-route legend entry");

    // ── Click road E12 as the incident → auto re-plan (no Solve press) ──
    click(document.querySelector('.map-hit[data-edge="E12"]'));
    await waitFor(() => $$(".closure-chip").length === 1, 8000, "closure chip for E12");
    await waitFor(() => !!document.querySelector(".auto-plan-chip") || toastTitles().includes("Fleet re-routed"), 30000, "auto re-plan started");
    await waitFor(
      () => toastTitles().includes("Fleet re-routed") && toastDescriptions().some((d) => d.startsWith("1 closure")),
      120000,
      '"Fleet re-routed" toast (1 closure)',
    );
    await waitFor(() => !document.querySelector(".auto-plan-chip"), 30000, "auto re-plan chip cleared");

    // ── Unblock → fleet re-plans back automatically ────────────────────
    click(document.querySelector(".closure-chip button"));
    await waitFor(() => $$(".closure-chip").length === 0, 8000, "closure chips cleared");
    await waitFor(
      () => toastTitles().includes("Fleet re-routed") && toastDescriptions().some((d) => d.startsWith("0 closures")),
      120000,
      '"Fleet re-routed" toast (0 closures, reopen)',
    );

    // ── No runtime errors ──────────────────────────────────────────────
    expect(runtimeErrors).toEqual([]);
    } catch (err) {
      // Dump state so failures are diagnosable from the log alone.
      console.log("DIAG toasts-now:", JSON.stringify($$(".toast-item").map((t) => `${t.querySelector(".toast-title")?.textContent}: ${t.querySelector(".toast-message")?.textContent}`)));
      console.log("DIAG toasts-seen:", JSON.stringify(Array.from(seenToasts)));
      console.log("DIAG auto-chip:", $$(".auto-plan-chip").length, "| closure-chips:", $$(".closure-chip").length, "| cars:", $$(".fleet-car-svg").length, "| vtrip:", $$(".map-vtrip-line").length);
      console.log("DIAG runtime-errors:", JSON.stringify(runtimeErrors.slice(0, 6)));
      throw err;
    }
  }, 300000);
});
