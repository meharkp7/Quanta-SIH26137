import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import type { ScenarioSummary } from "@/api";

// ── Formatting helpers shared by every section ─────────────────────────────
export const fmtKm = (meters: number) => `${(meters / 1000).toFixed(1)} km`;
export const fmtHours = (seconds: number) => `${(seconds / 3600).toFixed(1)} hrs`;
export const fmtEta = (seconds: number) => {
  const total = Math.max(0, Math.round(seconds / 60));
  const h = Math.floor(total / 60);
  const m = total % 60;
  return h > 0 ? `${h}h ${m.toString().padStart(2, "0")}m` : `${m}m`;
};
export const fmtClock = (seconds: number) => `${Math.round(seconds)}s`;
export const fmtClockTime = () => {
  const now = new Date();
  const date = now.toLocaleDateString("en-GB", { weekday: "short", day: "2-digit", month: "short", year: "numeric" });
  const time = now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  return `${date} · ${time}`;
};

// Numbered section header — the "1. LOGIN" … "5. FORECASTING" titles.
export function SectionHead({ n, title, sub, right }: { n: number; title: string; sub: string; right?: ReactNode }) {
  return (
    <div className="q-section-head">
      <div className="q-section-titles">
        <h1>
          <span className="q-section-num">{n}.</span> {title}
        </h1>
        <p>{sub}</p>
      </div>
      {right}
    </div>
  );
}

// Card container used everywhere (image: bordered panels, holo edge in theme).
export function Box({ title, action, children, className = "", pad = true, style }: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  pad?: boolean;
  style?: CSSProperties;
}) {
  return (
    <section className={`q-box ${className}`} style={style}>
      {title && (
        <header className="q-box-head">
          <h3>{title}</h3>
          {action}
        </header>
      )}
      <div className={pad ? "q-box-body" : "q-box-body flush"}>{children}</div>
    </section>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="q-field">
      <span>{label}</span>
      {children}
      {hint && <em>{hint}</em>}
    </label>
  );
}

export function Select({ value, onChange, options, ariaLabel, className = "" }: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string; group?: string }[];
  ariaLabel?: string;
  className?: string;
}) {
  const groups = useMemo(() => {
    const list: { group: string; items: { value: string; label: string }[] }[] = [];
    for (const option of options) {
      const name = option.group || "";
      const last = list[list.length - 1];
      if (last && last.group === name) last.items.push(option);
      else list.push({ group: name, items: [option] });
    }
    return list;
  }, [options]);
  return (
    <select className={`q-select ${className}`} aria-label={ariaLabel} value={value} onChange={(e) => onChange(e.target.value)}>
      {groups.map((group, index) =>
        group.group ? (
          <optgroup key={`${group.group}-${index}`} label={group.group}>
            {group.items.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </optgroup>
        ) : (
          group.items.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)
        ),
      )}
    </select>
  );
}

// Range slider styled like the image's disruption controls.
export function Slider({ label, value, min, max, step = 1, unit, onChange, disabled }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit: string;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className={`q-slider ${disabled ? "disabled" : ""}`}>
      <div className="q-slider-head">
        <span>{label}</span>
        <strong>
          {value}
          {unit}
        </strong>
      </div>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export function Btn({ children, onClick, disabled, kind = "primary", type = "button", full, title }: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  kind?: "primary" | "ghost" | "quiet";
  type?: "button" | "submit";
  full?: boolean;
  title?: string;
}) {
  return (
    <button type={type} title={title} className={`q-btn ${kind} ${full ? "full" : ""}`} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

// KPI chip — icon tile + big number + label (image's stat row).
export function Stat({ icon, value, label, tone }: { icon: ReactNode; value: ReactNode; label: string; tone?: string }) {
  return (
    <div className="q-stat">
      <span className={`q-stat-icon ${tone || ""}`}>{icon}</span>
      <span className="q-stat-text">
        <strong>{value}</strong>
        <small>{label}</small>
      </span>
    </div>
  );
}

export function Pill({ children, tone = "green", dot }: { children: ReactNode; tone?: "green" | "amber" | "red" | "blue"; dot?: boolean }) {
  return (
    <span className={`q-pill ${tone}`}>
      {dot && <i />}
      {children}
    </span>
  );
}

// Status cell used by the route tables: coloured dot + label.
export function Status({ tone, children }: { tone: "green" | "amber" | "red" | "grey"; children: ReactNode }) {
  return (
    <span className={`q-status ${tone}`}>
      <i />
      {children}
    </span>
  );
}

export function Loader({ label }: { label: string }) {
  return (
    <div className="q-loader">
      <span className="q-spinner" />
      {label}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="q-empty">{children}</p>;
}

// CSV order upload — parses the file, keeps the rows the backend accepts.
export type ParsedOrders = {
  fileName: string;
  rows: { id: string; node: string; demand: number; earliest_s: number; latest_s: number; service_s: number }[];
  note: string;
};

export function parseOrdersCsv(text: string, fileName: string, validNodes: string[], fallbackNodes: string[]): ParsedOrders | { error: string } {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines.length < 2) return { error: `${fileName}: no data rows found (header + rows expected).` };
  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const col = (names: string[]) => names.map((n) => header.indexOf(n)).find((i) => i >= 0) ?? -1;
  const idAt = col(["id", "order_id", "request_id", "job"]);
  const nodeAt = col(["node", "access_node_id", "node_id", "location"]);
  const demandAt = col(["demand", "qty", "quantity", "load"]);
  const earlyAt = col(["earliest_s", "earliest", "start_s"]);
  const lateAt = col(["latest_s", "latest", "end_s", "deadline"]);
  const serviceAt = col(["service_s", "service", "service_time"]);
  if (demandAt < 0) return { error: `${fileName}: a "demand" column is required.` };
  const known = new Set(validNodes);
  const rows: ParsedOrders["rows"] = [];
  let snapped = 0;
  let dropped = 0;
  lines.slice(1).forEach((line, index) => {
    const cells = line.split(",").map((c) => c.trim());
    const demand = Number(cells[demandAt]);
    if (!Number.isFinite(demand)) return;
    let node = nodeAt >= 0 ? cells[nodeAt] : "";
    if (!known.has(node)) {
      if (node) snapped += 1;
      node = fallbackNodes[index % Math.max(1, fallbackNodes.length)] || "";
    }
    if (!node) {
      dropped += 1;
      return;
    }
    rows.push({
      id: idAt >= 0 && cells[idAt] ? cells[idAt] : `ORD-${index + 1}`,
      node,
      demand,
      earliest_s: earlyAt >= 0 && Number.isFinite(Number(cells[earlyAt])) ? Number(cells[earlyAt]) : 0,
      latest_s: lateAt >= 0 && Number.isFinite(Number(cells[lateAt])) ? Number(cells[lateAt]) : 28800,
      service_s: serviceAt >= 0 && Number.isFinite(Number(cells[serviceAt])) ? Number(cells[serviceAt]) : 30,
    });
  });
  if (!rows.length) return { error: `${fileName}: no usable rows (need a numeric demand column).` };
  if (rows.length > 200) return { error: `${fileName}: ${rows.length} rows — the workspace accepts at most 200 deliveries.` };
  const notes = [`${rows.length} orders loaded`];
  if (snapped) notes.push(`${snapped} snapped to the nearest known delivery road`);
  if (dropped) notes.push(`${dropped} skipped (no delivery node)`);
  return { fileName, rows, note: `${rows.length} orders loaded` };
}

// Simple count-up for KPI numbers (skips when reduced motion is requested).
export function useCountUp(target: number | null, ms = 900) {
  const [value, setValue] = useState(0);
  const fromRef = useRef(0);
  useEffect(() => {
    if (target == null || !Number.isFinite(target)) return;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
    if (reduced) {
      setValue(target);
      return;
    }
    const from = fromRef.current;
    const start = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const pct = Math.min(1, (now - start) / ms);
      const eased = 1 - (1 - pct) ** 3;
      const next = from + (target - from) * eased;
      setValue(next);
      fromRef.current = next;
      if (pct < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return value;
}

// Scenario list → select options grouped like the image (city fixtures first).
export function scenarioOptions(scenarios: ScenarioSummary[]) {
  const available = scenarios.filter((s) => s.available);
  const list = available.length ? available : scenarios;
  const fixtures = list.filter((s) => !s.id.startsWith("DELHI_"));
  const delhi = list.filter((s) => s.id.startsWith("DELHI_"));
  return [
    ...fixtures.map((s) => ({ value: s.id, label: s.label, group: "City fixtures" })),
    ...delhi.map((s) => ({ value: s.id, label: s.label, group: "Real Delhi maps" })),
  ];
}

export const revealStyle = (delay: number) => ({ "--d": `${delay}ms` } as CSSProperties);
