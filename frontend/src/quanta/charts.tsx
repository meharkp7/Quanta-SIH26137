import { useMemo } from "react";

// Compact objective label for log-scale ticks (122230 → 122k, 56302358 → 56m).
function formatObjective(v: number, wide: boolean): string {
  if (!wide) return Math.round(v).toLocaleString();
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${(v / 1e9).toFixed(1)}b`;
  if (abs >= 1e6) return `${(v / 1e6).toFixed(1)}m`;
  if (abs >= 1e3) return `${(v / 1e3).toFixed(0)}k`;
  return Math.round(v).toLocaleString();
}

// ── Convergence chart: optimizer best-objective trace (real /api/solve data) ─
// The best-so-far series is often flat on a big city graph (the constructive
// start is already strong), so the population mean rides along as a second
// dashed series — both are recorded by the backend, nothing is invented.
export function ConvergenceChart({ values, label, mean, meanLabel, height = 190 }: {
  values: number[];
  label: string;
  mean?: number[];
  meanLabel?: string;
  height?: number;
}) {
  const view = useMemo(() => {
    const W = 520;
    const H = height;
    const padL = 46;
    const padR = 12;
    const padT = 14;
    const padB = 26;
    const data = values.length ? values : [];
    if (data.length < 2) return null;
    const second = (mean || []).filter((v) => Number.isFinite(v));
    // One scale for both series: they are objectives in the same units.
    const all = [...data, ...second];
    const min = Math.min(...all);
    const max = Math.max(...all);
    const span = max - min || Math.abs(max) || 1;
    // Objectives span orders of magnitude on real city graphs → log-y when wide.
    const useLog = max > 0 && min > 0 && max / Math.max(min, 1e-9) > 8;
    const yOf = (v: number) => {
      if (useLog) {
        const lo = Math.log10(Math.max(min * 0.98, 1e-6));
        const hi = Math.log10(max * 1.02);
        const t = hi === lo ? 0.5 : (Math.log10(Math.max(v, 1e-6)) - lo) / (hi - lo);
        return padT + (1 - t) * (H - padT - padB);
      }
      return padT + (1 - (v - min + span * 0.08) / (span * 1.16)) * (H - padT - padB);
    };
    const xOf = (i: number) => padL + (i / (data.length - 1)) * (W - padL - padR);
    const points = data.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
    const area = `${padL},${H - padB} ${points} ${W - padR},${H - padB}`;
    const secondPoints = second.length > 1
      ? second.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ")
      : null;
    const ticks = [0, Math.floor((data.length - 1) / 2), data.length - 1];
    const yTicks = useLog
      ? [min, (Math.sqrt(min * max)), max]
      : [min, min + span / 2, max];
    return { W, H, padL, padR, padT, padB, points, area, secondPoints, ticks, yTicks, yOf, xOf, min, max, useLog, n: data.length };
  }, [values, mean, height]);

  if (!view) return <p className="q-empty">Run a solve to record the optimizer's convergence trace.</p>;
  const last = values[values.length - 1];
  const first = values[0];
  const improved = first - last;
  return (
    <div className="q-chart">
      <svg viewBox={`0 0 ${view.W} ${view.H}`} role="img" aria-label="Algorithm convergence">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <line key={f} x1={view.padL} x2={view.W - view.padR} y1={view.padT + f * (view.H - view.padT - view.padB)} y2={view.padT + f * (view.H - view.padT - view.padB)} className="q-grid" />
        ))}
        <polygon points={view.area} className="q-chart-area" />
        {view.secondPoints && <polyline points={view.secondPoints} className="q-chart-line-2" fill="none" />}
        <polyline points={view.points} className="q-chart-line" fill="none" />
        <circle cx={view.xOf(view.n - 1)} cy={view.yOf(last)} r="4" className="q-chart-head" />
        {view.yTicks.map((v, i) => (
          <text key={i} x={view.padL - 7} y={view.yOf(v) + 3} textAnchor="end" className="q-axis">
            {formatObjective(v, view.useLog)}
          </text>
        ))}
        {view.ticks.map((i) => (
          <text key={i} x={view.xOf(i)} y={view.H - 8} textAnchor="middle" className="q-axis">
            {i}
          </text>
        ))}
      </svg>
      <div className="q-chart-foot">
        <span className="q-chart-legend">
          <i className="q-swatch line" /> {label}
        </span>
        {view.secondPoints && (
          <span className="q-chart-legend">
            <i className="q-swatch line-2" /> {meanLabel || "population mean"}
          </span>
        )}
        <span className="q-axis-name">Iterations → {view.n} evaluated</span>
        {improved > 0 && <span className="q-chart-good">−{improved.toFixed(0)} objective from the first candidate</span>}
      </div>
    </div>
  );
}

// ── Road-level forecast chart: recorded history + forecast with P10–P90 band ─
export function ForecastChart({ history, forecast, markerAt, unit, height = 230 }: {
  history: { t: number; v: number }[];
  forecast: { t: number; v: number; lo?: number | null; hi?: number | null }[];
  markerAt?: number | null;
  unit: string;
  height?: number;
}) {
  const view = useMemo(() => {
    const W = 640;
    const H = height;
    const padL = 44;
    const padR = 14;
    const padT = 14;
    const padB = 28;
    const all = [
      ...history.map((p) => p.v),
      ...forecast.flatMap((p) => [p.v, p.lo ?? p.v, p.hi ?? p.v]),
    ].filter((v) => Number.isFinite(v));
    if (all.length < 2) return null;
    let min = Math.min(...all);
    let max = Math.max(...all);
    const pad = (max - min || 1) * 0.15;
    min = Math.max(0, min - pad);
    max = max + pad;
    const times = [...history.map((p) => p.t), ...forecast.map((p) => p.t)].sort((a, b) => a - b);
    const t0 = times[0];
    const t1 = times[times.length - 1];
    const xOf = (t: number) => padL + ((t - t0) / Math.max(1, t1 - t0)) * (W - padL - padR);
    const yOf = (v: number) => padT + (1 - (v - min) / Math.max(1e-6, max - min)) * (H - padT - padB);
    const line = (pts: { t: number; v: number }[]) => pts.map((p) => `${xOf(p.t).toFixed(1)},${yOf(p.v).toFixed(1)}`).join(" ");
    const bandTop = forecast.filter((p) => p.hi != null).map((p) => `${xOf(p.t).toFixed(1)},${yOf(p.hi as number).toFixed(1)}`);
    const bandBottom = forecast.filter((p) => p.lo != null).reverse().map((p) => `${xOf(p.t).toFixed(1)},${yOf(p.lo as number).toFixed(1)}`);
    const band = bandTop.length && bandBottom.length ? [...bandTop, ...bandBottom].join(" ") : null;
    const yTicks = [min, min + (max - min) / 2, max];
    return { W, H, padL, padR, padT, padB, xOf, yOf, line, band, yTicks, t0, t1, min, max };
  }, [history, forecast, height]);

  if (!view) return <p className="q-empty">No forecast issues recorded for this episode yet.</p>;
  const xTicks = (() => {
    const ts = [...history.map((p) => p.t), ...forecast.map((p) => p.t)].sort((a, b) => a - b);
    if (!ts.length) return [];
    const picks = [ts[0], ts[Math.floor(ts.length / 3)], ts[Math.floor((2 * ts.length) / 3)], ts[ts.length - 1]];
    return Array.from(new Set(picks));
  })();
  const label = (t: number) => {
    const base = Math.floor(t / 60);
    return `${String(Math.floor(base / 60)).padStart(2, "0")}:${String(base % 60).padStart(2, "0")}`;
  };
  return (
    <div className="q-chart">
      <svg viewBox={`0 0 ${view.W} ${view.H}`} role="img" aria-label="Road-level traffic forecast">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <line key={f} x1={view.padL} x2={view.W - view.padR} y1={view.padT + f * (view.H - view.padT - view.padB)} y2={view.padT + f * (view.H - view.padT - view.padB)} className="q-grid" />
        ))}
        {view.band && <polygon points={view.band} className="q-band" />}
        {history.length > 1 && <polyline points={view.line(history)} className="q-hist-line" fill="none" />}
        {forecast.length > 1 && <polyline points={view.line(forecast)} className="q-fc-line" fill="none" />}
        {forecast.length > 0 && <circle cx={view.xOf(forecast[forecast.length - 1].t)} cy={view.yOf(forecast[forecast.length - 1].v)} r="4" className="q-fc-head" />}
        {markerAt != null && markerAt >= view.t0 && markerAt <= view.t1 && (
          <>
            <line x1={view.xOf(markerAt)} x2={view.xOf(markerAt)} y1={view.padT} y2={view.H - view.padB} className="q-marker" />
            <text x={view.xOf(markerAt) + 5} y={view.padT + 11} className="q-axis">model start</text>
          </>
        )}
        {view.yTicks.map((v, i) => (
          <text key={i} x={view.padL - 7} y={view.yOf(v) + 3} textAnchor="end" className="q-axis">{v.toFixed(0)}</text>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={view.xOf(t)} y={view.H - 8} textAnchor="middle" className="q-axis">{label(t)}</text>
        ))}
      </svg>
      <div className="q-chart-foot">
        <span className="q-chart-legend"><i className="q-swatch hist" /> Historical</span>
        <span className="q-chart-legend"><i className="q-swatch fc" /> Forecast</span>
        <span className="q-axis-name">{unit}</span>
      </div>
    </div>
  );
}

// Small horizontal objective bars for the algorithm comparison view.
export function ObjectiveBars({ rows }: { rows: { method: string; objective: number; feasible: boolean }[] }) {
  const max = Math.max(1, ...rows.map((r) => r.objective));
  return (
    <div className="q-bars">
      {rows.map((row) => (
        <div className="q-bar-row" key={row.method}>
          <span className="q-bar-name">{row.method}</span>
          <div className="q-bar-track">
            <div
              className={`q-bar-fill ${row.feasible ? "" : "bad"}`}
              style={{ width: `${Math.max(3, (row.objective / max) * 100)}%` } as React.CSSProperties}
            />
          </div>
          <strong>{row.objective.toLocaleString(undefined, { maximumFractionDigits: 0 })}</strong>
        </div>
      ))}
    </div>
  );
}
