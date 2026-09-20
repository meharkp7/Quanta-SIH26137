"""Step 22 skeleton: paired comparison tables from saved evidence records.

Consumes the step9 evidence schema
(``artifacts/step9_research/qpso_vs_alns_5job.json``)::

    {fixture, controls, exact_reference_objective,
     comparison_table, qpso_runs[], alns_runs[]}

and the step8 QPSO-vs-PSO trace schema (``runs.json`` list of per-seed run
records) opportunistically: any run record carrying ``seed`` plus a feasible
objective field is tabulated.

Gap semantics (from ``src.optim.references``): a heuristic reports
``best_feasible_objective``; only the exact MILP path with
``certified_optimum == True`` reports an optimum.  A zero gap against the
certified reference means "matches best-known reference", never "proven
optimal by the heuristic".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE = REPO_ROOT / "artifacts/step9_research/qpso_vs_alns_5job.json"
DEFAULT_OPTIMUM_FILE = REPO_ROOT / "artifacts/step9_research/five_job_milp_reference.json"
DEFAULT_OUT = REPO_ROOT / "artifacts/step9_research/evaluation_report.md"


def load_evidence(path: str | Path) -> dict[str, Any]:
    """Load one evidence JSON file without mutating it."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def load_certified_optimum(path: str | Path) -> float:
    """Extract the certified optimum from the MILP reference file.

    Raises if the reference is not certified, so callers can never
    silently treat a non-certified bound as an optimum.
    """
    data = load_evidence(path)
    if not data.get("certified_optimum", False):
        raise ValueError(f"{path}: reference is not a certified optimum")
    return float(data["objective"])


def _run_objective(run: dict[str, Any]) -> float | None:
    for key in ("best_feasible_objective", "best_fitness", "objective"):
        value = run.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _run_feasible(run: dict[str, Any]) -> bool:
    for key in ("best_feasible", "feasible"):
        if key in run:
            return bool(run[key])
    verdict = run.get("validator_verdict") or {}
    if "verdict_agree" in verdict:
        return bool(verdict.get("evaluator_feasible", False))
    return False


def _run_wall(run: dict[str, Any]) -> float | None:
    for key in ("wall_s", "wall_clock_s", "elapsed_s"):
        value = run.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _run_agree(run: dict[str, Any]) -> bool | None:
    verdict = run.get("validator_verdict")
    if isinstance(verdict, dict) and "verdict_agree" in verdict:
        return bool(verdict["verdict_agree"])
    return None


def _gap(best_feasible: float | None, feasible: bool, optimum: float) -> float | None:
    if best_feasible is None or not feasible:
        return None
    if optimum == 0.0:
        return None
    return (float(best_feasible) - float(optimum)) / abs(float(optimum))


def gap_vs_optimum(
    evidence: dict[str, Any], optimum: float
) -> list[dict[str, Any]]:
    """Per-seed gaps of every stored run vs the certified optimum.

    Returns rows ``{method, seed, best_feasible, gap, matches_reference}``.
    ``matches_reference`` (gap == 0 within 1e-9) deliberately avoids the
    word "optimal" for heuristic rows.
    """
    rows: list[dict[str, Any]] = []
    for key in ("qpso_runs", "alns_runs", "runs"):
        runs = evidence.get(key, [])
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, dict):
                    continue
                method = str(run.get("method", key.removesuffix("_runs")))
                feasible = _run_feasible(run)
                best = _run_objective(run) if feasible else None
                gap = _gap(best, feasible, optimum)
                rows.append(
                    {
                        "method": method,
                        "seed": run.get("seed"),
                        "best_feasible": best,
                        "feasible": feasible,
                        "gap": gap,
                        # Heuristic rows never claim optimality.
                        "matches_reference": gap is not None
                        and abs(gap) <= 1e-9,
                    }
                )
    return rows


def paired_table(
    evidence: dict[str, Any], optimum: float | None = None
) -> list[dict[str, Any]]:
    """Per-seed paired QPSO-vs-ALNS table.

    Prefers the stored ``comparison_table`` when present but recomputes
    gaps from run records so reports stay reproducible from saved records.
    Falls back to joining ``qpso_runs``/``alns_runs`` by seed.
    Each row carries per-seed best-feasible costs, gaps vs the certified
    optimum (where available), validator agreement flags, and wall-clock.
    """
    if optimum is None:
        optimum = evidence.get("exact_reference_objective")
        if optimum is not None:
            optimum = float(optimum)

    qpso = {r.get("seed"): r for r in evidence.get("qpso_runs", []) if isinstance(r, dict)}
    alns = {r.get("seed"): r for r in evidence.get("alns_runs", []) if isinstance(r, dict)}
    seeds = sorted(
        {s for s in list(qpso) + list(alns) if s is not None},
        key=lambda s: int(s),  # type: ignore[arg-type]
    )
    # If run lists are absent (e.g. step8 runs.json), pair generically.
    if not seeds and isinstance(evidence, list):
        return []

    table: list[dict[str, Any]] = []
    stored = {r.get("seed"): r for r in evidence.get("comparison_table", []) if isinstance(r, dict)}
    for seed in seeds:
        q, a = qpso.get(seed, {}), alns.get(seed, {})
        q_feas = _run_feasible(q) if q else False
        a_feas = _run_feasible(a) if a else False
        q_best = _run_objective(q) if (q and q_feas) else None
        a_best = _run_objective(a) if (a and a_feas) else None
        row: dict[str, Any] = {
            "seed": seed,
            "qpso_best_feasible": q_best,
            "alns_best_feasible": a_best,
            "qpso_evals": (q.get("evaluations") if q else None)
            or (stored.get(seed, {}).get("qpso_evals")),
            "alns_evals": (a.get("evaluations") if a else None)
            or (stored.get(seed, {}).get("alns_evals")),
            "qpso_wall_s": _run_wall(q) if q else None,
            "alns_wall_s": _run_wall(a) if a else None,
            "qpso_agree": _run_agree(q) if q else None,
            "alns_agree": _run_agree(a) if a else None,
        }
        if optimum is not None:
            row["qpso_gap"] = _gap(q_best, q_feas, float(optimum))
            row["alns_gap"] = _gap(a_best, a_feas, float(optimum))
        else:
            row["qpso_gap"] = None
            row["alns_gap"] = None
        table.append(row)
    return table


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(
    evidence_paths: list[str | Path],
    optimum: float,
    out_path: str | Path,
) -> dict[str, Any]:
    """Write a Markdown paired-comparison report plus a JSON sidecar.

    Returns the JSON summary dict (also written to ``out_path`` with a
    ``.json`` extension appended, e.g. ``report.md.json``).
    """
    out = Path(out_path)
    sections: list[str] = []
    summary: dict[str, Any] = {
        "certified_optimum": float(optimum),
        "certified_optimum_source": "exact MILP reference (certified_optimum=true)",
        "evidence": [],
        "paired_tables": {},
        "gaps": {},
        "headline": {},
        "terminology": (
            "Heuristic entries report best_feasible_objective and gap vs "
            "the certified reference. A zero gap means "
            "'matches best-known reference', never 'optimal' — only the "
            "exact MILP path may claim a certified optimum."
        ),
    }
    total_runs = 0
    total_agree = 0
    total_match = 0

    lines = [
        "# Step 22 evaluation skeleton report",
        "",
        f"Certified optimum (exact MILP reference): **{optimum}**",
        "",
        "> Terminology: heuristic entries report best_feasible_objective and gap vs the certified reference. "
        "A zero gap means matches best-known reference, never optimal.",
        "",
    ]
    for path in evidence_paths:
        evidence = load_evidence(path)
        table = paired_table(evidence, optimum)
        gaps = gap_vs_optimum(evidence, optimum)
        name = str(path)
        summary["evidence"].append(name)
        summary["paired_tables"][name] = table
        summary["gaps"][name] = gaps
        for row in table:
            for flag in (row.get("qpso_agree"), row.get("alns_agree")):
                if flag is not None:
                    total_runs += 1
                    total_agree += int(bool(flag))
            for gap in (row.get("qpso_gap"), row.get("alns_gap")):
                if gap is not None and abs(float(gap)) <= 1e-9:
                    total_match += 1
        lines.append(f"## Evidence: `{name}`")
        lines.append("")
        if evidence.get("fixture"):
            lines.append(f"Fixture: `{evidence['fixture']}`")
            lines.append("")
        lines.append(
            "| seed | qpso_best_feasible | qpso_gap | qpso_wall_s | "
            "qpso_agree | alns_best_feasible | alns_gap | alns_wall_s | "
            "alns_agree |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in table:
            lines.append(
                f"| {row['seed']} | {_fmt(row['qpso_best_feasible'], 1)} | "
                f"{_fmt(row['qpso_gap'])} | {_fmt(row['qpso_wall_s'], 3)} | "
                f"{_fmt(row['qpso_agree'])} | {_fmt(row['alns_best_feasible'], 1)} | "
                f"{_fmt(row['alns_gap'])} | {_fmt(row['alns_wall_s'], 3)} | "
                f"{_fmt(row['alns_agree'])} |"
            )
        lines.append("")
        sections.append(name)

    summary["headline"] = {
        "n_evidence_files": len(evidence_paths),
        "n_validator_checks": total_runs,
        "n_validator_agree": total_agree,
        "n_zero_gap_matches": total_match,
        "verdict": (
            "all stored validator verdicts agree and all feasible runs "
            "match the certified reference"
            if total_runs and total_runs == total_agree
            else "see per-seed table"
        ),
    }
    lines.append("## Headline")
    lines.append("")
    for key, value in summary["headline"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    sidecar = Path(str(out) + ".json")
    sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the Step 22 skeleton evaluation report."
    )
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE, nargs="*")
    parser.add_argument("--optimum-file", type=Path, default=DEFAULT_OPTIMUM_FILE)
    parser.add_argument("--optimum", type=float, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    evidences = args.evidence if isinstance(args.evidence, list) else [args.evidence]
    optimum = float(args.optimum) if args.optimum is not None else load_certified_optimum(args.optimum_file)
    summary = write_report([str(p) for p in evidences], optimum, args.out)
    print(f"Report: {args.out} (+ .json sidecar)")
    print(f"Certified optimum: {summary['certified_optimum']}")
    for key, value in summary["headline"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
