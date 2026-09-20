"""Step 22 skeleton tests: paired-table math, gap semantics, replay, CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.evaluation import compare, replay

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP9 = REPO_ROOT / "artifacts/step9_research/qpso_vs_alns_5job.json"
OPT_FILE = REPO_ROOT / "artifacts/step9_research/five_job_milp_reference.json"


def _synthetic() -> dict:
    def run(method: str, seed: int, obj: float | None, feas: bool = True):
        return {
            "method": method,
            "seed": seed,
            "best_feasible": feas,
            "best_feasible_objective": obj,
            "evaluations": 2000,
            "wall_s": 0.1 * seed,
            "route": {"V1": ["J1"]},
            "validator_verdict": {
                "evaluator_feasible": feas,
                "evaluator_objective": obj,
                "independent_validator_feasible": feas,
                "verdict_agree": True,
            },
        }

    return {
        "fixture": "fixtures/step3/base/scenario.json",
        "exact_reference_objective": 100.0,
        "qpso_runs": [run("qpso", 1, 110.0), run("qpso", 2, None, False)],
        "alns_runs": [run("alns", 1, 105.0), run("alns", 2, 100.0)],
    }


def test_paired_table_math_synthetic():
    ev = _synthetic()
    table = compare.paired_table(ev, 100.0)
    assert len(table) == 2
    row1 = next(r for r in table if r["seed"] == 1)
    assert row1["qpso_gap"] == pytest.approx(0.10)
    assert row1["alns_gap"] == pytest.approx(0.05)
    assert row1["qpso_wall_s"] == pytest.approx(0.1)
    assert row1["qpso_agree"] is True
    row2 = next(r for r in table if r["seed"] == 2)
    assert row2["qpso_best_feasible"] is None
    assert row2["qpso_gap"] is None
    assert row2["alns_gap"] == pytest.approx(0.0)


def test_gap_semantics_never_optimal(tmp_path):
    ev = _synthetic()
    rows = compare.gap_vs_optimum(ev, 100.0)
    assert rows
    for row in rows:
        assert "optimal" not in {k.lower() for k in row}
        assert "matches_reference" in row
    match = next(r for r in rows if r["method"] == "alns" and r["seed"] == 2)
    assert match["gap"] == pytest.approx(0.0)
    assert match["matches_reference"] is True
    # Report wording must not label heuristics optimal.
    out = tmp_path / "rep.md"
    compare.write_report([STEP9], 1575.0, out)
    text = " ".join(out.read_text(encoding="utf-8").lower().split())
    assert "matches best-known reference" in text
    assert "qpso optimal" not in text and "alns optimal" not in text


def test_replay_agreement_real_step9():
    result = replay.replay_evidence(STEP9)
    assert result["n_routes"] == 6  # 3 qpso + 3 alns stored routes
    assert result["n_disagree"] == 0
    assert all(r["agree"] for r in result["results"])


def test_replay_read_only():
    before = STEP9.read_bytes()
    replay.replay_evidence(STEP9)
    assert STEP9.read_bytes() == before


def test_cli_smoke(tmp_path):
    out = tmp_path / "evaluation_report.md"
    proc = subprocess.run(
        [sys.executable, "-m", "src.evaluation.compare", "--out", str(out)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    sidecar = Path(str(out) + ".json")
    assert sidecar.exists()
    summary = json.loads(sidecar.read_text(encoding="utf-8"))
    assert summary["certified_optimum"] == pytest.approx(1575.0)
