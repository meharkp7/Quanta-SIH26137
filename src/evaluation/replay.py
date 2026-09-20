"""Step 22 skeleton: deterministic read-only replay of stored routes.

Given an evidence JSON containing routes plus a scenario reference, each
stored route is re-validated through the project's independent validator
path used in ``src.optim.qpso_vs_alns`` (its ``validate_route`` helper,
which combines ``src.routing.validator.evaluate_scenario`` with a
``RouteEvaluator`` re-check).  Reports agree/disagree per route.

Read-only: evidence files are never mutated; replay builds fresh evaluator
instances over a freshly loaded scenario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.contracts.scenario import Scenario
from src.optim.qpso_vs_alns import snapshot_config, validate_route
from src.routing.route_evaluator import RouteEvaluator

from .compare import REPO_ROOT, load_evidence

RUN_LIST_KEYS = ("qpso_runs", "alns_runs", "runs")


def _resolve_fixture(evidence: dict[str, Any], evidence_path: str | Path | None) -> Path:
    fixture = evidence.get("fixture")
    if fixture:
        candidate = Path(str(fixture))
        if candidate.is_absolute() and candidate.exists():
            return candidate
        repo_candidate = REPO_ROOT / candidate
        if repo_candidate.exists():
            return repo_candidate
    if evidence_path is not None:
        sibling = Path(evidence_path).parent / "scenario.json"
        if sibling.exists():
            return sibling
    raise FileNotFoundError("cannot resolve scenario fixture for replay")


def replay_evidence(
    evidence_path: str | Path,
    *,
    fixture: str | Path | None = None,
) -> dict[str, Any]:
    """Re-validate every stored route in one evidence file.

    Returns ``{evidence, fixture, results[]}`` where each result carries
    ``method, seed, stored_*`` fields plus ``replay_*`` fields and an
    ``agree`` flag (replay verdict matches the stored verdict) and a
    ``disagree`` flag per route is implicit in ``agree == False``.
    """
    evidence = load_evidence(evidence_path)
    fixture_path = Path(fixture) if fixture else _resolve_fixture(evidence, evidence_path)
    scenario = Scenario.model_validate_json(Path(fixture_path).read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    for key in RUN_LIST_KEYS:
        runs = evidence.get(key, [])
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            route = run.get("route")
            # Step8 trace records store routes as signature lists; skip
            # records without a replayable vehicle->customers mapping.
            if not isinstance(route, dict) or not route:
                continue
            evaluator = RouteEvaluator(scenario, config=snapshot_config())
            replay = validate_route(
                scenario, evaluator, {str(k): [str(c) for c in v] for k, v in route.items()}
            )
            stored = run.get("validator_verdict") if isinstance(run.get("validator_verdict"), dict) else {}
            stored_feas = stored.get("evaluator_feasible")
            stored_agree = stored.get("verdict_agree")
            entry: dict[str, Any] = {
                "method": str(run.get("method", key.removesuffix("_runs"))),
                "seed": run.get("seed"),
                "route": {str(k): [str(c) for c in v] for k, v in route.items()},
                "stored_evaluator_feasible": stored_feas,
                "stored_verdict_agree": stored_agree,
                "stored_objective": stored.get("evaluator_objective"),
                "replay_evaluator_feasible": replay["evaluator_feasible"],
                "replay_independent_feasible": replay["independent_validator_feasible"],
                "replay_objective": replay["evaluator_objective"],
                "replay_verdict_agree": replay["verdict_agree"],
                "agree": (
                    (bool(stored_feas) == bool(replay["evaluator_feasible"]))
                    and (bool(stored_agree) == bool(replay["verdict_agree"]))
                    if stored
                    else bool(replay["verdict_agree"])
                ),
            }
            # Objective cross-check where the stored record has one.
            if stored.get("evaluator_objective") is not None:
                entry["objective_match"] = abs(
                    float(stored["evaluator_objective"]) - float(replay["evaluator_objective"])
                ) <= 1e-9
                entry["agree"] = bool(entry["agree"]) and bool(entry["objective_match"])
            results.append(entry)

    n_agree = sum(1 for r in results if r["agree"])
    return {
        "evidence": str(evidence_path),
        "fixture": str(fixture_path),
        "n_routes": len(results),
        "n_agree": n_agree,
        "n_disagree": len(results) - n_agree,
        "results": results,
    }
