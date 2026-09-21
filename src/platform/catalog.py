"""Scenario discovery and loading for the demo platform."""

from __future__ import annotations

import json
from pathlib import Path

from src.contracts.scenario import Scenario

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "fixtures" / "step3"

SCENARIO_CATALOG = (
    {
        "id": "S3_BASE",
        "label": "5-job city fixture",
        "role": "live demo",
        "path": FIXTURE_ROOT / "base" / "scenario.json",
        "description": "One depot, five deliveries, two vehicles, directed roads.",
    },
    {
        "id": "feasible_reference",
        "label": "Known feasible plan",
        "role": "validator pass",
        "path": FIXTURE_ROOT / "cases" / "feasible_reference" / "scenario.json",
        "description": "Hand-checked legal routes used as a regression fixture.",
    },
    {
        "id": "excess_load",
        "label": "Overloaded vehicle",
        "role": "validator fail",
        "path": FIXTURE_ROOT / "cases" / "excess_load" / "scenario.json",
        "description": "Capacity violation the independent checker must reject.",
    },
    {
        "id": "missed_window",
        "label": "Missed time window",
        "role": "validator fail",
        "path": FIXTURE_ROOT / "cases" / "missed_window" / "scenario.json",
        "description": "A late service start the checker must reject.",
    },
    {
        "id": "closure_with_detour",
        "label": "Closure + detour",
        "role": "dynamic traffic",
        "path": FIXTURE_ROOT / "cases" / "closure_with_detour" / "scenario.json",
        "description": "A closeable road with a legal alternative path.",
    },
    {
        "id": "early_arrival_wait",
        "label": "Early arrival / wait",
        "role": "windows",
        "path": FIXTURE_ROOT / "cases" / "early_arrival_wait" / "scenario.json",
        "description": "Vehicle arrives early and must wait for the window.",
    },
    {
        "id": "wrong_way",
        "label": "Wrong-way path",
        "role": "validator fail",
        "path": FIXTURE_ROOT / "cases" / "wrong_way" / "scenario.json",
        "description": "Illegal directed movement the checker must reject.",
    },
    {
        "id": "DELHI_CP",
        "label": "Delhi — Dwarka Sector 12 (real OSM)",
        "role": "real Delhi render",
        "path": PROJECT_ROOT / "artifacts" / "corpus_v2" / "maps" / "map_003" / "scenario.json",
        "description": "Real OSM Delhi network (dwarka_sector12-map003, validation split): 1929 edges, 841 nodes, 70 requests, 15 vehicles. Read-only reference; render/solve via the same live service.",
    },
)


def list_scenarios() -> list[dict]:
    rows = []
    for item in SCENARIO_CATALOG:
        path = Path(item["path"])
        rows.append(
            {
                "id": item["id"],
                "label": item["label"],
                "role": item["role"],
                "description": item["description"],
                "available": path.is_file(),
            }
        )
    return rows


def scenario_entry(scenario_id: str) -> dict:
    for item in SCENARIO_CATALOG:
        if item["id"] == scenario_id:
            return item
    raise KeyError(f"Unknown scenario: {scenario_id}")


def load_scenario(scenario_id: str = "S3_BASE") -> Scenario:
    path = Path(scenario_entry(scenario_id)["path"])
    return Scenario.model_validate_json(path.read_text(encoding="utf-8"))


def load_scenario_json(scenario_id: str = "S3_BASE") -> dict:
    path = Path(scenario_entry(scenario_id)["path"])
    return json.loads(path.read_text(encoding="utf-8"))
