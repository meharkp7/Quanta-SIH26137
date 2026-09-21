"""FastAPI API and web dispatcher for the current SIH26137 platform.

Run from the repository root:

    python -m app.server

The API is intentionally a thin facade over ``src.platform.service`` and
``src.runtime.loop``.  Solver, validation, routing, and SUMO behavior stay in
the domain modules so the HTTP layer does not duplicate project logic.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.platform.service import PlatformService, SolveOptions
from src.runtime.loop import DemoLoop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

service = PlatformService()
loop = DemoLoop(service)

app = FastAPI(
    title="Quanta SIH26137",
    description=(
        "Adaptive quantum-inspired vehicle routing under dynamic traffic. "
        "The API exposes the current platform service, independent validator, "
        "runtime loop, shortest-path reference, solver comparisons, and SUMO replay."
    ),
    version="0.4.0",
)

SolveMethod = Literal["constructive", "qpso", "pso", "alns", "milp"]


class SolveRequest(BaseModel):
    scenario_id: str = Field(default="S3_BASE", min_length=1)
    method: SolveMethod = "qpso"
    particles: int = Field(default=12, ge=2, le=40)
    evaluations: int = Field(default=40, ge=4, le=400)
    seed: int = 7
    closed_edge_ids: list[str] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    scenario_id: str = Field(default="S3_BASE", min_length=1)
    plan: dict[str, list[str]] | None = None
    closed_edge_ids: list[str] = Field(default_factory=list)


class PathRequest(BaseModel):
    scenario_id: str = Field(default="S3_BASE", min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    closed_edge_ids: list[str] = Field(default_factory=list)


class CompareRequest(SolveRequest):
    methods: list[SolveMethod] = Field(
        default_factory=lambda: ["constructive", "qpso", "pso", "alns", "milp"]
    )


class LoopRequest(SolveRequest):
    execute_sumo: bool = True


class SumoRequest(BaseModel):
    gui: bool = True


class ReplayRequest(BaseModel):
    scenario_id: str = Field(default="S3_BASE", min_length=1)
    plan: dict[str, list[str]] | None = None
    closed_edge_ids: list[str] = Field(default_factory=list)


def _solve_options(request: SolveRequest) -> SolveOptions:
    """Convert an API request to the domain-level solver options."""
    return SolveOptions(
        method=request.method,
        particles=request.particles,
        evaluations=request.evaluations,
        seed=request.seed,
        closed_edge_ids=tuple(request.closed_edge_ids),
    )


def _bad_request(exc: Exception) -> HTTPException:
    """Keep internal exception details useful while using a consistent status."""
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/health")
def health() -> dict:
    """Lightweight liveness/readiness information for the dispatcher."""
    return {
        "status": "ok",
        "service": "Quanta SIH26137",
        "repository_root": str(PROJECT_ROOT),
        "static_dir": str(STATIC_DIR),
    }


@app.get("/api/meta")
def meta() -> dict:
    return service.project_meta()


@app.get("/api/evidence")
def evidence() -> dict:
    """Expose saved Step 13/14 measurements for the presentation UI."""
    artifact_dir = PROJECT_ROOT / "artifacts" / "step13_forecaster_v1"
    manifest_path = artifact_dir / "manifest.json"
    uncertainty_path = artifact_dir / "uncertainty.json"
    if not manifest_path.is_file() or not uncertainty_path.is_file():
        demo_path = PROJECT_ROOT / "artifacts" / "demo_evidence.json"
        if demo_path.is_file():
            demo = json.loads(demo_path.read_text(encoding="utf-8"))
            gnn = demo.get("gnn_pilot", {})
            horizons = gnn.get("mae_by_horizon_s", {})
            test_metrics = (
                {str(h): {"mae": float(v)} for h, v in horizons.items()}
                if horizons
                else {}
            )
            return {
                "available": True,
                "demo": True,
                "provenance": demo.get(
                    "provenance",
                    "Measured on deleted 250-episode pilot corpus; v2 corpus training pending; values are static demo, not live inference.",
                ),
                "artifact": "demo_evidence",
                "model_mae": sum(float(v) for v in horizons.values()) / len(horizons) if horizons else None,
                "temporal_baseline_mae": gnn.get("temporal_only_mae"),
                "test_metrics": test_metrics,
                "uncertainty": {
                    "nominal_coverage": gnn.get("uncertainty_nominal"),
                    "validation": {"actual_coverage": gnn.get("uncertainty_actual_coverage_validation")},
                    "test": {"actual_coverage": gnn.get("uncertainty_actual_coverage_test")},
                },
                "step14_demo": demo.get("step14_demo"),
                "step9": demo.get("step9"),
                "data_audit": demo.get("data_audit"),
            }
        return {"available": False, "reason": "forecaster_v1 artifacts are not present"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    test_metrics = manifest.get("metrics", {}).get("test", {})
    maes = [float(item["mae"]) for item in test_metrics.values() if item.get("mae") is not None]
    return {
        "available": True,
        "artifact": manifest.get("artifact", "forecaster_v1"),
        "model_version": manifest.get("architecture", {}).get("width"),
        "test_metrics": test_metrics,
        "model_mae": sum(maes) / len(maes) if maes else None,
        "temporal_baseline_mae": manifest.get("baseline_metrics", {}).get("test", {}).get("temporal_only", {}).get("mae"),
        "uncertainty": uncertainty,
        "training_cutoff_s": manifest.get("training_cutoff_s"),
        "split_policy": manifest.get("split_policy"),
    }


@app.get("/api/demo/drl")
def demo_drl() -> dict:
    """Canned DRL scope decision, honestly labeled as demo (no PPO trained)."""
    demo_path = PROJECT_ROOT / "artifacts" / "demo_evidence.json"
    if not demo_path.is_file():
        raise HTTPException(status_code=404, detail="demo_evidence.json is not present")
    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    drl = demo.get("drl_demo")
    if not drl:
        raise HTTPException(status_code=404, detail="drl_demo is not present")
    return drl


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {"scenarios": service.list_scenarios()}


@app.get("/api/scenarios/{scenario_id}")
def scenario_graph(scenario_id: str, closed: str = "") -> dict:
    closed_ids = [item.strip() for item in closed.split(",") if item.strip()]
    try:
        return service.graph(scenario_id, closed_ids)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/scenarios/{scenario_id}/closeable-edges")
def closeable_edges(scenario_id: str) -> dict:
    """Return the roads that can be used by the UI/clients as closures."""
    try:
        return {
            "scenario_id": scenario_id,
            "edges": service.closeable_edges(scenario_id),
        }
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/validate")
def validate(request: ValidateRequest) -> dict:
    try:
        return service.validate(
            request.scenario_id,
            request.plan,
            request.closed_edge_ids,
        )
    except Exception as exc:
        raise _bad_request(exc) from exc


@app.post("/api/solve")
def solve(request: SolveRequest) -> dict:
    try:
        return service.solve(
            _solve_options(request),
            scenario_id=request.scenario_id,
        )
    except Exception as exc:
        raise _bad_request(exc) from exc


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict:
    try:
        return service.compare(
            request.scenario_id,
            particles=request.particles,
            evaluations=request.evaluations,
            seed=request.seed,
            closed_edge_ids=request.closed_edge_ids,
            methods=request.methods,
        )
    except Exception as exc:
        raise _bad_request(exc) from exc


@app.post("/api/path")
def shortest_path(request: PathRequest) -> dict:
    try:
        return service.shortest_path(
            request.source,
            request.target,
            request.scenario_id,
            request.closed_edge_ids,
        )
    except Exception as exc:
        raise _bad_request(exc) from exc


@app.post("/api/loop")
def run_loop(request: LoopRequest) -> dict:
    try:
        state = loop.run(
            request.scenario_id,
            closed_edge_ids=request.closed_edge_ids,
            particles=request.particles,
            evaluations=request.evaluations,
            seed=request.seed,
            method=request.method,
            execute_sumo=request.execute_sumo,
        )
        return loop.as_dict(state)
    except Exception as exc:
        raise _bad_request(exc) from exc


@app.post("/api/sumo")
def run_sumo(request: SumoRequest) -> dict:
    try:
        return service.run_sumo(gui=request.gui)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/sumo/replay")
def replay_sumo(request: ReplayRequest | None = None) -> dict:
    """Run a headless SUMO episode and return FCD frames for the UI."""
    request = request or ReplayRequest()
    try:
        return service.replay_sumo(
            scenario_id=request.scenario_id,
            plan=request.plan,
            closed_edge_ids=request.closed_edge_ids,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")


@app.get("/")
def index() -> FileResponse:
    frontend_index = FRONTEND_DIST / "index.html"
    legacy_index = STATIC_DIR / "index.html"
    index_path = frontend_index if frontend_index.is_file() else legacy_index
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="UI index.html is missing")
    return FileResponse(index_path)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
