"""FastAPI dispatcher for the SIH26137 demo.

Run from the repository root:

    python -m app.server
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.platform.service import PlatformService, SolveOptions
from src.runtime.loop import DemoLoop

STATIC_DIR = Path(__file__).resolve().parent / "static"
service = PlatformService()
loop = DemoLoop(service)

app = FastAPI(
    title="Quanta SIH26137",
    description="Quantum-inspired traffic routing dispatcher",
    version="0.2.0",
)


class SolveRequest(BaseModel):
    scenario_id: str = "S3_BASE"
    method: str = "qpso"
    particles: int = Field(default=12, ge=2, le=40)
    evaluations: int = Field(default=40, ge=4, le=400)
    seed: int = 7
    closed_edge_ids: list[str] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    scenario_id: str = "S3_BASE"
    plan: dict[str, list[str]] | None = None
    closed_edge_ids: list[str] = Field(default_factory=list)


class PathRequest(BaseModel):
    scenario_id: str = "S3_BASE"
    source: str
    target: str
    closed_edge_ids: list[str] = Field(default_factory=list)


class CompareRequest(SolveRequest):
    methods: list[str] = Field(
        default_factory=lambda: ["constructive", "qpso", "pso", "alns"]
    )


class LoopRequest(SolveRequest):
    execute_sumo: bool = True


class SumoRequest(BaseModel):
    gui: bool = True


@app.get("/api/meta")
def meta() -> dict:
    return service.project_meta()


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {"scenarios": service.list_scenarios()}


@app.get("/api/scenarios/{scenario_id}")
def scenario_graph(scenario_id: str, closed: str = "") -> dict:
    closed_ids = [item for item in closed.split(",") if item]
    try:
        return service.graph(scenario_id, closed_ids)
    except KeyError as exc:
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
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/solve")
def solve(request: SolveRequest) -> dict:
    try:
        return service.solve(
            SolveOptions(
                method=request.method,
                particles=request.particles,
                evaluations=request.evaluations,
                seed=request.seed,
                closed_edge_ids=tuple(request.closed_edge_ids),
            ),
            scenario_id=request.scenario_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/loop")
def run_loop(request: LoopRequest) -> dict:
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


@app.post("/api/sumo")
def run_sumo(request: SumoRequest) -> dict:
    try:
        return service.run_sumo(gui=request.gui)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/sumo/replay")
def replay_sumo() -> dict:
    try:
        return service.replay_sumo()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
