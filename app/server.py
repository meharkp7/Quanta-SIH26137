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
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.platform.service import PlatformService, SolveOptions
from src.runtime.loop import DemoLoop
from app.auth import config, verify_company
from app.workspace import router as workspace_router

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

app.include_router(workspace_router)


@app.middleware("http")
async def company_access(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and request.method != "OPTIONS":
        # Public reads contain only catalog/demo data, never a company's saved records.
        public_read = request.method == "GET" and path.startswith((
            "/api/health", "/api/meta", "/api/scenarios", "/api/evidence", "/api/demo/"))
        company = request.headers.get("x-company-id", "")
        token = request.headers.get("authorization", "")
        demo_allowed = config("QUANTA_ALLOW_DEMO", "true").lower() == "true"
        demo_compute = path in ("/api/solve", "/api/compare", "/api/path", "/api/validate", "/api/sumo/replay") or path.startswith("/api/workspace/")
        configured = bool(config("SUPABASE_URL"))
        try:
            if token or company:
                role = await run_in_threadpool(verify_company, token, company)
                if request.method == "POST" and role == "viewer":
                    raise HTTPException(403, "Viewers can inspect saved results. A dispatcher or owner can run computations.")
                if ("/train" in path or path in ("/api/sumo", "/api/loop")) and role != "owner":
                    raise HTTPException(403, "Only company owners may use this operation.")
            elif configured and not public_read:
                if not (demo_allowed and demo_compute and request.headers.get("x-quanta-demo") == "true"):
                    raise HTTPException(401, "Sign in or enter the demo workspace.")
            elif not configured and path.startswith("/api/workspace/") and not demo_allowed:
                raise HTTPException(401, "Company authentication is required.")
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    return await call_next(request)

# CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=config("QUANTA_ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8765").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


class ForecasterTrainRequest(BaseModel):
    epochs: int = Field(default=60, ge=1, le=500)
    width: int = Field(default=32, ge=8, le=256)
    lr: float = Field(default=2e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    heads: int = Field(default=4, ge=1, le=16)
    layers: int = Field(default=2, ge=1, le=8)
    batch_size: int = Field(default=2, ge=1, le=16)
    seed: int = 26137
    device: str = Field(default="cpu")
    scheduler: str = Field(default="cosine")
    patience: int = Field(default=25, ge=0, le=200)
    final_eval: bool = Field(
        default=True,
        description="False skips the memory-heavy in-training finale; "
        "finalize with scripts/finalize_forecaster.py instead.",
    )
    stream: bool = Field(
        default=True,
        description="Stream one episode at a time (O(episode) RAM, no padding "
        "waste). Required for thousand-episode corpora.",
    )
    stride: int = Field(
        default=1, ge=1, le=9,
        description="Train on every k-th window per episode (stream only).",
    )


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
    candidates = [
        PROJECT_ROOT / "artifacts" / "forecaster_v2",
        PROJECT_ROOT / "artifacts" / "step13_forecaster_v1",
    ]
    artifact_dir = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "manifest.json").is_file()
            and (candidate / "uncertainty.json").is_file()
        ),
        candidates[1],
    )
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


@app.get("/api/demo/story")
def demo_story() -> dict:
    """Scripted closed-loop demo narrative using only existing capabilities.

    Each step carries an optional ``action`` (``{http_method, endpoint,
    params}``) with parameters verified against the live backend at small
    demo budgets. The frontend executes actions through the normal
    solve/replay/compare/evidence state — never a parallel universe.
    """
    budget = {"particles": 6, "evaluations": 12, "seed": 7}
    steps = [
        {
            "id": "baseline",
            "title": "Solve a clean baseline",
            "caption": (
                "QPSO searches the open S3_BASE network at a small demo "
                "budget. The result lands in the normal route view with an "
                "independent validator report — note the objective value, it "
                "is the 'before' for step 4."
            ),
            "action": {
                "http_method": "POST",
                "endpoint": "/api/solve",
                "params": {
                    "scenario_id": "S3_BASE",
                    "method": "qpso",
                    **budget,
                    "closed_edge_ids": [],
                },
            },
        },
        {
            "id": "replay-incident",
            "title": "Watch E23 close at t=50s",
            "caption": (
                "The SUMO replay enforces the fixture incident: E23 closes at "
                "t=50s. Clearing rule — vehicles already on the link clear "
                "it, new entry is forbidden. Scrub the timeline and watch "
                "the per-frame closure state flip."
            ),
            "action": {
                "http_method": "POST",
                "endpoint": "/api/sumo/replay",
                "params": {
                    "scenario_id": "S3_BASE",
                    "plan": None,
                    "closed_edge_ids": [],
                },
            },
        },
        {
            "id": "compare",
            "title": "Compare optimizers",
            "caption": (
                "Same network, same small budget, three methods "
                "(QPSO / PSO / ALNS). The compare strip shows feasibility, "
                "objective, and latency side by side — QPSO is the default "
                "because it wins here, not by declaration."
            ),
            "action": {
                "http_method": "POST",
                "endpoint": "/api/compare",
                "params": {
                    "scenario_id": "S3_BASE",
                    "methods": ["qpso", "pso", "alns"],
                    **budget,
                    "closed_edge_ids": [],
                },
            },
        },
        {
            "id": "detour",
            "title": "Close E12 and re-solve",
            "caption": (
                "E12 (N1→N2) is now a hard closure. The solver must detour "
                "around it — the compare strip shows the cost delta against "
                "the step-1 baseline, and the map overlays the previous "
                "route in cyan dashed for a before/after view."
            ),
            "action": {
                "http_method": "POST",
                "endpoint": "/api/solve",
                "params": {
                    "scenario_id": "S3_BASE",
                    "method": "qpso",
                    **budget,
                    "closed_edge_ids": ["E12"],
                },
            },
        },
        {
            "id": "replay-closed",
            "title": "Replay with incident state",
            "caption": (
                "Replay the incident world and scrub the timeline: each "
                "frame carries its own closure list, the incident banner "
                "names the enforced schedule, and closure markers track the "
                "scrubber. (SUMO replays the open network here — closing "
                "E12 or E23 outright leaves no feasible episode, which the "
                "validator reports instead of faking one.)"
            ),
            "action": {
                "http_method": "POST",
                "endpoint": "/api/sumo/replay",
                "params": {
                    "scenario_id": "S3_BASE",
                    "plan": None,
                    "closed_edge_ids": [],
                },
            },
        },
        {
            "id": "evidence",
            "title": "Evidence checkpoint",
            "caption": (
                "Close the loop with real numbers: the evidence tab shows "
                "measured forecast error, the honest baseline comparison, "
                "and uncertainty coverage from saved artifacts — DEMO "
                "labeled where v2 training is still pending."
            ),
            "action": {
                "http_method": "GET",
                "endpoint": "/api/evidence",
                "params": {},
            },
        },
    ]
    return {"scenario_id": "S3_BASE", "steps": steps}


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


@app.get("/api/models/forecaster/status")
def forecaster_status() -> dict:
    """Training/corpus status for the joint GNN-Transformer forecaster."""
    try:
        return service.forecaster_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/models/forecaster/train")
def forecaster_train(request: ForecasterTrainRequest | None = None) -> dict:
    """Launch joint GNN-Transformer training on artifacts/corpus_v2.

    The Step 13 model is one joint network (edge-aware GNN spatial encoder
    + temporal Transformer); this endpoint trains it on the real corpus_v2
    episodes in a background process and returns immediately. Poll
    ``GET /api/models/forecaster/status`` for progress; results land in
    ``artifacts/forecaster_v2`` where ``GET /api/evidence`` picks them up.
    """
    request = request or ForecasterTrainRequest()
    try:
        return service.start_forecaster_training(
            epochs=request.epochs,
            width=request.width,
            lr=request.lr,
            weight_decay=request.weight_decay,
            dropout=request.dropout,
            heads=request.heads,
            layers=request.layers,
            batch_size=request.batch_size,
            seed=request.seed,
            device=request.device,
            scheduler=request.scheduler,
            patience=request.patience,
            final_eval=request.final_eval,
            stream=request.stream,
            stride=request.stride,
        )
    except ValueError as exc:
        raise _bad_request(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ──────────────────────────────────────────────────────────────────────
# Training Dashboard (read-only, non-blocking, with 5s in-memory cache)
# ──────────────────────────────────────────────────────────────────────

class _TrainingCache:
    """Tiny in-memory TTL cache for training artifacts."""

    def __init__(self, ttl_seconds: float = 5.0):
        self._ttl = ttl_seconds
        self._data: dict[str, tuple[float, dict]] = {}

    def get(self, key: str) -> dict | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self._ttl:
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: dict) -> None:
        self._data[key] = (time.time(), value)

    def clear(self) -> None:
        self._data.clear()


_training_cache = _TrainingCache(ttl_seconds=5.0)

# Candidate training artifact directories (in priority order)
_TRAINING_ARTIFACT_DIRS = [
    PROJECT_ROOT / "artifacts" / "forecaster_v2",
    PROJECT_ROOT / "artifacts" / "forecaster_mps_pilot",
    PROJECT_ROOT / "artifacts" / "step13_forecaster_v1",
]

# Max points to return in the status endpoint for loss curves
_MAX_CURVE_POINTS = 100


def _find_latest_training_dir() -> Path | None:
    """Return the most recently modified training artifact directory."""
    for d in _TRAINING_ARTIFACT_DIRS:
        if d.is_dir() and (d / "training_curve.json").is_file():
            return d
    return None


def _read_json_cached(cache_key: str, path: Path) -> dict | list | None:
    """Read JSON from disk with TTL caching."""
    cached = _training_cache.get(cache_key)
    if cached is not None:
        return cached
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _training_cache.set(cache_key, data)
        return data
    except Exception:
        return None


def _read_training_curve(dir_path: Path) -> list[dict] | None:
    return _read_json_cached(f"curve:{dir_path}", dir_path / "training_curve.json")


def _read_training_status(dir_path: Path) -> dict | None:
    return _read_json_cached(f"status:{dir_path}", dir_path / "training_status.json")


def _read_manifest(dir_path: Path) -> dict | None:
    return _read_json_cached(f"manifest:{dir_path}", dir_path / "manifest.json")


def _estimate_time_remaining(curve: list[dict], total_epochs: int, started_at: float | None) -> dict | None:
    """Estimate remaining time based on epochs completed so far."""
    if not curve or started_at is None:
        return None
    epochs_done = len(curve)
    if epochs_done == 0:
        return None
    elapsed = time.time() - started_at
    if elapsed <= 0:
        return None
    secs_per_epoch = elapsed / epochs_done
    remaining_epochs = max(0, total_epochs - epochs_done)
    est_remaining_secs = secs_per_epoch * remaining_epochs
    return {
        "epochs_completed": epochs_done,
        "epochs_remaining": remaining_epochs,
        "elapsed_seconds": round(elapsed, 1),
        "estimated_remaining_seconds": round(est_remaining_secs, 1),
        "estimated_completion_timestamp": round(time.time() + est_remaining_secs, 1),
        "seconds_per_epoch": round(secs_per_epoch, 2),
    }


def _compute_early_stopping(curve: list[dict], patience: int) -> dict:
    """Compute early stopping status from curve."""
    if not curve or patience <= 0:
        return {"enabled": bool(patience), "patience": patience, "epochs_since_improvement": 0, "stopped_early": False}
    # Find best validation MAE and its epoch
    best_val = float("inf")
    best_epoch = 0
    for row in curve:
        val = row.get("validation_speed_mae")
        if val is not None and val < best_val:
            best_val = val
            best_epoch = row.get("epoch", 0)
    epochs_since = len(curve) - best_epoch
    stopped_early = epochs_since >= patience
    return {
        "enabled": True,
        "patience": patience,
        "best_epoch": best_epoch,
        "best_validation_mae": best_val if best_val != float("inf") else None,
        "epochs_since_improvement": epochs_since,
        "stopped_early": stopped_early,
    }


def _build_status_payload() -> dict:
    """Build the rich training status payload."""
    # Find the active or latest training directory
    training_dir = _find_latest_training_dir()
    if training_dir is None:
        return {
            "state": "no_active_training",
            "message": "No training runs found in artifacts/",
            "active_training": False,
        }

    curve = _read_training_curve(training_dir)
    status = _read_training_status(training_dir)
    manifest = _read_manifest(training_dir)

    if curve is None:
        return {
            "state": "error",
            "message": f"training_curve.json not found in {training_dir}",
            "active_training": False,
        }

    # Determine if training is currently running
    is_running = False
    started_at = None
    config = {}
    pid = None
    if status:
        is_running = status.get("state") == "running" and status.get("alive", False)
        started_at = status.get("started_at")
        config = status.get("config", {})
        pid = status.get("pid")

    # Also check manifest for config if status doesn't have it
    if not config and manifest:
        config = manifest.get("hyperparameters", {})

    total_epochs = config.get("epochs", 350)
    patience = config.get("patience", 30)
    device = config.get("device", "mps")

    # Last N points for curves (for sparklines)
    curve_points = curve[-_MAX_CURVE_POINTS:] if len(curve) > _MAX_CURVE_POINTS else curve
    epochs = [row["epoch"] for row in curve_points]
    train_loss = [row["train_loss"] for row in curve_points]
    train_speed_loss = [row["train_speed_loss"] for row in curve_points]
    train_traversal_loss = [row["train_traversal_loss"] for row in curve_points]
    val_mae = [row["validation_speed_mae"] for row in curve_points]
    lrs = [row["lr"] for row in curve_points]

    # Best validation MAE and epoch
    best_val = float("inf")
    best_epoch = 0
    for row in curve:
        val = row.get("validation_speed_mae")
        if val is not None and val < best_val:
            best_val = val
            best_epoch = row.get("epoch", 0)

    early_stopping = _compute_early_stopping(curve, patience)
    eta = _estimate_time_remaining(curve, total_epochs, started_at)

    # Model config summary
    model_config = {
        "width": config.get("width", manifest.get("architecture", {}).get("width") if manifest else 64),
        "heads": config.get("heads", manifest.get("architecture", {}).get("heads") if manifest else 4),
        "layers": config.get("layers", manifest.get("architecture", {}).get("layers") if manifest else 2),
        "dropout": config.get("dropout", manifest.get("architecture", {}).get("dropout") if manifest else 0.1),
        "batch_size": config.get("batch_size", 4),
        "lr": config.get("lr", 0.001),
        "weight_decay": config.get("weight_decay", 0.0001),
        "scheduler": config.get("scheduler", "cosine"),
        "device": device,
    }

    return {
        "state": "running" if is_running else "completed",
        "active_training": is_running,
        "training_dir": str(training_dir),
        "pid": pid,
        "current_epoch": len(curve),
        "total_epochs": total_epochs,
        "progress_pct": round((len(curve) / total_epochs) * 100, 1) if total_epochs else 0,
        "best_val_mae": best_val if best_val != float("inf") else None,
        "best_epoch": best_epoch,
        "early_stopping": early_stopping,
        "eta": eta,
        "device": device,
        "model_config": model_config,
        "curves": {
            "epochs": epochs,
            "train_loss": train_loss,
            "train_speed_loss": train_speed_loss,
            "train_traversal_loss": train_traversal_loss,
            "val_mae": val_mae,
            "learning_rates": lrs,
        },
        "last_update": time.time(),
    }


@app.get("/api/training/status")
def training_status() -> dict:
    """Live-ish training progress (cached for 5s to avoid disk thrashing)."""
    return _build_status_payload()


@app.get("/api/training/history")
def training_history() -> dict:
    """Full training_curve.json for charting (cached for 5s)."""
    training_dir = _find_latest_training_dir()
    if training_dir is None:
        return {
            "state": "no_active_training",
            "message": "No training runs found in artifacts/",
            "curve": [],
        }
    curve = _read_training_curve(training_dir)
    if curve is None:
        raise HTTPException(status_code=404, detail="training_curve.json not found")
    return {
        "state": "ok",
        "training_dir": str(training_dir),
        "curve": curve,
    }


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


@app.get("/{path:path}", include_in_schema=False)
def frontend_route(path: str) -> FileResponse:
    if path.startswith(("api/", "assets/", "static/")) or "." in path.rsplit("/", 1)[-1]:
        raise HTTPException(404, "Not found")
    return index()


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
