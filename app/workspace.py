"""Stateless workspace computations. Submitted snapshots never alter the catalog."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.contracts.scenario import Scenario
from src.platform.service import PlatformService, SolveOptions


class SnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    graph: dict
    method: Literal["qpso", "pso", "alns", "constructive", "milp"] = "qpso"
    particles: int = Field(default=12, ge=2, le=40)
    evaluations: int = Field(default=40, ge=4, le=400)
    seed: int = 7
    closed_edge_ids: list[str] = Field(default_factory=list, max_length=10000)
    traffic_factor: float = Field(default=1, ge=0.2, le=1)
    plan: dict[str, list[str]] | None = None
    source: str | None = None
    target: str | None = None
    methods: list[Literal["qpso", "pso", "alns", "constructive", "milp"]] = Field(
        default_factory=lambda: ["qpso", "pso", "alns"], min_length=1, max_length=5)


def snapshot_scenario(request: SnapshotRequest) -> Scenario:
    graph = request.graph
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    jobs = graph.get("requests", [])
    fleet = graph.get("fleet", [])
    if not (2 <= len(nodes) <= 5000 and 1 <= len(edges) <= 15000
            and 1 <= len(jobs) <= 200 and 1 <= len(fleet) <= 50):
        raise ValueError("Use 2–5000 nodes, 1–15000 roads, 1–200 deliveries and 1–50 vehicles.")
    edge_ids = {e["id"] for e in edges}
    if not set(request.closed_edge_ids) <= edge_ids:
        raise ValueError("A selected closure is not part of this network.")
    digest = hashlib.sha256(json.dumps(graph, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    return Scenario.model_validate({
        "schema_version": "1.2", "scenario_id": f"workspace-{digest}",
        "source_name": "workspace_snapshot", "source_checksum": digest,
        "units": {"distance": "m", "time": "s", "speed": "m/s", "demand": "load_units"},
        "coordinate_transform": {"scale": 1, "translation_x_m": 0, "translation_y_m": 0,
            "description": "OSM projected" if graph.get("geo", {}).get("available") else "Synthetic Cartesian coordinates in metres."},
        "graph_version": f"workspace:{digest}:{request.traffic_factor}",
        "nodes": [{"node_id": n["id"], "x_m": n["x"], "y_m": n["y"],
                   "kind": n.get("kind", "junction").split(".")[-1].lower(), "zone_id": n.get("zone", "Z1")} for n in nodes],
        "edges": [{"edge_id": e["id"], "parent_road_id": e["id"],
                   "from_node": e["from"], "to_node": e["to"], "length_m": e["length_m"],
                   "speed_limit_mps": e["speed_mps"] * request.traffic_factor,
                   "road_class": e.get("road_class", "local").split(".")[-1].lower(), "lane_count": 1,
                   "capacity_veh_per_hour": 600, "open_by_default": e.get("open", True)} for e in edges],
        "requests": [{"request_id": j["id"], "original_customer_id": j["id"],
                      "original_x": 0, "original_y": 0, "access_node_id": j["node"], "access_distance_m": 0,
                      "demand": j["demand"], "known_at_s": 0, "release_s": 0,
                      "earliest_service_start_s": j.get("earliest_s", 0),
                      "latest_service_start_s": j.get("latest_s", 28800),
                      "service_duration_s": j.get("service_s", 30)} for j in jobs],
        "fleet": [{"vehicle_id": v["id"], "capacity": v["capacity"],
                   "start_node_id": v["depot"], "depot_node_id": v["depot"]} for v in fleet],
        "seeds": {name: request.seed for name in ("scenario_seed", "road_seed", "traffic_seed",
                   "incident_seed", "window_seed", "optimizer_seed", "learning_seed")},
        "generator_version": "workspace-v1", "dataset_split": "demo", "configuration_version": "1",
    })


class SnapshotService(PlatformService):
    def __init__(self, scenario: Scenario):
        super().__init__()  # HERE feed handle; workspace never queries it, but keep base invariants.
        self.snapshot = scenario
        # Private, request-local cache. A user's snapshot cannot overwrite another user's solve.
        self._solve_cache = {}

    def solve(self, *args, **kwargs):
        result = super().solve(*args, **kwargs)
        # The core optimizer allows partial plans with an unserved-job penalty.
        # A workspace run is only complete when every configured delivery is served.
        if not result['evaluation']['all_served']:
            result['evaluation']['feasible'] = False
            if result.get('status') != 'unavailable':
                result['status'] = 'incomplete'
        return result

    def _scenario(self, scenario_id, closed_edge_ids=(), *, live_traffic=False,
                  strict=True, meta_out=None):
        closed = set(closed_edge_ids)
        scenario = self.snapshot.model_copy(update={"edges": tuple(
            e.model_copy(update={"open_by_default": False}) if e.edge_id in closed else e
            for e in self.snapshot.edges)})
        # Same HERE contract as the base class: an overlay request is strict by
        # default (fails loudly without a key) and reports its honest meta.
        if live_traffic:
            scenario, meta = self.here.apply(scenario, strict=strict)
            if meta_out is not None:
                meta_out.clear()
                meta_out.update(meta)
        return scenario

    def _budget_for(self, scenario_id, options):
        if len(self.snapshot.edges) <= self.LARGE_GRAPH_EDGES:
            return options, None
        return SolveOptions(method=options.method, particles=min(options.particles, 6),
            evaluations=min(options.evaluations, 24), seed=options.seed,
            closed_edge_ids=options.closed_edge_ids,
            live_traffic=options.live_traffic), "Large-network budget capped to 6 particles / 24 evaluations."


router = APIRouter(prefix="/api/workspace", tags=["Workspace"])


@router.post("/{operation}")
def compute(operation: Literal["solve", "compare", "path", "replay", "validate"], request: SnapshotRequest):
    try:
        scenario = snapshot_scenario(request)
        service = SnapshotService(scenario)
        if request.method == "milp" and len(scenario.requests) > 12:
            raise ValueError("MILP is available for at most 12 deliveries. Choose QPSO, PSO or ALNS.")
        if operation == "validate":
            return {"valid": True, "nodes": len(scenario.nodes), "edges": len(scenario.edges)}
        if operation == "path":
            if not request.source or not request.target:
                raise ValueError("Choose both an origin and a destination.")
            return service.shortest_path(request.source, request.target, scenario.scenario_id, request.closed_edge_ids)
        if operation == "replay":
            if request.plan is None:
                raise ValueError("Solve and validate the scenario before starting playback.")
            checked = service.validate(scenario.scenario_id, request.plan, request.closed_edge_ids)
            if not checked["validation"]["feasible"] or not checked["validation"]["all_served"]:
                raise ValueError("Playback requires a feasible route plan for this snapshot.")
            return service._mock_replay(scenario, scenario_id=scenario.scenario_id,
                plan=request.plan, closed_edge_ids=request.closed_edge_ids)
        if operation == "compare":
            methods = [m for m in request.methods if m != "milp" or len(scenario.requests) <= 12]
            return service.compare(scenario.scenario_id, particles=request.particles,
                evaluations=request.evaluations, seed=request.seed,
                closed_edge_ids=request.closed_edge_ids, methods=methods)
        return service.solve(SolveOptions(method=request.method, particles=request.particles,
            evaluations=request.evaluations, seed=request.seed,
            closed_edge_ids=tuple(request.closed_edge_ids)), scenario.scenario_id)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
