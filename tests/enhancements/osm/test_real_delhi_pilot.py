"""Regression tests for src.data.real_delhi_pilot.

These tests stub the SUMO/TraCI boundary (this environment has no SUMO
binaries) but exercise every real piece of orchestration logic this
module owns: map generation via ``generate_osm_scenario``, the
physical-distinctness guard, real VRP route-plan solving via
``InitialSolutionBuilder``, and manifest/coverage writing through the
actual production ``causal_episodes`` helpers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.contracts.core_types import NodeKind
from src.contracts.scenario import RoadEdge, RoadNode
from src.data.causal_episodes import CausalEpisodeResult
from src.data.osm_demand_generator import OSMDemandConfig
from src.data.osm_ingestion import OSMNetwork
from src.data.real_delhi_pilot import (
    RealDelhiMapSpec,
    RealDelhiPilotError,
    generate_real_delhi_pilot,
)
import src.data.real_delhi_pilot as real_delhi_pilot


def _node(node_id: str, x: float, y: float) -> RoadNode:
    return RoadNode(node_id=node_id, x_m=x, y_m=y, kind=NodeKind.JUNCTION, zone_id="osm")


def _edge(edge_id: str, u: str, v: str) -> RoadEdge:
    return RoadEdge(
        edge_id=edge_id,
        parent_road_id=f"r:{u}-{v}",
        from_node=u,
        to_node=v,
        length_m=100.0,
        road_class="local",
        speed_limit_mps=10.0,
        lane_count=1,
        capacity_veh_per_hour=900.0,
        provenance="openstreetmap",
    )


def _grid(offset: int, size: int = 4) -> OSMNetwork:
    def nid(row: int, col: int) -> str:
        return f"osm:n:{offset}:{row}-{col}"

    nodes = [
        _node(nid(row, col), col * 100.0 + offset * 10_000, row * 100.0)
        for row in range(size)
        for col in range(size)
    ]
    edges = []
    for row in range(size):
        for col in range(size):
            if col + 1 < size:
                edges.append(_edge(f"e:{offset}:{row}-{col}:R", nid(row, col), nid(row, col + 1)))
                edges.append(_edge(f"e:{offset}:{row}-{col}:L", nid(row, col + 1), nid(row, col)))
            if row + 1 < size:
                edges.append(_edge(f"e:{offset}:{row}-{col}:D", nid(row, col), nid(row + 1, col)))
                edges.append(_edge(f"e:{offset}:{row}-{col}:U", nid(row + 1, col), nid(row, col)))
    return OSMNetwork(nodes=tuple(nodes), edges=tuple(edges), source_crs="EPSG:32643", projected=True)


@pytest.fixture(autouse=True)
def _stub_sumo_boundary(monkeypatch):
    """Stub only the real SUMO/TraCI call; every other function used by
    generate_real_delhi_pilot is the real production implementation."""

    networks = {i: _grid(i) for i in range(4)}

    def fake_load_graphml(path, *, config=None):
        index = int(path.split(":")[1])
        return networks[index]

    def fake_run_sumo_causal_episode(
        *, scenario, output_dir, config, episode_id, split, events, traffic_only, route_plan=None, **kwargs
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "observations.csv").write_text("timestamp_s,edge_id,speed_mps\n")
        (output_dir / "trajectories.csv").write_text("vehicle_id,timestamp_s,edge_id\n")
        if not traffic_only:
            assert route_plan is not None, "non-traffic-only episode must receive a route_plan"
            assert hasattr(route_plan, "vehicle_routes")
        return output_dir

    def fake_finalize(scenario, folder, eid, split, config):
        (Path(folder) / "labels.jsonl").write_text("")
        return CausalEpisodeResult(
            episode_dir=Path(folder),
            episode_id=eid,
            split=split,
            wall_clock_s=0.0,
            disk_bytes=0,
            label_coverage={},
        )

    def fake_audit(folder):
        return {"ok": True, "episode": str(folder)}

    monkeypatch.setattr(real_delhi_pilot, "load_graphml", fake_load_graphml)
    monkeypatch.setattr(real_delhi_pilot, "run_sumo_causal_episode", fake_run_sumo_causal_episode)
    monkeypatch.setattr(real_delhi_pilot, "_finalize_existing_episode", fake_finalize)
    monkeypatch.setattr(real_delhi_pilot, "audit_episode", fake_audit)


def _map_specs(*, duplicate: bool = False) -> dict[int, RealDelhiMapSpec]:
    return {
        i: RealDelhiMapSpec(
            graphml_path=f"fake:{0 if duplicate and i == 1 else i}:.graphml",
            demand=OSMDemandConfig(customer_count=6, fleet_size=2, vehicle_capacity=20.0, seed=i),
        )
        for i in range(4)
    }


def test_full_pilot_runs_end_to_end_with_real_route_plans(tmp_path):
    coverage = generate_real_delhi_pilot(
        tmp_path / "pilot", _map_specs(), duration_s=1200, traffic_only=False
    )
    assert coverage["episodes"] == 6
    manifest = (tmp_path / "pilot" / "split_manifest.json").is_file()
    assert manifest
    assert (tmp_path / "pilot" / "coverage.json").is_file()
    assert (tmp_path / "pilot" / "audit.json").is_file()


def test_duplicate_physical_map_is_rejected(tmp_path):
    with pytest.raises(RealDelhiPilotError, match="not physically distinct"):
        generate_real_delhi_pilot(
            tmp_path / "pilot", _map_specs(duplicate=True), duration_s=1200
        )


def test_missing_map_slot_is_rejected(tmp_path):
    specs = _map_specs()
    del specs[3]
    with pytest.raises(RealDelhiPilotError, match="exactly slots"):
        generate_real_delhi_pilot(tmp_path / "pilot", specs, duration_s=1200)


def test_infeasible_fleet_is_rejected_before_any_sumo_call(tmp_path):
    specs = _map_specs()
    specs[0] = RealDelhiMapSpec(
        graphml_path="fake:0:.graphml",
        demand=OSMDemandConfig(
            customer_count=6,
            fleet_size=1,
            vehicle_capacity=1.0,
            demand_min=5.0,
            demand_max=5.0,
            seed=0,
        ),
    )
    with pytest.raises(Exception):
        generate_real_delhi_pilot(tmp_path / "pilot", specs, duration_s=1200, traffic_only=False)


def test_traffic_only_mode_does_not_require_a_route_plan(tmp_path):
    coverage = generate_real_delhi_pilot(
        tmp_path / "pilot", _map_specs(), duration_s=1200, traffic_only=True
    )
    assert coverage["episodes"] == 6