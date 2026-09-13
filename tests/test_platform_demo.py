from pathlib import Path

from src.platform.service import PlatformService, SolveOptions
from src.runtime.loop import DemoLoop
from src.sim.fcd_replay import parse_fcd


def test_graph_and_reference_plan_are_feasible():
    service = PlatformService()
    graph = service.graph("S3_BASE")
    assert graph["nodes"]
    assert graph["edges"]
    checked = service.validate("S3_BASE", graph["reference_plan"])
    assert checked["validation"]["feasible"] is True


def test_overloaded_plan_is_rejected():
    service = PlatformService()
    checked = service.validate(
        "S3_BASE",
        {"V1": ["J1", "J2", "J3", "J4", "J5"], "V2": []},
    )
    assert checked["validation"]["feasible"] is False
    assert checked["validation"]["capacity"] is False


def test_qpso_returns_routes_on_tiny_budget():
    service = PlatformService()
    result = service.solve(
        SolveOptions(method="qpso", particles=4, evaluations=12, seed=7),
        scenario_id="S3_BASE",
    )
    assert result["method"] == "QPSO"
    assert "evaluation" in result
    assert result["trace"]["best"]


def test_dijkstra_path_is_exact_and_legal():
    service = PlatformService()
    path = service.shortest_path("N0", "N2", "S3_BASE")
    assert path["exact"] is True
    assert path["feasible"] is True
    assert path["edge_ids"]


def test_demo_loop_records_baseline_forecast():
    state = DemoLoop().run(
        "S3_BASE",
        particles=4,
        evaluations=12,
        seed=3,
        execute_sumo=False,
    )
    assert state.forecast_mode == "persistence"
    assert state.scope_action in {"KEEP", "GLOBAL"}
    assert state.solve["evaluation"]


def test_fcd_replay_reads_existing_sumo_trace():
    fcd = Path("artifacts/step6_headless/sumo_output/fcd.xml")
    if not fcd.is_file():
        return
    frames = parse_fcd(fcd)
    vans = {veh["id"] for frame in frames for veh in frame["vehicles"] if veh["kind"] == "delivery"}
    assert "V1" in vans and "V2" in vans
    closed = next(frame for frame in frames if frame["t"] >= 50)
    assert "E23" in closed["closed"]
