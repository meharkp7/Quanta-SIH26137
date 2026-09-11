import json
import pytest
from src.sim.fixture import fixture_plan
from src.sim.sim_state import SimState
from tests.step6.test_step6_sumo import load_base_scenario, _sumo_available


def test_cargo_owner_and_service_duration():
    scenario = load_base_scenario()
    state = SimState(scenario)
    state.load_route_cargo(fixture_plan(scenario))
    assert [v.remaining_load for v in state.snapshot_vehicles()] == [7, 3]
    with pytest.raises(ValueError, match="own"):
        state.begin_service("V2", "J1", 12)
    state.begin_service("V1", "J1", 12)
    with pytest.raises(ValueError, match="duration"):
        state.complete_service("V1", "J1", 13)
    state.complete_service("V1", "J1", 22)
    assert state.snapshot_vehicles()[0].remaining_load == 5

@pytest.mark.sumo
@pytest.mark.skipif(not _sumo_available(), reason="SUMO required")
def test_fixture_physical_and_service_conservation(tmp_path):
    from src.sim.sumo_runner import run_episode
    scenario = load_base_scenario()
    result = run_episode(scenario, fixture_plan(scenario), tmp_path)
    assert result.delivered_count == 5
    assert result.failed_count == result.pending_count == result.teleport_events == 0
    assert result.service_begin_events == result.service_complete_events == 5
    assert all(v["state"] == "finished" and v["remaining_load"] == 0 for v in result.vehicle_summaries)
    requests = {r.request_id: r for r in scenario.requests}
    for record in result.request_summaries:
        request = requests[record["request_id"]]
        assert request.earliest_service_start_s <= record["service_start_s"] <= request.latest_service_start_s
        assert record["service_end_s"] - record["service_start_s"] >= request.service_duration_s
    progress = json.loads((tmp_path / "progress.json").read_text())
    at_closure = next(p for p in progress if p["sim_time_s"] == 50)
    clearing = {v["vehicle_id"] for v in at_closure["vehicles"] if v["edge_id"] == "E23"}
    assert clearing, "Exercise already-entered vehicles clearing the link"
    for step in progress:
        if step["sim_time_s"] > 50:
            occupants = {v["vehicle_id"] for v in step["vehicles"] if v["edge_id"] == "E23"}
            assert occupants <= clearing, "New vehicle entered a closed edge"
            clearing &= occupants
    assert not clearing
    assert {vid for p in progress for vid in p["arrived_vehicle_ids"]} == {"V1", "V2", "bg_0", "bg_1"}
