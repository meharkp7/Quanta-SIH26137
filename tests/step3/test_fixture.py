import json
from pathlib import Path
import pytest
from src.routing.validator import evaluate_scenario, shortest_directed_path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures" / "step3"

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def scenario(case):
    return load_json(FIXTURE / "cases" / case / "scenario.json")

def expected(case):
    return load_json(FIXTURE / "cases" / case / "expected.json")

REFERENCE_PLAN = {"V1": ["J1", "J2", "J3"], "V2": ["J4", "J5"]}

def test_reference_solution():
    r = evaluate_scenario(scenario("feasible_reference"), REFERENCE_PLAN)
    assert r.feasible
    assert r.capacity_feasible and r.connectivity_feasible and r.time_window_feasible and r.all_requests_served
    v1, v2 = r.vehicles["V1"], r.vehicles["V2"]
    assert (v1.load, v1.driving_time_s, v1.waiting_time_s, v1.service_time_s, v1.elapsed_time_s) == pytest.approx((7,70,10,35,115))
    assert v1.service_starts_s == {"J1":10.0,"J2":35.0,"J3":60.0}
    assert (v2.load, v2.driving_time_s, v2.waiting_time_s, v2.service_time_s, v2.elapsed_time_s) == pytest.approx((3,70,40,20,130))
    assert v2.service_starts_s == {"J4":80.0,"J5":100.0}

def test_early_arrival_waiting():
    r = evaluate_scenario(scenario("early_arrival_wait"), REFERENCE_PLAN)
    assert r.feasible
    assert r.vehicles["V1"].service_starts_s["J2"] == pytest.approx(35)
    assert r.vehicles["V1"].service_starts_s["J3"] == pytest.approx(60)
    assert r.vehicles["V2"].service_starts_s["J4"] == pytest.approx(80)
    assert r.vehicles["V1"].waiting_time_s == pytest.approx(10)
    assert r.vehicles["V2"].waiting_time_s == pytest.approx(40)

def test_excess_load_only_fails_capacity():
    plan = {"V1":["J1","J2","J3","J4"], "V2":["J5"]}
    r = evaluate_scenario(scenario("excess_load"), plan)
    assert not r.feasible
    assert not r.capacity_feasible
    assert r.connectivity_feasible
    assert r.time_window_feasible
    assert r.vehicles["V1"].load == pytest.approx(9)

def test_missed_window_fails_time_window():
    r = evaluate_scenario(scenario("missed_window"), REFERENCE_PLAN)
    assert not r.feasible
    assert r.capacity_feasible and r.connectivity_feasible
    assert not r.time_window_feasible
    assert r.vehicles["V1"].service_starts_s["J3"] == pytest.approx(75)

def test_reverse_traversal_is_not_inferred():
    r = shortest_directed_path(scenario("wrong_way"), "N7", "N2")
    assert r is None

def test_closure_selects_detour():
    sc = scenario("closure_with_detour")
    normal = shortest_directed_path(sc, "N1", "N2")
    detour = shortest_directed_path(sc, "N1", "N2", ["E12"])
    assert normal == (("E12",), pytest.approx(10.0))
    assert detour == (("E16","E62"), pytest.approx(30.0))

def test_closure_route_remains_feasible():
    sc = scenario("closure_with_detour")
    r = evaluate_scenario(sc, REFERENCE_PLAN, closed_edge_ids=["E12"])
    assert r.feasible
    assert r.vehicles["V1"].service_starts_s["J2"] == pytest.approx(50)
    assert r.vehicles["V1"].service_starts_s["J3"] == pytest.approx(70)
    assert r.vehicles["V1"].legs[1].edge_ids == ("E16","E62")

def test_no_invalid_request_windows_in_base():
    sc = scenario("feasible_reference")
    assert all(j["earliest_start_s"] <= j["latest_start_s"] for j in sc["requests"])
