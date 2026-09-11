"""Run the independently evaluated five-job SUMO fixture."""
from pathlib import Path
import argparse
import json
from src.contracts.scenario import Scenario
from src.contracts.routing import RoutePlan, VehicleRoute, StopLeg
from src.routing.validator import evaluate_scenario
from .sumo_runner import run_episode

ROOT = Path(__file__).resolve().parents[2]

def fixture_plan(scenario):
    orders = {"V1": ["J1", "J2", "J3"], "V2": ["J4", "J5"]}
    evaluation = evaluate_scenario(scenario, orders)
    if not evaluation.feasible:
        raise ValueError("Fixture reference is infeasible")
    edges = {e.edge_id: e for e in scenario.edges}
    routes = []
    for vid, result in evaluation.vehicles.items():
        legs = []
        departure = 0.0
        for index, leg in enumerate(result.legs):
            legs.append(StopLeg(from_stop_id=leg.from_node, to_stop_id=leg.to_node,
                physical_edge_ids=leg.edge_ids, travel_time_s=leg.travel_time_s,
                distance_m=sum(edges[e].length_m for e in leg.edge_ids),
                congestion_exposure=0, departure_time_s=departure,
                arrival_time_s=departure + leg.travel_time_s))
            if index < len(result.customer_order):
                departure = result.service_ends_s[result.customer_order[index]]
        routes.append(VehicleRoute(vehicle_id=vid, customer_order=result.customer_order,
            legs=tuple(legs), expected_departure_s=0, expected_return_s=result.elapsed_time_s,
            expected_driving_time_s=result.driving_time_s,
            expected_waiting_time_s=result.waiting_time_s,
            expected_service_time_s=result.service_time_s,
            served_request_ids=(), frozen_prefix_edge_ids=()))
    distance = sum(leg.distance_m for route in routes for leg in route.legs)
    elapsed = sum(route.expected_return_s for route in routes)
    return RoutePlan(scenario_id=scenario.scenario_id, state_version="initial",
        route_version="fixture-v1", generated_at_s=0, vehicle_routes=tuple(routes),
        objective_value=elapsed, objective_time=elapsed, objective_distance=distance,
        objective_congestion=0, route_change_penalty=0,
        changed_vehicle_ids=tuple(orders), generated_by="independent_fixture_evaluator")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "step6_verified")
    parser.add_argument("--gui", action="store_true", help="Display actual SUMO movement")
    args = parser.parse_args()
    scenario = Scenario.model_validate_json((ROOT / "fixtures/step3/base/scenario.json").read_text())
    result = run_episode(scenario, fixture_plan(scenario), args.output, gui=args.gui)
    print(json.dumps(result.to_dict(), indent=2))
    if result.delivered_count != 5 or result.teleport_events or any(v["state"] != "finished" for v in result.vehicle_summaries):
        raise SystemExit("Fixture acceptance failed")

if __name__ == "__main__":
    main()
