"""Turn internal objects into JSON the demo UI can render."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.contracts.scenario import Scenario
from src.routing.route_evaluator import RoutePlanEvaluation
from src.routing.validator import EvaluationResult

# Real-Delhi scenarios carry native OSM projected coordinates (UTM 43N,
# EPSG:32643 — see the graphml `crs` graph attribute). The fixture scenarios
# use small synthetic Cartesian coordinates. Convert once per payload so the
# UI can render real geography on a tile map.
_SOURCE_CRS = "EPSG:32643"
_TARGET_CRS = "EPSG:4326"

try:
    from pyproj import Transformer as _Transformer

    _LATLON = _Transformer.from_crs(_SOURCE_CRS, _TARGET_CRS, always_xy=True)
except Exception:
    _LATLON = None


def is_geo_scenario(scenario: Scenario) -> bool:
    """True when node coordinates are real projected metres (UTM 43N)."""
    try:
        description = (scenario.coordinate_transform.description or "").lower()
    except Exception:
        description = ""
    if "osm projected" in description:
        return True
    # Fallback: fixtures live near the origin; UTM 43N eastings/northings
    # for Delhi are ~1e5..9e5 / ~1e6..9e6.
    try:
        return all(
            1e5 <= float(node.x_m) <= 9e5 and 1e6 <= float(node.y_m) <= 9e6
            for node in scenario.nodes
        )
    except Exception:
        return False


def to_latlon(x_m: float, y_m: float) -> tuple[float, float] | None:
    """Convert projected metres to (lat, lon); None if unavailable."""
    if _LATLON is None:
        return None
    try:
        lon, lat = _LATLON.transform(float(x_m), float(y_m))
    except Exception:
        return None
    return float(lat), float(lon)


def scenario_payload(scenario: Scenario, *, closed_edge_ids: Sequence[str] = ()) -> dict:
    closed = set(closed_edge_ids)
    geo = is_geo_scenario(scenario) and _LATLON is not None
    nodes = []
    for node in scenario.nodes:
        entry: dict = {
            "id": node.node_id,
            "x": node.x_m,
            "y": node.y_m,
            "kind": str(node.kind),
            "zone": node.zone_id,
        }
        if geo:
            converted = to_latlon(node.x_m, node.y_m)
            if converted is not None:
                entry["lat"], entry["lon"] = converted
            else:
                geo = False
        nodes.append(entry)
    return {
        "scenario_id": scenario.scenario_id,
        "graph_version": scenario.graph_version,
        "geo": {
            "available": bool(geo),
            "crs": _TARGET_CRS,
            "source_crs": _SOURCE_CRS,
        },
        "units": scenario.units.model_dump(),
        "nodes": nodes,
        "edges": [
            {
                "id": edge.edge_id,
                "from": edge.from_node,
                "to": edge.to_node,
                "length_m": edge.length_m,
                "speed_mps": edge.speed_limit_mps,
                "road_class": str(edge.road_class),
                "open": bool(edge.open_by_default) and edge.edge_id not in closed,
                "free_flow_s": edge.free_flow_time_s,
            }
            for edge in scenario.edges
        ],
        "requests": [
            {
                "id": job.request_id,
                "node": job.access_node_id,
                "demand": job.demand,
                "earliest_s": job.earliest_service_start_s,
                "latest_s": job.latest_service_start_s,
                "service_s": job.service_duration_s,
            }
            for job in scenario.requests
        ],
        "fleet": [
            {
                "id": vehicle.vehicle_id,
                "capacity": vehicle.capacity,
                "depot": vehicle.depot_node_id,
            }
            for vehicle in scenario.fleet
        ],
    }


def validator_payload(result: EvaluationResult) -> dict:
    return {
        "feasible": result.feasible,
        "capacity": result.capacity_feasible,
        "windows": result.time_window_feasible,
        "connectivity": result.connectivity_feasible,
        "all_served": result.all_requests_served,
        "unserved": list(result.unserved_request_ids),
        "window_violations": [
            {
                "vehicle": item[0],
                "job": item[1],
                "start_s": item[2],
                "latest_s": item[3],
            }
            for item in result.window_violations
        ],
        "disconnected": [
            {
                "vehicle": item[0],
                "from": item[1],
                "to": item[2],
                "job": item[3],
            }
            for item in result.disconnected_legs
        ],
        "vehicles": {
            vid: {
                "order": list(veh.customer_order),
                "load": veh.load,
                "capacity": veh.capacity,
                "capacity_ok": veh.capacity_feasible,
                "windows_ok": veh.time_window_feasible,
                "connected": veh.connectivity_feasible,
                "drive_s": veh.driving_time_s,
                "wait_s": veh.waiting_time_s,
                "service_s": veh.service_time_s,
                "elapsed_s": veh.elapsed_time_s,
                "starts": dict(veh.service_starts_s),
                "edges": [list(leg.edge_ids) for leg in veh.legs],
            }
            for vid, veh in result.vehicles.items()
        },
    }


def evaluation_payload(evaluation: RoutePlanEvaluation) -> dict:
    vehicles = []
    for veh in evaluation.vehicle_evaluations:
        edge_ids: list[str] = []
        if veh.physical_route is not None:
            for leg in veh.physical_route.legs:
                edge_ids.extend(str(eid) for eid in getattr(leg, "edge_ids", ()))
        if not edge_ids:
            for stop in veh.stops:
                edge_ids.extend(str(eid) for eid in stop.edge_ids)
        vehicles.append(
            {
                "id": veh.vehicle_id,
                "order": list(veh.customer_ids),
                "feasible": veh.feasible,
                "capacity_ok": veh.capacity_feasible,
                "windows_ok": veh.time_window_feasible,
                "connected": veh.connectivity_feasible,
                "depot_ok": veh.depot_feasible,
                "load": veh.maximum_load_units,
                "capacity": veh.capacity_units,
                "distance_m": veh.total_distance_m,
                "drive_s": veh.total_travel_time_s,
                "wait_s": veh.total_waiting_time_s,
                "service_s": veh.total_service_time_s,
                "elapsed_s": veh.elapsed_time_s,
                "edge_ids": edge_ids,
                "stops": [
                    {
                        "job": stop.customer_id,
                        "node": stop.node_id,
                        "arrive_s": stop.arrival_time_s,
                        "wait_s": stop.waiting_time_s,
                        "start_s": stop.service_start_time_s,
                        "end_s": stop.service_end_time_s,
                        "earliest_s": stop.earliest_time_s,
                        "latest_s": stop.latest_time_s,
                        "window_ok": stop.time_window_feasible,
                    }
                    for stop in veh.stops
                ],
            }
        )
    return {
        "feasible": evaluation.feasible,
        "capacity": evaluation.capacity_feasible,
        "windows": evaluation.time_window_feasible,
        "connectivity": evaluation.connectivity_feasible,
        "commitments": evaluation.commitment_feasible,
        "depot": evaluation.depot_feasible,
        "all_served": evaluation.all_requests_served,
        "unserved": list(evaluation.unserved_customer_ids),
        "objective": evaluation.objective_value,
        "time_s": evaluation.total_elapsed_time_s,
        "distance_m": evaluation.total_distance_m,
        "drive_s": evaluation.total_travel_time_s,
        "wait_s": evaluation.total_waiting_time_s,
        "service_s": evaluation.total_service_time_s,
        "congestion_s": max(
            0.0,
            float(evaluation.total_travel_time_s)
            - sum(
                _free_flow_hint(veh)
                for veh in evaluation.vehicle_evaluations
            ),
        ),
        "vehicles": vehicles,
        "violations": [
            {
                "name": getattr(item, "name", str(item)),
                "detail": str(item),
            }
            for item in evaluation.violations
        ],
    }


def _free_flow_hint(vehicle: Any) -> float:
    route = getattr(vehicle, "physical_route", None)
    if route is None:
        return float(getattr(vehicle, "total_travel_time_s", 0.0))
    return float(getattr(route, "total_free_flow_time_s", vehicle.total_travel_time_s))


def plan_orders(evaluation: RoutePlanEvaluation) -> dict[str, list[str]]:
    return {
        veh.vehicle_id: [str(cid) for cid in veh.customer_ids]
        for veh in evaluation.vehicle_evaluations
    }


def normalize_orders(plan: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    return {str(vid): [str(cid) for cid in order] for vid, order in plan.items()}
