"""Independent Step 3 routing and feasibility evaluator.

This module intentionally does not import optimizer/search code. It computes
legal directed shortest paths, route timing, capacity, service-start windows,
and return-to-depot feasibility from the fixture representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf
from typing import Any, Mapping, Sequence

EPS = 1e-9

@dataclass(frozen=True)
class LegResult:
    from_node: str
    to_node: str
    edge_ids: tuple[str, ...]
    travel_time_s: float

@dataclass(frozen=True)
class VehicleEvaluation:
    vehicle_id: str
    customer_order: tuple[str, ...]
    load: float
    capacity: float
    capacity_feasible: bool
    connectivity_feasible: bool
    time_window_feasible: bool
    driving_time_s: float
    waiting_time_s: float
    service_time_s: float
    elapsed_time_s: float
    service_starts_s: Mapping[str, float]
    service_ends_s: Mapping[str, float]
    legs: tuple[LegResult, ...]

@dataclass(frozen=True)
class EvaluationResult:
    feasible: bool
    capacity_feasible: bool
    connectivity_feasible: bool
    time_window_feasible: bool
    all_requests_served: bool
    vehicles: Mapping[str, VehicleEvaluation]
    unserved_request_ids: tuple[str, ...] = ()
    disconnected_legs: tuple[tuple[str, str, str, str], ...] = ()
    window_violations: tuple[tuple[str, str, float, float], ...] = ()


def shortest_directed_path(
    scenario: Mapping[str, Any],
    source: str,
    target: str,
    closed_edge_ids: Sequence[str] = (),
) -> tuple[tuple[str, ...], float] | None:
    """Return the minimum-travel-time legal directed path, or None."""
    if source == target:
        return (), 0.0

    outgoing: dict[str, list[dict[str, Any]]] = {}
    for edge in scenario["edges"]:
        outgoing.setdefault(edge["from_node"], []).append(edge)

    closed = set(closed_edge_ids)
    distances = {source: 0.0}
    previous: dict[str, tuple[str, str]] = {}
    heap: list[tuple[float, str]] = [(0.0, source)]

    while heap:
        distance, node = heappop(heap)
        if distance > distances.get(node, inf) + EPS:
            continue
        if node == target:
            break
        for edge in outgoing.get(node, ()):
            if not edge.get("open_by_default", True):
                continue
            if edge["edge_id"] in closed:
                continue
            travel_time = edge["length_m"] / edge["speed_limit_mps"]
            next_node = edge["to_node"]
            candidate = distance + travel_time
            if candidate < distances.get(next_node, inf) - EPS:
                distances[next_node] = candidate
                previous[next_node] = (node, edge["edge_id"])
                heappush(heap, (candidate, next_node))

    if target not in distances:
        return None

    edge_ids: list[str] = []
    node = target
    while node != source:
        previous_node, edge_id = previous[node]
        edge_ids.append(edge_id)
        node = previous_node
    edge_ids.reverse()
    return tuple(edge_ids), distances[target]


def evaluate_scenario(
    scenario: Mapping[str, Any],
    plan: Mapping[str, Sequence[str]],
    *,
    closed_edge_ids: Sequence[str] = (),
) -> EvaluationResult:
    """Evaluate a vehicle->job-order plan without scenario-specific logic."""
    jobs = {job["request_id"]: job for job in scenario["requests"]}
    vehicles = {vehicle["vehicle_id"]: vehicle for vehicle in scenario["fleet"]}

    assigned: list[str] = []
    results: dict[str, VehicleEvaluation] = {}
    disconnected: list[tuple[str, str, str, str]] = []
    window_violations: list[tuple[str, str, float, float]] = []

    for vehicle_id, vehicle in vehicles.items():
        order = tuple(plan.get(vehicle_id, ()))
        assigned.extend(order)
        unknown = [job_id for job_id in order if job_id not in jobs]
        if unknown:
            raise ValueError(f"Unknown request IDs in {vehicle_id}: {unknown}")
        if len(set(order)) != len(order):
            raise ValueError(f"Duplicate request in {vehicle_id}: {order}")

        load = sum(jobs[job_id]["demand"] for job_id in order)
        capacity_ok = load <= vehicle["capacity"] + EPS
        current_node = vehicle["depot_node"]
        current_time = 0.0
        driving = waiting = service = 0.0
        starts: dict[str, float] = {}
        ends: dict[str, float] = {}
        legs: list[LegResult] = []
        connectivity_ok = True
        windows_ok = True

        for job_id in order:
            job = jobs[job_id]
            target = job["node_id"]
            path = shortest_directed_path(scenario, current_node, target, closed_edge_ids)
            if path is None:
                connectivity_ok = False
                disconnected.append((vehicle_id, current_node, target, job_id))
                current_node = target
                continue

            edge_ids, travel_time = path
            legs.append(LegResult(current_node, target, edge_ids, travel_time))
            driving += travel_time
            arrival = current_time + travel_time
            start = max(arrival, job["earliest_start_s"])
            waiting += max(0.0, job["earliest_start_s"] - arrival)
            if start > job["latest_start_s"] + EPS:
                windows_ok = False
                window_violations.append((vehicle_id, job_id, start, job["latest_start_s"]))
            end = start + job["service_duration_s"]
            service += job["service_duration_s"]
            starts[job_id] = start
            ends[job_id] = end
            current_time = end
            current_node = target

        return_path = shortest_directed_path(scenario, current_node, vehicle["depot_node"], closed_edge_ids)
        if return_path is None:
            connectivity_ok = False
            disconnected.append((vehicle_id, current_node, vehicle["depot_node"], "__return__"))
        else:
            edge_ids, travel_time = return_path
            legs.append(LegResult(current_node, vehicle["depot_node"], edge_ids, travel_time))
            driving += travel_time
            current_time += travel_time

        results[vehicle_id] = VehicleEvaluation(
            vehicle_id=vehicle_id,
            customer_order=order,
            load=load,
            capacity=vehicle["capacity"],
            capacity_feasible=capacity_ok,
            connectivity_feasible=connectivity_ok,
            time_window_feasible=windows_ok,
            driving_time_s=driving,
            waiting_time_s=waiting,
            service_time_s=service,
            elapsed_time_s=current_time,
            service_starts_s=starts,
            service_ends_s=ends,
            legs=tuple(legs),
        )

    required = set(jobs)
    assigned_set = set(assigned)
    all_served = assigned_set == required and len(assigned) == len(assigned_set)
    capacity_ok = all(v.capacity_feasible for v in results.values())
    connectivity_ok = all(v.connectivity_feasible for v in results.values())
    windows_ok = all(v.time_window_feasible for v in results.values())

    return EvaluationResult(
        feasible=capacity_ok and connectivity_ok and windows_ok and all_served,
        capacity_feasible=capacity_ok,
        connectivity_feasible=connectivity_ok,
        time_window_feasible=windows_ok,
        all_requests_served=all_served,
        vehicles=results,
        unserved_request_ids=tuple(sorted(required - assigned_set)),
        disconnected_legs=tuple(disconnected),
        window_violations=tuple(window_violations),
    )
