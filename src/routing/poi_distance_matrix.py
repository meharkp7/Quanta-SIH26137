"""Explicit POI routing-cache precomputation helpers.

This module provides a small orchestration layer around
``DirectedPathBuilder.precompute_poi_paths``.  It intentionally does not
replace the route evaluator or change shortest-path correctness semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

from src.routing.path_builder import (
    DirectedPathBuilder,
    PathPrecomputeStats,
)


@dataclass(frozen=True)
class POIPrecomputeStats:
    """Scenario-level result of an explicit POI cache warm-up."""

    poi_node_ids: tuple[str, ...]
    departure_times_s: tuple[float, ...]
    path_stats: PathPrecomputeStats


def poi_nodes_from_scenario(scenario: Any) -> tuple[str, ...]:
    """Return deterministic routing POIs used by vehicles and requests."""
    vehicles = getattr(scenario, "fleet", ())
    requests = getattr(scenario, "requests", ())

    nodes: set[str] = set()
    for vehicle in vehicles:
        nodes.add(str(vehicle.start_node_id))
        nodes.add(str(vehicle.depot_node_id))
    for request in requests:
        nodes.add(str(request.access_node_id))

    return tuple(sorted(nodes))


def departure_time_grid(
    scenario: Any,
    *,
    margin_s: float = 0.0,
) -> tuple[float, ...]:
    """Build a deterministic departure-time grid from scenario requests.

    Release/earliest/latest times are included because they are the temporal
    anchors at which customer legs can begin.  The default margin is zero;
    callers can explicitly widen the grid when their workload warrants it.
    """
    margin = float(margin_s)
    if not isfinite(margin) or margin < 0.0:
        raise ValueError("margin_s must be finite and non-negative")

    times: set[float] = set()
    for request in getattr(scenario, "requests", ()):
        for name in (
            "release_s",
            "earliest_service_start_s",
            "latest_service_start_s",
        ):
            value = float(getattr(request, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"request {request.request_id!r}: {name} must be finite and non-negative")
            times.add(value)
            if margin:
                times.add(max(0.0, value - margin))
                times.add(value + margin)

    if not times:
        return (0.0,)
    return tuple(sorted(times))


def precompute_poi_distance_matrix(
    scenario: Any,
    path_builder: DirectedPathBuilder,
    *,
    margin_s: float = 0.0,
    poi_node_ids: Iterable[str] | None = None,
    departure_times_s: Iterable[float] | None = None,
) -> POIPrecomputeStats:
    """Warm the path cache for the scenario's POI pairs.

    This is intentionally explicit rather than automatically invoked by the
    evaluator: precomputation has a fixed upfront cost and is not guaranteed
    to outperform lazy caching for small/sparse workloads.
    """
    pois = (
        tuple(dict.fromkeys(str(node) for node in poi_node_ids))
        if poi_node_ids is not None
        else poi_nodes_from_scenario(scenario)
    )
    departures = (
        tuple(sorted({float(t) for t in departure_times_s}))
        if departure_times_s is not None
        else departure_time_grid(scenario, margin_s=margin_s)
    )

    if not pois:
        empty = PathPrecomputeStats(0, len(departures), 0, 0, 0, 0, 0)
        return POIPrecomputeStats(pois, departures, empty)

    stats = path_builder.precompute_poi_paths(pois, departures)
    return POIPrecomputeStats(pois, departures, stats)
