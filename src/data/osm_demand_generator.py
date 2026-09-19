"""Generate a fully validated Quanta ``Scenario`` directly on a real OSM road
graph.

Unlike ``osm_scenario.OSMScenarioBuilder`` (which attaches an OSM network to
demand/fleet state that already exists), this module *creates* demand and
fleet placement natively on real OSM node IDs. There is no synthetic grid
underneath it and no coordinate snapping: every request and vehicle sits on
an OSM node that genuinely exists in the ingested network.

Design constraints that shaped this module:

* The road graph is directed (one-way streets, slip roads, etc. are real).
  A node being reachable *from* the depot does not imply the depot is
  reachable *from* that node, and two customers that are each individually
  reachable from the depot are not guaranteed to be mutually reachable.
  Building a nearest-neighbour reference schedule (which needs shortest
  paths *between customers*, not just depot-to-customer) over an
  arbitrarily chosen reachable set can therefore hit nodes with no legal
  path between them.

  This module avoids that failure mode structurally: candidate customer
  nodes are drawn only from the depot's strongly connected component
  (SCC) -- the set of nodes that can both reach the depot and be reached
  from it. Membership in the same SCC guarantees every pair of candidates
  is mutually reachable, so the reference-schedule construction can never
  raise ``PathNotFoundError``.
* Depot placement is chosen by graph connectivity (largest SCC among a
  pool of high-degree candidates), not arbitrarily, so the generator does
  not accidentally pick a dead-end or a disconnected service road as the
  depot.
* Time windows are derived from an actual nearest-neighbour reference
  schedule computed with real shortest-path travel times on the real
  graph -- not arbitrary jitter -- so they are internally consistent with
  how long it actually takes to get around the real road network.
* Capacity infeasibility is caught before Scenario construction, with the
  actual total demand and total fleet capacity surfaced in the error
  message.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import random
from math import inf
from typing import Iterable

from src.contracts.core_types import NodeKind
from src.contracts.scenario import (
    CoordinateTransform,
    RandomSeeds,
    Request,
    RoadNode,
    Scenario,
    Units,
    Vehicle,
)
from src.routing.path_builder import DirectedPathBuilder, PathNotFoundError
from .osm_ingestion import OSMNetwork


class OSMDemandGenerationError(ValueError):
    """Raised when a real-network demand/fleet scenario cannot be built."""


@dataclass(frozen=True, slots=True)
class OSMDemandConfig:
    """Policy for generating demand/fleet directly on an OSM network."""

    customer_count: int
    fleet_size: int
    vehicle_capacity: float
    seed: int = 0

    demand_min: float = 1.0
    demand_max: float = 8.0

    service_duration_s: float = 180.0

    # Reference-schedule -> time-window derivation.
    depot_start_time_s: float = 0.0
    time_window_slack_s: float = 900.0

    # Depot selection: how many high-degree candidates to evaluate for SCC
    # size before picking the most connected one. Larger values are more
    # thorough but cost one BFS pair per candidate.
    depot_candidate_pool_size: int = 25

    depot_node_id: str | None = None

    def __post_init__(self) -> None:
        if self.customer_count <= 0:
            raise OSMDemandGenerationError("customer_count must be positive")
        if self.fleet_size <= 0:
            raise OSMDemandGenerationError("fleet_size must be positive")
        if self.vehicle_capacity <= 0:
            raise OSMDemandGenerationError("vehicle_capacity must be positive")
        if self.demand_min <= 0 or self.demand_max < self.demand_min:
            raise OSMDemandGenerationError(
                "demand_max must be >= demand_min > 0"
            )
        if self.service_duration_s < 0:
            raise OSMDemandGenerationError("service_duration_s must be >= 0")
        if self.time_window_slack_s <= 0:
            raise OSMDemandGenerationError("time_window_slack_s must be positive")
        if self.depot_candidate_pool_size <= 0:
            raise OSMDemandGenerationError(
                "depot_candidate_pool_size must be positive"
            )


@dataclass(frozen=True, slots=True)
class OSMDemandGenerationResult:
    scenario: Scenario
    depot_node_id: str
    customer_node_ids: tuple[str, ...]
    unreachable_candidates_skipped: int


def _forward_reachable(adjacency: dict[str, list[str]], start: str) -> set[str]:
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _build_adjacency(network: OSMNetwork) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    forward: dict[str, list[str]] = {}
    backward: dict[str, list[str]] = {}
    for edge in network.edges:
        if not edge.open_by_default:
            continue
        forward.setdefault(str(edge.from_node), []).append(str(edge.to_node))
        backward.setdefault(str(edge.to_node), []).append(str(edge.from_node))
    return forward, backward


def _strongly_connected_component(
    forward: dict[str, list[str]],
    backward: dict[str, list[str]],
    node: str,
) -> set[str]:
    reachable_from = _forward_reachable(forward, node)
    reachable_to = _forward_reachable(backward, node)
    return reachable_from & reachable_to


def _choose_depot(
    network: OSMNetwork,
    forward: dict[str, list[str]],
    backward: dict[str, list[str]],
    config: OSMDemandConfig,
) -> tuple[str, set[str]]:
    if config.depot_node_id is not None:
        depot = str(config.depot_node_id)
        if depot not in {str(n.node_id) for n in network.nodes}:
            raise OSMDemandGenerationError(
                f"depot_node_id {depot!r} is not a node in this OSM network"
            )
        return depot, _strongly_connected_component(forward, backward, depot)

    # Rank candidates by total degree (a cheap, deterministic proxy for
    # connectivity) and evaluate the SCC of the top pool. This avoids an
    # O(nodes) full-BFS sweep on very large real networks while still
    # reliably avoiding dead-ends and disconnected service roads.
    degree: dict[str, int] = {}
    for node in network.nodes:
        node_id = str(node.node_id)
        degree[node_id] = len(forward.get(node_id, ())) + len(backward.get(node_id, ()))

    ranked = sorted(degree.items(), key=lambda item: (-item[1], item[0]))
    pool = [node_id for node_id, deg in ranked[: config.depot_candidate_pool_size] if deg > 0]
    if not pool:
        raise OSMDemandGenerationError(
            "OSM network has no node with both incoming and outgoing edges; "
            "cannot place a depot"
        )

    best_node: str | None = None
    best_scc: set[str] = set()
    for candidate in pool:
        scc = _strongly_connected_component(forward, backward, candidate)
        if len(scc) > len(best_scc):
            best_node, best_scc = candidate, scc

    assert best_node is not None
    return best_node, best_scc


def _sumo_compatible_path(
    network: OSMNetwork,
    start_node: str,
    end_node: str,
    *,
    previous_edge_id: str | None = None,
) -> tuple[list[str], float]:
    """Return a shortest path that respects SUMO's no-immediate-U-turn rule.

    The state contains the previous edge's origin node.  This is deliberately
    equivalent to the physical feasibility rule used by the SUMO route builder:
    an edge B->A is forbidden immediately after A->B.  The special case where
    reversing is the only outgoing movement is treated as a genuine dead-end,
    matching the route builder's SUMO-export behaviour.
    """
    if start_node == end_node:
        return [], 0.0

    edges = {
        str(edge.edge_id): edge
        for edge in network.edges
        if edge.open_by_default
    }
    outgoing: dict[str, list] = {}
    for edge in edges.values():
        outgoing.setdefault(str(edge.from_node), []).append(edge)
    for values in outgoing.values():
        values.sort(key=lambda edge: str(edge.edge_id))

    initial_prev_from: str | None = None
    if previous_edge_id is not None:
        previous = edges.get(str(previous_edge_id))
        if previous is not None:
            initial_prev_from = str(previous.from_node)

    start_state = (str(start_node), initial_prev_from)
    distances = {start_state: 0.0}
    previous_state: dict[
        tuple[str, str | None],
        tuple[tuple[str, str | None], str],
    ] = {}
    queue = [(0.0, str(start_node), "", str(start_node), initial_prev_from)]

    goal_state: tuple[str, str | None] | None = None

    while queue:
        cost, _, _, node, prev_from = heapq.heappop(queue)
        state = (node, prev_from)
        if cost > distances.get(state, inf):
            continue
        if node == str(end_node):
            goal_state = state
            break

        choices = outgoing.get(node, ())
        legal = [
            edge
            for edge in choices
            if prev_from is None or str(edge.to_node) != prev_from
        ]

        # At a genuine cul-de-sac, SUMO can only leave by reversing.
        if not legal and prev_from is not None:
            legal = list(choices)

        for edge in legal:
            next_state = (str(edge.to_node), str(edge.from_node))
            next_cost = cost + float(edge.free_flow_time_s)
            if next_cost < distances.get(next_state, inf):
                distances[next_state] = next_cost
                previous_state[next_state] = (state, str(edge.edge_id))
                heapq.heappush(
                    queue,
                    (
                        next_cost,
                        str(edge.to_node),
                        str(edge.edge_id),
                        str(edge.to_node),
                        str(edge.from_node),
                    ),
                )

    if goal_state is None:
        raise PathNotFoundError(
            f"No SUMO-compatible no-U-turn path exists from "
            f"{start_node!r} to {end_node!r}"
        )

    edge_ids: list[str] = []
    state = goal_state
    while state != start_state:
        prior, edge_id = previous_state[state]
        edge_ids.append(edge_id)
        state = prior
    edge_ids.reverse()
    return edge_ids, distances[goal_state]


def _find_feasible_customer_order(
    network: OSMNetwork,
    depot_node_id: str,
    customer_node_ids: tuple[str, ...],
    *,
    rng: random.Random,
    max_attempts: int = 12,
) -> tuple[str, ...] | None:
    """Find a complete depot->customers->depot SUMO-compatible chain.

    SCC membership guarantees directed reachability, but it does not encode
    the turn-state carried between consecutive physical legs.  This preflight
    therefore validates the actual stateful no-U-turn constraint before a
    scenario is emitted.  A failed customer realization is rejected and can
    be deterministically resampled by the caller.
    """
    if not customer_node_ids:
        return ()

    customers = list(customer_node_ids)

    for _ in range(max_attempts):
        remaining = set(customers)
        order: list[str] = []
        current = str(depot_node_id)
        previous_edge_id: str | None = None

        while remaining:
            candidates = list(remaining)
            rng.shuffle(candidates)

            feasible: list[tuple[float, str, list[str]]] = []
            for candidate in candidates:
                try:
                    path, travel_time = _sumo_compatible_path(
                        network,
                        current,
                        candidate,
                        previous_edge_id=previous_edge_id,
                    )
                except PathNotFoundError:
                    continue
                feasible.append((travel_time, candidate, path))

                # Avoid O(n) shortest-path evaluations once we have a clearly
                # feasible continuation. Sorting a small set gives stable,
                # route-efficient reference chains without exhaustive search.
                if len(feasible) >= min(4, len(candidates)):
                    break

            if not feasible:
                break

            _, selected, path = min(
                feasible,
                key=lambda item: (item[0], item[1]),
            )
            order.append(selected)
            remaining.remove(selected)
            current = selected
            if path:
                previous_edge_id = path[-1]

        if not remaining:
            try:
                _sumo_compatible_path(
                    network,
                    current,
                    str(depot_node_id),
                    previous_edge_id=previous_edge_id,
                )
            except PathNotFoundError:
                continue
            return tuple(order)

    return None


def _reference_schedule(
    path_builder: DirectedPathBuilder,
    depot_node_id: str,
    customer_node_ids: tuple[str, ...],
    *,
    start_time_s: float,
    service_duration_s: float,
    customer_order: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Nearest-neighbour arrival-time schedule using real shortest paths.

    All ``customer_node_ids`` are guaranteed (by construction, upstream) to
    lie in the depot's strongly connected component, so every
    ``shortest_path`` call below is between mutually reachable nodes and
    cannot raise ``PathNotFoundError``. It is still caught defensively: a
    caller-supplied ``depot_node_id`` bypasses the SCC-derived candidate
    pool, so this stays a hard error with a clear message rather than a
    silent skip.
    """
    arrival_at_depot = start_time_s
    current = depot_node_id
    current_time = start_time_s
    remaining = list(customer_order or customer_node_ids)
    schedule: dict[str, float] = {}

    if customer_order is not None:
        if set(customer_order) != set(customer_node_ids):
            raise OSMDemandGenerationError(
                "customer_order must contain exactly the generated customer nodes"
            )
        for best_node in customer_order:
            try:
                leg = path_builder.shortest_path(
                    current, best_node, departure_time_s=current_time
                )
            except PathNotFoundError as exc:
                raise OSMDemandGenerationError(
                    f"Reference schedule path missing from {current!r} "
                    f"to {best_node!r}"
                ) from exc
            arrival = current_time + leg.travel_time_s
            schedule[best_node] = arrival
            current = best_node
            current_time = arrival + service_duration_s
    else:
        while remaining:
            try:
                best_node = min(
                    remaining,
                    key=lambda node_id: path_builder.shortest_path(
                        current, node_id, departure_time_s=current_time
                    ).travel_time_s,
                )
            except PathNotFoundError as exc:
                raise OSMDemandGenerationError(
                    "No directed path exists between two candidate nodes while "
                    "building the reference schedule; this should be impossible "
                    "for candidates drawn from the depot's strongly connected "
                    "component. If depot_node_id was overridden explicitly, "
                    "verify it has a non-trivial SCC."
                ) from exc

            leg = path_builder.shortest_path(
                current, best_node, departure_time_s=current_time
            )
            arrival = current_time + leg.travel_time_s
            schedule[best_node] = arrival

            current = best_node
            current_time = arrival + service_duration_s
            remaining.remove(best_node)

    del arrival_at_depot  # documents intent; depot return is not scheduled here
    return schedule


def generate_osm_scenario(
    network: OSMNetwork,
    config: OSMDemandConfig,
    *,
    scenario_id: str,
    source_name: str = "osm",
    dataset_split: str = "generated",
    configuration_version: str = "osm-demand-generator-v1",
    generator_version: str = "osm-demand-generator-v1",
) -> OSMDemandGenerationResult:
    """Build a validated Scenario with demand/fleet placed on real OSM nodes."""
    if not network.projected:
        raise OSMDemandGenerationError(
            "OSM network must be projected into a metric CRS before demand "
            "generation (x_m/y_m must be metres, not lon/lat degrees)"
        )
    if not network.nodes:
        raise OSMDemandGenerationError("OSM network contains no nodes")

    forward, backward = _build_adjacency(network)
    depot_node_id, depot_scc = _choose_depot(network, forward, backward, config)

    candidate_pool = sorted(depot_scc - {depot_node_id})
    if len(candidate_pool) < config.customer_count:
        raise OSMDemandGenerationError(
            f"customer_count={config.customer_count} exceeds the number of "
            f"nodes reachable to/from the chosen depot ({len(candidate_pool)} "
            f"available in its strongly connected component); lower "
            "customer_count, choose a smaller network region, or supply a "
            "different depot_node_id"
        )

    all_node_ids = {str(n.node_id) for n in network.nodes}
    unreachable_candidates_skipped = len(all_node_ids) - 1 - len(candidate_pool)

    rng = random.Random(config.seed)
    path_builder = DirectedPathBuilder(network.edges)

    # SCC membership proves directed reachability, but the SUMO route builder
    # carries the previous physical edge between legs and forbids immediate
    # U-turns.  Generate only customer realizations for which at least one
    # complete depot->customers->depot chain is physically SUMO-compatible.
    customer_node_ids: tuple[str, ...] | None = None
    customer_order: tuple[str, ...] | None = None
    max_realization_attempts = max(24, min(96, config.customer_count * 2))

    for _attempt in range(max_realization_attempts):
        sampled = tuple(
            sorted(rng.sample(candidate_pool, config.customer_count))
        )
        feasible_order = _find_feasible_customer_order(
            network,
            depot_node_id,
            sampled,
            rng=rng,
        )
        if feasible_order is not None:
            customer_node_ids = sampled
            customer_order = feasible_order
            break

    if customer_node_ids is None or customer_order is None:
        raise OSMDemandGenerationError(
            "Could not construct a SUMO-compatible customer realization after "
            f"{max_realization_attempts} deterministic attempts. The depot SCC "
            f"contains {len(candidate_pool)} candidate nodes, but the sampled "
            "customer sets could not form a complete no-U-turn route chain. "
            "Try a different seed, smaller customer_count, or a larger OSM region."
        )

    schedule = _reference_schedule(
        path_builder,
        depot_node_id,
        customer_node_ids,
        start_time_s=config.depot_start_time_s,
        service_duration_s=config.service_duration_s,
        customer_order=customer_order,
    )

    demands = {
        node_id: rng.uniform(config.demand_min, config.demand_max)
        for node_id in customer_node_ids
    }
    total_demand = sum(demands.values())
    total_capacity = config.fleet_size * config.vehicle_capacity
    if total_demand > total_capacity:
        raise OSMDemandGenerationError(
            f"infeasible fleet: total demand ({total_demand:.2f}) exceeds "
            f"total fleet capacity ({total_capacity:.2f}) = "
            f"fleet_size({config.fleet_size}) x "
            f"vehicle_capacity({config.vehicle_capacity:.2f}); increase "
            "fleet_size or vehicle_capacity, or lower demand_max"
        )

    requests = tuple(
        Request(
            request_id=f"osm-req:{node_id}",
            original_customer_id=node_id,
            original_x=next(n.x_m for n in network.nodes if str(n.node_id) == node_id),
            original_y=next(n.y_m for n in network.nodes if str(n.node_id) == node_id),
            access_node_id=node_id,
            access_distance_m=0.0,
            demand=demands[node_id],
            known_at_s=0.0,
            release_s=0.0,
            earliest_service_start_s=max(0.0, schedule[node_id] - config.time_window_slack_s),
            latest_service_start_s=schedule[node_id] + config.time_window_slack_s,
            service_duration_s=config.service_duration_s,
        )
        for node_id in customer_node_ids
    )

    fleet = tuple(
        Vehicle(
            vehicle_id=f"osm-veh:{i}",
            capacity=config.vehicle_capacity,
            start_node_id=depot_node_id,
            depot_node_id=depot_node_id,
        )
        for i in range(config.fleet_size)
    )

    customer_ids = set(customer_node_ids)
    marked_nodes = tuple(
        node.model_copy(
            update={
                "kind": (
                    NodeKind.DEPOT
                    if str(node.node_id) == depot_node_id
                    else NodeKind.CUSTOMER_ACCESS
                    if str(node.node_id) in customer_ids
                    else node.kind
                )
            }
        )
        for node in network.nodes
    )

    seeds = RandomSeeds(
        scenario_seed=config.seed,
        road_seed=config.seed,
        traffic_seed=config.seed,
        incident_seed=config.seed,
        window_seed=config.seed,
        optimizer_seed=config.seed,
        learning_seed=config.seed,
    )

    scenario = Scenario(
        schema_version="1.2",
        scenario_id=scenario_id,
        source_name=source_name,
        source_checksum=None,
        units=Units(),
        coordinate_transform=CoordinateTransform(
            scale=1.0,
            translation_x_m=0.0,
            translation_y_m=0.0,
            description="native OSM projected coordinates; no transform applied",
        ),
        graph_version=f"{scenario_id}:osm-demand-generator",
        nodes=marked_nodes,
        edges=network.edges,
        requests=requests,
        fleet=fleet,
        seeds=seeds,
        generator_version=generator_version,
        dataset_split=dataset_split,
        configuration_version=configuration_version,
        field_provenance={
            "osm_source_crs": network.source_crs or "unknown",
            "osm_depot_node_id": depot_node_id,
            "osm_depot_scc_size": str(len(depot_scc)),
            "osm_unreachable_candidates_skipped": str(unreachable_candidates_skipped),
            "osm_sumo_route_preflight": "depot-customers-depot-no-immediate-u-turn",
            "osm_reference_customer_order": ",".join(customer_order),
            "osm_demand_generator": generator_version,
        },
    )

    return OSMDemandGenerationResult(
        scenario=scenario,
        depot_node_id=depot_node_id,
        customer_node_ids=customer_node_ids,
        unreachable_candidates_skipped=unreachable_candidates_skipped,
    )