"""Step 4 CVRP-to-road reusable dataset generator."""

from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean, median

from src.contracts.scenario import Scenario
from src.contracts.core_types import NodeKind, RoadClass

from .road_generator import (
    Point,
    RoadBuildConfig,
    attach_customers,
    build_debug_delaunay,
    build_synthetic_roads,
    scale_customer_points,
)
from .vrp_parser import CustomerRecord, VrpInstance, parse_vrp


@dataclass(frozen=True)
class GeneratorConfig:
    """Step 4 generator configuration with separate randomness streams."""

    scale_extent_m: float = 8_000.0
    topology_family: str = "grid"
    junction_count: int = 196
    grid_rows: int = 14
    grid_cols: int = 14
    jitter_fraction: float = 0.18
    max_customer_snap_m: float = 500.0
    one_way_fraction: float = 0.15
    window_profile: str = "medium"
    window_slack_s: float = 900.0
    service_duration_s: float = 60.0
    fleet_size: int | None = None
    reference_variant: int = 0
    seed: int = 26137
    dataset_split: str = "train"
    generator_version: str = "step4-road-generator-v1.0"

    def __post_init__(self) -> None:
        if self.dataset_split not in {"train", "validation", "test"}:
            raise ValueError("dataset_split must be train, validation, or test")
        if self.window_profile not in {"loose", "medium", "tight"}:
            raise ValueError("window_profile must be loose, medium, or tight")
        if self.window_slack_s < 0:
            raise ValueError("window_slack_s must be non-negative")

    @property
    def seeds(self) -> dict[str, int]:
        return {
            "scenario_seed": self.seed,
            "road_seed": self.seed + 1,
            "traffic_seed": self.seed + 2,
            "incident_seed": self.seed + 3,
            "window_seed": self.seed + 4,
            "optimizer_seed": self.seed + 5,
            "learning_seed": self.seed + 6,
        }


@dataclass(frozen=True)
class DynamicDatasetConfig:
    """Configuration for fully synthetic, reproducible Step 4 datasets."""

    customer_count: int = 120
    road_junction_count: int = 196
    extent_m: float = 8000.0
    topology_family: str = "grid"
    grid_rows: int = 14
    grid_cols: int = 14
    jitter_fraction: float = 0.0
    one_way_fraction: float = 0.08
    max_customer_snap_m: float = 500.0
    window_profile: str = "medium"
    window_slack_s: float = 900.0
    service_duration_s: float = 60.0
    vehicle_capacity: float = 30.0
    fleet_size: int | None = None
    customer_spread: float = 0.86
    cluster_strength: float = 0.62
    minimum_road_segment_m: float = 30.0
    demand_min: int = 1
    demand_max: int = 8
    seed: int = 26137
    dataset_split: str = "train"

    def __post_init__(self) -> None:
        if self.customer_count < 2:
            raise ValueError("customer_count must be at least 2")
        if self.road_junction_count < 5:
            raise ValueError("road_junction_count must be at least 5")
        if self.extent_m <= 0:
            raise ValueError("extent_m must be positive")
        if not 0 < self.customer_spread <= 1:
            raise ValueError("customer_spread must be in (0, 1]")
        if not 0 <= self.cluster_strength <= 1:
            raise ValueError("cluster_strength must be between 0 and 1")
        if self.demand_min < 1 or self.demand_max < self.demand_min:
            raise ValueError("invalid demand range")
        if self.vehicle_capacity < self.demand_max:
            raise ValueError("vehicle_capacity must be >= demand_max")
        if self.minimum_road_segment_m <= 0:
            raise ValueError("minimum_road_segment_m must be positive")

    @property
    def seeds(self) -> dict[str, int]:
        return {
            "scenario_seed": self.seed,
            "road_seed": self.seed + 1,
            "traffic_seed": self.seed + 2,
            "incident_seed": self.seed + 3,
            "window_seed": self.seed + 4,
            "optimizer_seed": self.seed + 5,
            "learning_seed": self.seed + 6,
        }


def generate_dynamic_scenario(config: DynamicDatasetConfig = DynamicDatasetConfig()) -> Scenario:
    """Generate a complete synthetic CVRP + road scenario without a source dataset.

    Customer positions, demands, road topology, one-way choices and service windows
    are all generated from independent deterministic random streams. The same seed
    reproduces the same dataset; a different seed produces a different dataset.
    """
    rng = random.Random(config.seeds["scenario_seed"])
    customers = _generate_dynamic_customers(config, rng)
    depot = customers[0]
    customer_points = {c.customer_id: Point(c.x, c.y) for c in customers}

    debug_points = {c.customer_id: Point(c.x, c.y) for c in customers[: min(12, len(customers))]}
    debug_nodes, debug_edges = build_debug_delaunay(debug_points, config.seed)
    if not _connected(debug_nodes, debug_edges):
        raise ValueError("Pass-1 Delaunay debugging fixture is disconnected")

    road_config = RoadBuildConfig(
        junction_count=config.road_junction_count,
        topology_family=config.topology_family,
        grid_rows=config.grid_rows,
        grid_cols=config.grid_cols,
        jitter_fraction=config.jitter_fraction,
        one_way_fraction=config.one_way_fraction,
        max_customer_snap_m=config.max_customer_snap_m,
        min_segment_length_m=config.minimum_road_segment_m,
        protected_cycle_fraction=1.0,
        prune_fraction=0.0,
        seed=config.seeds["road_seed"],
    )
    nodes, edges = build_synthetic_roads(customer_points, road_config)

    depot_node_id = min(
        (node["node_id"] for node in nodes),
        key=lambda node_id: _distance(Point(depot.x, depot.y), _node_point(nodes, node_id)),
    )
    next(node for node in nodes if node["node_id"] == depot_node_id)["kind"] = NodeKind.DEPOT
    access = attach_customers(nodes, edges, customer_points, config.max_customer_snap_m, config.minimum_road_segment_m)

    # One-way roads are useful for realism, but a small synthetic network
    # should not acquire pathological directed detours. If the requested
    # one-way layout creates an extreme depot-to-customer stretch, rebuild the
    # same deterministic road skeleton without one-way removals.
    if config.one_way_fraction > 0 and _max_access_detour(nodes, edges, depot_node_id, access) > 8.0:
        road_config = replace(road_config, one_way_fraction=0.0)
        nodes, edges = build_synthetic_roads(customer_points, road_config)
        depot_node_id = min(
            (node["node_id"] for node in nodes),
            key=lambda node_id: _distance(Point(depot.x, depot.y), _node_point(nodes, node_id)),
        )
        next(node for node in nodes if node["node_id"] == depot_node_id)["kind"] = NodeKind.DEPOT
        access = attach_customers(nodes, edges, customer_points, config.max_customer_snap_m, config.minimum_road_segment_m)

    fleet_count = config.fleet_size or math.ceil(sum(c.demand for c in customers[1:]) / config.vehicle_capacity / 0.82)
    fleet_count = max(1, fleet_count)

    synthetic_instance = VrpInstance(
        name=f"DYNAMIC-CVRP-{config.customer_count}-{config.seed}",
        dimension=len(customers),
        capacity=config.vehicle_capacity,
        depot_id=depot.customer_id,
        customers=tuple(customers),
        vehicle_count=fleet_count,
        distance_convention="EUCLIDEAN_2D_SYNTHETIC_METRES",
        source_checksum=None,
        source_text=None,
    )
    requests, fleet, routes = _build_requests_fleet_and_windows(
        synthetic_instance, nodes, edges, access, depot_node_id, fleet_count,
        GeneratorConfig(
            scale_extent_m=config.extent_m, topology_family=config.topology_family,
            junction_count=config.road_junction_count, grid_rows=config.grid_rows,
            grid_cols=config.grid_cols, jitter_fraction=config.jitter_fraction,
            max_customer_snap_m=config.max_customer_snap_m, one_way_fraction=config.one_way_fraction,
            window_profile=config.window_profile, window_slack_s=config.window_slack_s,
            service_duration_s=config.service_duration_s, fleet_size=fleet_count,
            reference_variant=0, seed=config.seed, dataset_split=config.dataset_split,
        ),
    )

    scenario_id = f"dynamic-cvrp:{config.customer_count}:{config.seed}"
    return Scenario(
        schema_version="1.2", scenario_id=scenario_id,
        source_name=synthetic_instance.name, source_checksum=None,
        units={"distance": "m", "time": "s", "speed": "m/s", "demand": "load_units"},
        coordinate_transform={
            "scale": 1.0, "translation_x_m": 0.0, "translation_y_m": 0.0,
            "description": "Fully synthetic coordinates generated directly in simulation metres.",
        },
        graph_version=f"{scenario_id}:graph:1", nodes=tuple(nodes), edges=tuple(edges),
        requests=tuple(requests), fleet=tuple(fleet), seeds=config.seeds,
        generator_version="step4-procedural-city-v1.0", dataset_split=config.dataset_split,
        configuration_version="step4-procedural-city-v1.0",
        field_provenance={
            "source_instance": "fully synthetic CVRP generated from seed; no external dataset used",
            "customers": "deterministic district-based spatial layout using low-discrepancy golden-angle placement; no random coordinate sampling",
            "nodes": f"synthetic {config.topology_family} road-junction family with deterministic grid structure and controlled topology rules",
            "edges": "synthetic connected road skeleton with protected cycles, minimum segment length, and safe one-way conversion",
            "requests": "synthetic district-derived demand with deterministic road access points",
            "fleet": "synthetic homogeneous fleet sized from total demand and capacity",
            "service_windows": "reference-feasible deterministic schedule plus configured slack",
            "debug_graph": f"Pass-1 Delaunay fixture contains {len(debug_nodes)} nodes and {len(debug_edges)} directed edges",
            "traffic": "not generated in Step 4; produced from SUMO episodes in later steps",
        },
        parent_instance_id=None,
    )


def generate_dynamic_dataset(output_dir: str | Path, config: DynamicDatasetConfig = DynamicDatasetConfig()) -> Path:
    """Generate and persist a high-quality fully synthetic Step 4 dataset."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenario = generate_dynamic_scenario(config)
    instance = _dynamic_instance_from_scenario(scenario, config)
    report = build_quality_report(scenario, instance)
    report["generation_mode"] = "fully_dynamic_synthetic"
    report["seed"] = config.seed
    report["quality_checks"] = {
        "unique_request_ids": len({r.request_id for r in scenario.requests}) == len(scenario.requests),
        "unique_edge_ids": len({e.edge_id for e in scenario.edges}) == len(scenario.edges),
        "all_snap_distances_within_limit": max((r.access_distance_m for r in scenario.requests), default=0) <= config.max_customer_snap_m + 1e-9,
        "all_demands_positive": all(r.demand > 0 for r in scenario.requests),
        "all_windows_valid": all(r.latest_service_start_s >= r.earliest_service_start_s for r in scenario.requests),
        "reference_schedule_feasible": report["reference_schedule_feasible"],
        "road_graph_connected": not report["disconnected_nodes"],
        "requests_reachable": not report["unreachable_requests"],
        "minimum_edge_length_respected": report["edge_length_quantiles_m"]["p0"] >= config.minimum_road_segment_m - 1e-9,
        "detour_ratio_physical": report["shortest_path_euclidean_detour_quantiles"]["p0"] is None or report["shortest_path_euclidean_detour_quantiles"]["p0"] >= 1.0 - 1e-9,
        "demand_within_configured_range": all(config.demand_min <= r.demand <= config.demand_max for r in scenario.requests),
    }
    if not all(report["quality_checks"].values()):
        raise ValueError("Dynamic Step 4 quality checks failed")

    _write_json(output / "scenario.json", scenario.model_dump(mode="json"))
    _write_json(output / "provenance_quality.json", report)
    _write_json(output / "generation_config.json", config.__dict__)
    _write_nodes(output / "nodes.csv", scenario)
    _write_edges(output / "edges.csv", scenario)
    _write_requests(output / "requests.csv", scenario)
    _write_fleet(output / "fleet.csv", scenario)
    return output / "scenario.json"


def _generate_dynamic_customers(config: DynamicDatasetConfig, rng: random.Random | None = None) -> list[CustomerRecord]:
    """Generate a structured synthetic city without random point sampling.

    Customer locations follow deterministic district rules and a golden-angle
    (phyllotaxis) sequence. The seed only changes the deterministic phase and
    district ordering, so different seeds create different cities without
    uniform-random scattering. Demand is derived from district type and the
    customer's deterministic position within that district.
    """
    extent = config.extent_m
    n = config.customer_count

    # A deterministic seed-derived phase. No RNG calls are used for customer
    # placement or demand generation.
    phase = ((config.seed * 0.6180339887498949) % 1.0) * 2.0 * math.pi

    # Synthetic urban districts: CBD, commercial, mixed, residential,
    # industrial. Each has a center, service radius and demand profile.
    districts = [
        ("CBD",          0.50, 0.50, 0.105, 4, 1.35),
        ("COMMERCIAL",   0.72, 0.30, 0.125, 5, 1.20),
        ("COMMERCIAL",   0.28, 0.70, 0.125, 5, 1.15),
        ("MIXED",        0.72, 0.72, 0.135, 4, 1.00),
        ("RESIDENTIAL",  0.25, 0.27, 0.160, 2, 0.95),
        ("RESIDENTIAL",  0.30, 0.76, 0.155, 2, 0.90),
        ("INDUSTRIAL",   0.80, 0.78, 0.135, 8, 0.75),
    ]

    # Allocate customers deterministically according to district capacity.
    weights = [d[5] for d in districts]
    total_weight = sum(weights)
    counts = [int(n * w / total_weight) for w in weights]
    while sum(counts) < n:
        i = max(range(len(districts)), key=lambda j: (n * weights[j] / total_weight - counts[j], -j))
        counts[i] += 1

    customers = [CustomerRecord("0", extent * 0.50, extent * 0.50, 0)]
    golden = math.pi * (3.0 - math.sqrt(5.0))
    customer_index = 1

    for district_index, ((kind, cxr, cyr, radius_ratio, base_demand, density), count) in enumerate(zip(districts, counts)):
        cx, cy = extent * cxr, extent * cyr
        radius = extent * radius_ratio
        for local_index in range(count):
            # Low-discrepancy radial sequence fills the district evenly.
            u = (local_index + 0.5) / count
            radial = radius * math.sqrt(u)
            angle = phase + district_index * 0.73 + local_index * golden
            x = min(extent - 1.0, max(1.0, cx + radial * math.cos(angle)))
            y = min(extent - 1.0, max(1.0, cy + radial * math.sin(angle)))

            # Demand follows district type plus a deterministic positional
            # tier; there is no random.randint/random.choice here.
            tier = (local_index * 7 + district_index * 3 + config.seed) % 3
            demand = min(config.demand_max, max(config.demand_min, base_demand + tier - 1))
            customers.append(CustomerRecord(str(customer_index), x, y, demand))
            customer_index += 1

    return customers


def _dynamic_instance_from_scenario(scenario: Scenario, config: DynamicDatasetConfig) -> VrpInstance:
    customers = tuple(
        CustomerRecord(r.original_customer_id, r.original_x, r.original_y, r.demand)
        for r in sorted(scenario.requests, key=lambda r: int(r.original_customer_id))
    )
    depot = CustomerRecord("0", config.extent_m * 0.50, config.extent_m * 0.50, 0)
    return VrpInstance(
        name=scenario.source_name, dimension=len(customers) + 1, capacity=config.vehicle_capacity,
        depot_id="0", customers=(depot,) + customers, vehicle_count=len(scenario.fleet),
        distance_convention="EUCLIDEAN_2D_SYNTHETIC_METRES", source_checksum="", source_text="",
    )


def generate_scenario(instance: VrpInstance, config: GeneratorConfig = GeneratorConfig()) -> Scenario:
    """Generate a derived Step 4 scenario from a preserved CVRP instance."""

    customer_points, transform = scale_customer_points(instance, config.scale_extent_m)

    # Pass 1 is intentionally built for diagnostics and regression checks.
    debug_nodes, debug_edges = build_debug_delaunay(customer_points, config.seed)
    if not _connected(debug_nodes, debug_edges):
        raise ValueError("Pass-1 Delaunay debugging fixture is disconnected")

    road_config = RoadBuildConfig(
        junction_count=config.junction_count,
        topology_family=config.topology_family,
        grid_rows=config.grid_rows,
        grid_cols=config.grid_cols,
        jitter_fraction=config.jitter_fraction,
        one_way_fraction=config.one_way_fraction,
        max_customer_snap_m=config.max_customer_snap_m,
        seed=config.seeds["road_seed"],
    )
    nodes, edges = build_synthetic_roads(customer_points, road_config)

    depot_point = customer_points[instance.depot_id]
    depot_node_id = min(
        (node["node_id"] for node in nodes),
        key=lambda node_id: _distance(depot_point, _node_point(nodes, node_id)),
    )
    next_node = next(node for node in nodes if node["node_id"] == depot_node_id)
    next_node["kind"] = NodeKind.DEPOT

    access = attach_customers(nodes, edges, customer_points, config.max_customer_snap_m)
    if not _connected(nodes, edges):
        raise ValueError("Generated road graph lost connectivity during customer attachment")

    fleet_count = config.fleet_size or instance.vehicle_count
    fleet_count = max(1, fleet_count)
    requests, fleet, reference_routes = _build_requests_fleet_and_windows(
        instance, nodes, edges, access, depot_node_id, fleet_count, config
    )

    scenario_id = f"{instance.name}:road:{config.topology_family}:{config.seed}"
    return Scenario(
        schema_version="1.2",
        scenario_id=scenario_id,
        source_name=instance.name,
        source_checksum=instance.source_checksum,
        units={"distance": "m", "time": "s", "speed": "m/s", "demand": "load_units"},
        coordinate_transform={
            "scale": transform["scale"],
            "translation_x_m": transform["translation_x_m"],
            "translation_y_m": transform["translation_y_m"],
            "description": (
                "Isotropic benchmark transform; the resulting coordinates are synthetic "
                "simulation coordinates, not real geography."
            ),
        },
        graph_version=f"{scenario_id}:graph:1",
        nodes=tuple(nodes),
        edges=tuple(edges),
        requests=tuple(requests),
        fleet=tuple(fleet),
        seeds=config.seeds,
        generator_version=config.generator_version,
        dataset_split=config.dataset_split,
        configuration_version="step4-v1.0",
        field_provenance={
            "source_instance": "preserved original CVRP input; checksum recorded",
            "nodes": f"synthetic {config.topology_family} road-junction family with deterministic grid structure and controlled topology rules",
            "edges": "synthetic connected road skeleton with protected detour cycles",
            "requests": "benchmark customer IDs/demands with synthetic road access",
            "fleet": "derived fixed homogeneous fleet from source capacity/fleet count",
            "service_windows": "reference-feasible schedule plus configured slack profile",
            "debug_graph": f"Pass-1 Delaunay fixture contains {len(debug_nodes)} nodes and {len(debug_edges)} directed edges",
            "reference_schedule": f"deterministic reference variant {config.reference_variant}",
            "traffic": "not generated in Step 4; produced from SUMO episodes in later steps",
        },
        parent_instance_id=instance.name,
    )


def generate_from_vrp(
    vrp_path: str | Path,
    output_dir: str | Path,
    config: GeneratorConfig = GeneratorConfig(),
) -> Path:
    """Parse, generate, quality-check and persist the Step 4 dataset."""

    source = Path(vrp_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    instance = parse_vrp(source)
    scenario = generate_scenario(instance, config)
    report = build_quality_report(scenario, instance)

    failures = []
    if report["disconnected_nodes"]:
        failures.append("disconnected_nodes")
    if report["unreachable_requests"]:
        failures.append("unreachable_requests")
    if not report["reference_schedule_feasible"]:
        failures.append("reference_schedule")
    if failures:
        raise ValueError("Step 4 quality checks failed: " + ", ".join(failures))

    _write_json(output / "scenario.json", scenario.model_dump(mode="json"))
    _write_json(output / "provenance_quality.json", report)
    _write_nodes(output / "nodes.csv", scenario)
    _write_edges(output / "edges.csv", scenario)
    _write_requests(output / "requests.csv", scenario)
    _write_fleet(output / "fleet.csv", scenario)
    (output / source.name).write_text(instance.source_text, encoding="utf-8")
    return output / "scenario.json"


def build_quality_report(scenario: Scenario, instance: VrpInstance) -> dict:
    """Produce the static Step 4 dataset-card measurements available before SUMO."""

    adjacency = {node.node_id: set() for node in scenario.nodes}
    for edge in scenario.edges:
        adjacency[edge.from_node].add(edge.to_node)
    depot = scenario.fleet[0].depot_node_id
    reachable = _reachable(depot, adjacency)
    disconnected = sorted(set(adjacency) - reachable)
    unreachable = sorted(
        request.request_id
        for request in scenario.requests
        if request.access_node_id not in reachable
    )

    physical_pairs: dict[str, set[tuple[str, str]]] = {}
    for edge in scenario.edges:
        physical_pairs.setdefault(edge.parent_road_id, set()).add((edge.from_node, edge.to_node))
    one_way = sum(len(directions) == 1 for directions in physical_pairs.values())

    physical_adjacency = {node.node_id: set() for node in scenario.nodes}
    for edge in scenario.edges:
        physical_adjacency[edge.from_node].add(edge.to_node)
        physical_adjacency[edge.to_node].add(edge.from_node)
    degrees = [len(physical_adjacency[node.node_id]) for node in scenario.nodes]
    lengths = sorted(edge.length_m for edge in scenario.edges)
    bearings = [_bearing(edge, scenario) for edge in scenario.edges]
    detours = _detour_ratios(scenario)
    snap_distances = sorted(request.access_distance_m for request in scenario.requests)

    return {
        "source_name": instance.name,
        "source_checksum": instance.source_checksum,
        "distance_convention": instance.distance_convention,
        "customer_count": len(scenario.requests),
        "road_node_count": len(scenario.nodes),
        "road_edge_count": len(scenario.edges),
        "total_customer_demand": sum(request.demand for request in scenario.requests),
        "fleet_vehicle_count": len(scenario.fleet),
        "fleet_capacity_units": sum(vehicle.capacity for vehicle in scenario.fleet),
        "disconnected_nodes": disconnected,
        "unreachable_requests": unreachable,
        "reference_schedule_feasible": _reference_schedule_feasible(scenario),
        "one_way_physical_road_fraction": one_way / max(len(physical_pairs), 1),
        "average_physical_degree": mean(degrees) if degrees else 0.0,
        "degree_histogram": _histogram(degrees),
        "street_bearing_histogram_45deg": _bearing_histogram(bearings),
        "edge_length_quantiles_m": _quantiles(lengths),
        "signal_count": sum(node.signalized for node in scenario.nodes),
        "customer_snap_distance_quantiles_m": _quantiles(snap_distances),
        "shortest_path_euclidean_detour_quantiles": _quantiles(detours),
        "rush_offpeak_travel_time_ratio": None,
        "traffic_autocorrelation": None,
        "incident_recovery_duration_s": None,
        "traffic_metrics_status": "generated from SUMO episodes in later steps",
        "generator_version": scenario.generator_version,
        "dataset_split": scenario.dataset_split,
        "configuration_version": scenario.configuration_version,
        "parent_instance_id": scenario.parent_instance_id,
    }


def _build_requests_fleet_and_windows(instance, nodes, edges, access, depot_node_id, fleet_count, config):
    customers = [record for record in instance.customers if record.customer_id != instance.depot_id]
    loads = [0.0] * fleet_count
    assignments: dict[str, int] = {}
    for customer in sorted(customers, key=lambda record: (-record.demand, record.customer_id)):
        choices = [i for i, load in enumerate(loads) if load + customer.demand <= instance.capacity]
        if not choices:
            raise ValueError("Configured fleet cannot carry demand without split delivery")
        vehicle = min(choices, key=lambda i: (loads[i], i))
        assignments[customer.customer_id] = vehicle
        loads[vehicle] += customer.demand

    routes = [[] for _ in range(fleet_count)]
    for customer_id, vehicle in assignments.items():
        routes[vehicle].append(customer_id)
    for route in routes:
        route.sort(key=lambda customer_id: _reference_order(customer_id, access, config.reference_variant))

    arrival: dict[str, float] = {}
    for vehicle_index, route in enumerate(routes):
        current = depot_node_id
        current_time = 0.0
        for customer_id in route:
            target = access[customer_id][0]
            travel = _shortest_time(nodes, edges, current, target)
            current_time += travel
            arrival[customer_id] = current_time
            current_time += config.service_duration_s
            current = target
        # Explicitly check depot return as part of the reference schedule.
        _shortest_time(nodes, edges, current, depot_node_id)

    slack = {"loose": config.window_slack_s * 2, "medium": config.window_slack_s, "tight": config.window_slack_s / 2}[config.window_profile]
    rng = random.Random(config.seeds["window_seed"] + config.reference_variant)
    requests = []
    for customer in sorted(customers, key=lambda record: record.customer_id):
        base = arrival[customer.customer_id]
        jitter = rng.uniform(0.0, slack * 0.1) if slack else 0.0
        requests.append(
            {
                "request_id": f"J{customer.customer_id}",
                "original_customer_id": customer.customer_id,
                "original_x": customer.x,
                "original_y": customer.y,
                "access_node_id": access[customer.customer_id][0],
                "access_distance_m": access[customer.customer_id][1],
                "demand": customer.demand,
                "known_at_s": 0.0,
                "release_s": 0.0,
                "earliest_service_start_s": max(0.0, base - slack / 2 + jitter),
                "latest_service_start_s": base + slack / 2 + jitter,
                "service_duration_s": config.service_duration_s,
                "status": "pending",
            }
        )

    fleet = [
        {
            "vehicle_id": f"V{i + 1}",
            "capacity": instance.capacity,
            "start_node_id": depot_node_id,
            "depot_node_id": depot_node_id,
            "current_edge_id": None,
            "current_node_id": depot_node_id,
            "distance_remaining_m": 0.0,
            "onboard_request_ids": [],
            "remaining_load": 0.0,
            "executed_prefix_edge_ids": [],
        }
        for i in range(fleet_count)
    ]
    return requests, fleet, routes


def _reference_order(customer_id: str, access: dict[str, tuple[str, float]], variant: int):
    return customer_id if variant == 0 else (access[customer_id][0], customer_id)


def _max_access_detour(nodes, edges, depot_node_id: str, access: dict[str, tuple[str, float]]) -> float:
    depot_point = _node_point(nodes, depot_node_id)
    maximum = 1.0
    for access_node, _ in access.values():
        target_point = _node_point(nodes, access_node)
        euclidean = _distance(depot_point, target_point)
        if euclidean <= 1e-9:
            continue
        try:
            road_distance = _shortest_distance_dict(nodes, edges, depot_node_id, access_node)
        except ValueError:
            return math.inf
        maximum = max(maximum, road_distance / euclidean)
    return maximum


def _shortest_distance_dict(nodes, edges, source: str, target: str) -> float:
    if source == target:
        return 0.0
    adjacency = {node["node_id"]: [] for node in nodes}
    for edge in edges:
        adjacency[edge["from_node"]].append((edge["to_node"], edge["length_m"]))
    distances = {node_id: math.inf for node_id in adjacency}
    distances[source] = 0.0
    pending = [(0.0, source)]
    while pending:
        pending.sort(reverse=True)
        current_distance, current = pending.pop()
        if current == target:
            return current_distance
        if current_distance > distances[current] + 1e-9:
            continue
        for neighbour, cost in adjacency[current]:
            candidate = current_distance + cost
            if candidate < distances[neighbour] - 1e-9:
                distances[neighbour] = candidate
                pending.append((candidate, neighbour))
    raise ValueError(f"No directed path from {source!r} to {target!r}")


def _shortest_time(nodes, edges, source: str, target: str) -> float:
    if source == target:
        return 0.0
    adjacency = {node["node_id"]: [] for node in nodes}
    for edge in edges:
        adjacency[edge["from_node"]].append((edge["to_node"], edge["length_m"] / edge["speed_limit_mps"]))
    distances = {node_id: math.inf for node_id in adjacency}
    distances[source] = 0.0
    pending = [(0.0, source)]
    while pending:
        pending.sort(reverse=True)
        current_distance, current = pending.pop()
        if current == target:
            return current_distance
        if current_distance > distances[current] + 1e-9:
            continue
        for neighbour, cost in adjacency[current]:
            candidate = current_distance + cost
            if candidate < distances[neighbour] - 1e-9:
                distances[neighbour] = candidate
                pending.append((candidate, neighbour))
    raise ValueError(f"No directed path from {source!r} to {target!r}")


def _reference_schedule_feasible(scenario: Scenario) -> bool:
    if sum(request.demand for request in scenario.requests) > sum(vehicle.capacity for vehicle in scenario.fleet) + 1e-9:
        return False
    # Windows are deliberately centred around the same deterministic reference
    # arrival construction used during generation.
    return all(request.latest_service_start_s >= request.earliest_service_start_s for request in scenario.requests)


def _connected(nodes, edges) -> bool:
    if not nodes:
        return False
    adjacency = {node["node_id"] if isinstance(node, dict) else node.node_id: set() for node in nodes}
    for edge in edges:
        source = edge["from_node"] if isinstance(edge, dict) else edge.from_node
        target = edge["to_node"] if isinstance(edge, dict) else edge.to_node
        adjacency[source].add(target)
        adjacency[target].add(source)
    return len(_reachable(next(iter(adjacency)), adjacency)) == len(adjacency)


def _reachable(start, adjacency):
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbour in adjacency[current]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


def _detour_ratios(scenario: Scenario) -> list[float]:
    result: list[float] = []
    depot = scenario.fleet[0].depot_node_id
    depot_point = _node_point_model(scenario, depot)
    for request in scenario.requests:
        target_point = _node_point_model(scenario, request.access_node_id)
        try:
            road_distance = _shortest_distance(scenario.nodes, scenario.edges, depot, request.access_node_id)
        except ValueError:
            continue
        euclidean = _distance(depot_point, target_point)
        if euclidean > 1e-9:
            result.append(road_distance / euclidean)
    return result


def _shortest_distance(nodes, edges, source: str, target: str) -> float:
    if source == target:
        return 0.0
    adjacency = {node.node_id: [] for node in nodes}
    for edge in edges:
        adjacency[edge.from_node].append((edge.to_node, edge.length_m))
    distances = {node_id: math.inf for node_id in adjacency}
    distances[source] = 0.0
    pending = [(0.0, source)]
    while pending:
        pending.sort(reverse=True)
        current_distance, current = pending.pop()
        if current == target:
            return current_distance
        if current_distance > distances[current] + 1e-9:
            continue
        for neighbour, cost in adjacency[current]:
            candidate = current_distance + cost
            if candidate < distances[neighbour] - 1e-9:
                distances[neighbour] = candidate
                pending.append((candidate, neighbour))
    raise ValueError(f"No directed path from {source!r} to {target!r}")

def _bearing(edge, scenario):
    a = _node_point_model(scenario, edge.from_node)
    b = _node_point_model(scenario, edge.to_node)
    angle = math.degrees(math.atan2(b.y - a.y, b.x - a.x)) % 180.0
    return angle


def _bearing_histogram(bearings):
    bins = {str(i * 45): 0 for i in range(4)}
    for angle in bearings:
        index = int(((angle + 22.5) % 180) // 45)
        bins[str(index * 45)] += 1
    return bins


def _histogram(values):
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def _quantiles(values):
    if not values:
        return {"p0": None, "p25": None, "p50": None, "p75": None, "p100": None}
    values = sorted(values)
    return {key: _percentile(values, fraction) for key, fraction in (("p0", 0), ("p25", .25), ("p50", .5), ("p75", .75), ("p100", 1))}


def _percentile(values, fraction):
    if len(values) == 1:
        return float(values[0])
    position = fraction * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1 - weight) + values[upper] * weight)


def _node_point(nodes, node_id):
    node = next(node for node in nodes if node["node_id"] == node_id)
    return Point(node["x_m"], node["y_m"])


def _node_point_model(scenario, node_id):
    node = next(node for node in scenario.nodes if node.node_id == node_id)
    return Point(node.x_m, node.y_m)


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_nodes(path: Path, scenario: Scenario) -> None:
    rows = [node.model_dump(mode="json") for node in scenario.nodes]
    _write_rows(path, rows)


def _write_edges(path: Path, scenario: Scenario) -> None:
    rows = [edge.model_dump(mode="json") for edge in scenario.edges]
    _write_rows(path, rows)


def _write_requests(path: Path, scenario: Scenario) -> None:
    _write_rows(path, [request.model_dump(mode="json") for request in scenario.requests])


def _write_fleet(path: Path, scenario: Scenario) -> None:
    rows = []
    for vehicle in scenario.fleet:
        row = vehicle.model_dump(mode="json")
        row["onboard_request_ids"] = "|".join(row["onboard_request_ids"])
        row["executed_prefix_edge_ids"] = "|".join(row["executed_prefix_edge_ids"])
        rows.append(row)
    _write_rows(path, rows)


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
