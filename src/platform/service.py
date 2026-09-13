"""Single service used by the UI, runtime loop, and demo scripts.

This is a facade. It does not reimplement QPSO, validation, or SUMO.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Sequence

from src.contracts.scenario import Scenario
from src.optim.common import FitnessResult
from src.optim.pso import PSOConfig, ParticleSwarmOptimizer
from src.optim.qpso import AdaptiveQPSO, AdaptiveQPSOConfig, RouteFitnessOracle
from src.optim.references import (
    ALNSConfig,
    ALNSReference,
    ExactSolverUnavailable,
    MILPArc,
    MILPNode,
    MILPVehicle,
    ReferenceRoute,
    SnapshotCVRPTW,
    SnapshotMILPReference,
)
from src.platform.catalog import PROJECT_ROOT, list_scenarios, load_scenario
from src.platform.serialize import (
    evaluation_payload,
    plan_orders,
    scenario_payload,
    validator_payload,
)
from src.routing.initial_solution import InitialSolutionBuilder
from src.routing.path_builder import DirectedPathBuilder, PathNotFoundError
from src.routing.route_encoding import Step7RouteEngine
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_plan import RoutePlan, VehicleRoute
from src.routing.validator import evaluate_scenario, shortest_directed_path

REFERENCE_ORDERS = {"V1": ["J1", "J2", "J3"], "V2": ["J4", "J5"]}


@dataclass(frozen=True)
class SolveOptions:
    method: str = "qpso"
    particles: int = 12
    evaluations: int = 40
    seed: int = 7
    closed_edge_ids: tuple[str, ...] = ()


class PlatformService:
    """Load fixtures, validate, solve, compare, and launch SUMO."""

    def project_meta(self) -> dict:
        return {
            "title": "Quanta — Adaptive Quantum-Inspired Vehicle Routing",
            "ps_id": "SIH26137",
            "ps_name": (
                "Quantum-Inspired Intelligent Traffic Route Optimization "
                "in Transportation Systems Using Metaheuristic Optimization"
            ),
            "organization": "Egreen Quanta",
            "category": "Transportation & Logistics",
            "source": "https://sih2026.vuce.in/ps/SIH26137",
            "official_listing": "https://www.sih.gov.in/sih2026PS",
            "what_sih_wants": [
                "A software platform, not a slide-only algorithm.",
                "Transportation network as a weighted graph.",
                "Quantum-inspired optimizer (QPSO) on classical hardware.",
                "Near-optimal vehicle routes under simulated or live traffic.",
                "Minimize travel time, distance, and congestion.",
                "Constraint handling, convergence analysis, and benchmarks.",
                "Compare QPSO with classical metaheuristics and exact methods.",
                "Show VRP and shortest-path capability, plus scalability.",
            ],
            "what_we_build": [
                "Graph + CVRPTW model with an independent validator.",
                "Random-key QPSO with repair, plus matched PSO and ALNS.",
                "Tiny exact MILP and Dijkstra references.",
                "SUMO execution of the 5-job fixture, including a closure.",
                "This dispatcher UI for solve, compare, close-road, and replay.",
            ],
            "what_is_live": [
                "Steps 1–8: contracts, fixture, roads, evaluator, SUMO, QPSO/PSO.",
                "Step 9 references: ALNS and snapshot MILP in src/optim/references.py.",
                "This platform facade, runtime loop, and demo UI.",
            ],
            "what_is_labelled_baseline": [
                "Persistence forecast (copy last speeds) — GNN/Transformer not trained.",
                "Rule-based KEEP/GLOBAL scope — PPO not trained.",
            ],
        }

    def list_scenarios(self) -> list[dict]:
        return list_scenarios()

    def graph(self, scenario_id: str = "S3_BASE", closed_edge_ids: Sequence[str] = ()) -> dict:
        scenario = self._scenario(scenario_id, closed_edge_ids)
        payload = scenario_payload(scenario, closed_edge_ids=closed_edge_ids)
        payload["reference_plan"] = REFERENCE_ORDERS
        return payload

    def validate(
        self,
        scenario_id: str = "S3_BASE",
        plan: dict[str, list[str]] | None = None,
        closed_edge_ids: Sequence[str] = (),
    ) -> dict:
        scenario = self._scenario(scenario_id, closed_edge_ids)
        orders = plan or REFERENCE_ORDERS
        result = evaluate_scenario(scenario, orders, closed_edge_ids=closed_edge_ids)
        return {
            "scenario_id": scenario.scenario_id,
            "plan": orders,
            "validation": validator_payload(result),
        }

    def solve(
        self,
        options: SolveOptions | None = None,
        scenario_id: str = "S3_BASE",
        *,
        include_route_plan: bool = False,
    ) -> dict:
        options = options or SolveOptions()
        if options.evaluations < options.particles:
            options = SolveOptions(
                method=options.method,
                particles=options.particles,
                evaluations=options.particles,
                seed=options.seed,
                closed_edge_ids=options.closed_edge_ids,
            )
        method = options.method.lower()
        if method == "qpso":
            return self._solve_swarm(scenario_id, options, algorithm="qpso", include_route_plan=include_route_plan)
        if method == "pso":
            return self._solve_swarm(scenario_id, options, algorithm="pso", include_route_plan=include_route_plan)
        if method == "constructive":
            return self._solve_constructive(scenario_id, options)
        if method == "alns":
            return self._solve_alns(scenario_id, options)
        if method == "milp":
            return self._solve_milp(scenario_id, options)
        raise ValueError(f"Unknown method: {options.method}")

    def compare(
        self,
        scenario_id: str = "S3_BASE",
        *,
        particles: int = 12,
        evaluations: int = 40,
        seed: int = 7,
        closed_edge_ids: Sequence[str] = (),
        methods: Sequence[str] = ("constructive", "qpso", "pso", "alns"),
    ) -> dict:
        rows = []
        traces = {}
        for method in methods:
            try:
                result = self.solve(
                    SolveOptions(
                        method=method,
                        particles=particles,
                        evaluations=evaluations,
                        seed=seed,
                        closed_edge_ids=tuple(closed_edge_ids),
                    ),
                    scenario_id=scenario_id,
                )
            except Exception as exc:
                rows.append(
                    {
                        "method": method,
                        "error": str(exc),
                        "feasible": False,
                    }
                )
                continue
            rows.append(
                {
                    "method": result["method"],
                    "feasible": result["evaluation"]["feasible"],
                    "objective": result["evaluation"]["objective"],
                    "time_s": result["evaluation"]["time_s"],
                    "distance_m": result["evaluation"]["distance_m"],
                    "congestion_s": result["evaluation"]["congestion_s"],
                    "elapsed_s": result["elapsed_s"],
                    "evaluations": result.get("evaluations"),
                    "certified": result.get("certified"),
                    "gap": result.get("gap"),
                    "status": result.get("status"),
                }
            )
            if result.get("trace"):
                traces[result["method"]] = result["trace"]
        return {
            "scenario_id": scenario_id,
            "closed_edge_ids": list(closed_edge_ids),
            "rows": rows,
            "traces": traces,
        }

    def shortest_path(
        self,
        source: str,
        target: str,
        scenario_id: str = "S3_BASE",
        closed_edge_ids: Sequence[str] = (),
    ) -> dict:
        scenario = self._scenario(scenario_id, closed_edge_ids)
        started = time.perf_counter()
        try:
            path = DirectedPathBuilder(
                scenario.edges,
                closed_edge_ids=closed_edge_ids,
            ).shortest_path(source, target)
        except PathNotFoundError as exc:
            return {
                "method": "dijkstra",
                "exact": True,
                "feasible": False,
                "error": str(exc),
                "elapsed_s": time.perf_counter() - started,
            }
        return {
            "method": "dijkstra",
            "exact": True,
            "feasible": True,
            "source": source,
            "target": target,
            "node_ids": list(path.node_ids),
            "edge_ids": list(path.edge_ids),
            "distance_m": path.distance_m,
            "time_s": path.travel_time_s,
            "congestion_s": path.congestion_delay_s,
            "elapsed_s": time.perf_counter() - started,
        }

    def replay_sumo(self, scenario_id: str = "S3_BASE") -> dict:
        """Run the fixture in headless SUMO and return map frames for the UI."""
        _ensure_sumo_env()
        from src.sim.fcd_replay import parse_fcd
        from src.sim.fixture import fixture_plan
        from src.sim.sumo_runner import run_episode

        output = PROJECT_ROOT / "artifacts" / "demo_sumo_ui"
        output.mkdir(parents=True, exist_ok=True)
        scenario = load_scenario(scenario_id)
        episode = run_episode(scenario, fixture_plan(scenario), output, gui=False)
        fcd_path = output / "sumo_output" / "fcd.xml"
        if not fcd_path.is_file():
            raise RuntimeError("SUMO finished but wrote no FCD trace")
        frames = parse_fcd(fcd_path)
        return {
            "mode": "replay",
            "episode": episode.to_dict(),
            "frames": frames,
            "duration_s": frames[-1]["t"] if frames else 0.0,
            "message": (
                "SUMO replay ready. Green/blue dots are delivery vans; "
                "gray dots are background cars. E23 closes at t=50s."
            ),
        }

    def run_sumo(self, *, gui: bool = False) -> dict:
        output = PROJECT_ROOT / "artifacts" / "demo_sumo"
        output.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "src.sim.fixture",
            "--output",
            str(output),
        ]
        if gui:
            command.append("--gui")
            process = subprocess.Popen(command, cwd=str(PROJECT_ROOT))
            return {
                "mode": "gui",
                "pid": process.pid,
                "output_dir": str(output),
                "message": "SUMO-GUI launched. Watch the two delivery vehicles and the road closure.",
            }

        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        result_path = output / "episode_result.json"
        payload: dict[str, Any] = {
            "mode": "headless",
            "returncode": completed.returncode,
            "output_dir": str(output),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-2000:],
        }
        if result_path.is_file():
            payload["episode"] = result_path.read_text(encoding="utf-8")
        return payload

    def closeable_edges(self, scenario_id: str = "S3_BASE") -> list[dict]:
        scenario = load_scenario(scenario_id)
        return [
            {
                "id": edge.edge_id,
                "from": edge.from_node,
                "to": edge.to_node,
                "open": edge.open_by_default,
            }
            for edge in scenario.edges
        ]

    def _scenario(self, scenario_id: str, closed_edge_ids: Sequence[str] = ()) -> Scenario:
        scenario = load_scenario(scenario_id)
        closed = set(closed_edge_ids)
        if not closed:
            return scenario
        edges = tuple(
            edge.model_copy(update={"open_by_default": False})
            if edge.edge_id in closed
            else edge
            for edge in scenario.edges
        )
        return scenario.model_copy(
            update={
                "edges": edges,
                "graph_version": f"{scenario.graph_version}:closed:{','.join(sorted(closed))}",
            }
        )

    def _stack(
        self,
        scenario: Scenario,
        closed_edge_ids: Sequence[str] = (),
    ) -> tuple[RouteEvaluator, Step7RouteEngine]:
        evaluator = RouteEvaluator(scenario, closed_edge_ids=closed_edge_ids)
        return evaluator, Step7RouteEngine(scenario, evaluator)

    def _seed_population(
        self,
        scenario: Scenario,
        engine: Step7RouteEngine,
        evaluator: RouteEvaluator,
        options: SolveOptions,
    ) -> tuple[tuple[float, ...], ...]:
        rng = random.Random(options.seed)
        dimension = engine.encoder.dimension
        population: list[tuple[float, ...]] = []
        try:
            initial = InitialSolutionBuilder(scenario, evaluator).build()
            if initial.complete:
                encoded = engine.encoder.encode(initial.route_plan)
                population.append(encoded.keys)
        except Exception:
            pass
        while len(population) < options.particles:
            population.append(tuple(rng.random() for _ in range(dimension)))
        return tuple(population[: options.particles])

    def _solve_swarm(
        self,
        scenario_id: str,
        options: SolveOptions,
        *,
        algorithm: str,
        include_route_plan: bool = False,
    ) -> dict:
        scenario = self._scenario(scenario_id, options.closed_edge_ids)
        evaluator, engine = self._stack(scenario, options.closed_edge_ids)
        oracle = RouteFitnessOracle(engine, planning_time_s=0.0, repair=True)
        population = self._seed_population(scenario, engine, evaluator, options)
        started = time.perf_counter()
        if algorithm == "qpso":
            config = AdaptiveQPSOConfig(
                dimensions=engine.encoder.dimension,
                lower_bound=0.0,
                upper_bound=1.0,
                population_size=options.particles,
                max_evaluations=options.evaluations,
                seed=options.seed,
            )
            result = AdaptiveQPSO(
                config, oracle, initial_population=population
            ).optimize()
        else:
            config = PSOConfig(
                dimensions=engine.encoder.dimension,
                lower_bound=0.0,
                upper_bound=1.0,
                population_size=options.particles,
                max_evaluations=options.evaluations,
                seed=options.seed,
            )
            result = ParticleSwarmOptimizer(
                config, oracle, initial_population=population
            ).optimize()
        elapsed = time.perf_counter() - started
        candidate = engine.evaluate_keys(result.best_position, repair=True)
        evaluation = candidate.repaired_evaluation
        payload = {
            "method": algorithm.upper(),
            "status": "feasible" if evaluation.feasible else "no_feasible_incumbent",
            "elapsed_s": elapsed,
            "evaluations": result.evaluations,
            "iterations": result.iterations,
            "plan": plan_orders(evaluation),
            "evaluation": evaluation_payload(evaluation),
            "trace": {
                "best": list(result.history_best),
                "mean": list(result.history_mean),
                "diversity": list(result.history_route_diversity),
            },
            "closed_edge_ids": list(options.closed_edge_ids),
            "seed": options.seed,
        }
        if include_route_plan:
            payload["route_plan_object"] = evaluation.route_plan
            payload["evaluation_object"] = evaluation
        return payload

    def _solve_constructive(self, scenario_id: str, options: SolveOptions) -> dict:
        scenario = self._scenario(scenario_id, options.closed_edge_ids)
        evaluator, _engine = self._stack(scenario, options.closed_edge_ids)
        started = time.perf_counter()
        initial = InitialSolutionBuilder(scenario, evaluator).build()
        elapsed = time.perf_counter() - started
        return {
            "method": "CONSTRUCTIVE",
            "status": "feasible" if initial.evaluation.feasible else "incomplete",
            "elapsed_s": elapsed,
            "evaluations": initial.attempt_count,
            "plan": plan_orders(initial.evaluation),
            "evaluation": evaluation_payload(initial.evaluation),
            "trace": None,
            "closed_edge_ids": list(options.closed_edge_ids),
            "seed": options.seed,
        }

    def _id_maps(self, scenario: Scenario) -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
        customer_to_int = {
            job.request_id: index + 1
            for index, job in enumerate(scenario.requests)
        }
        vehicle_to_int = {
            vehicle.vehicle_id: index + 1
            for index, vehicle in enumerate(scenario.fleet)
        }
        return (
            customer_to_int,
            {value: key for key, value in customer_to_int.items()},
            vehicle_to_int,
            {value: key for key, value in vehicle_to_int.items()},
        )

    def _solve_alns(self, scenario_id: str, options: SolveOptions) -> dict:
        scenario = self._scenario(scenario_id, options.closed_edge_ids)
        evaluator, engine = self._stack(scenario, options.closed_edge_ids)
        c2i, i2c, v2i, i2v = self._id_maps(scenario)
        initial = InitialSolutionBuilder(scenario, evaluator).build()
        try:
            engine.encoder.encode(initial.route_plan)
            start_routes = {
                v2i[route.vehicle_id]: [c2i[cid] for cid in route.customer_ids]
                for route in initial.route_plan.vehicle_routes
            }
        except Exception:
            start_routes = None

        def alns_evaluator(route: ReferenceRoute) -> FitnessResult:
            orders = {vid: [] for vid in v2i}
            for vehicle_int, customers in route.vehicles:
                orders[i2v[vehicle_int]] = [i2c[cid] for cid in customers]
            try:
                encoded = engine.encoder.encode(
                    RoutePlan.from_routes(
                        [
                            VehicleRoute.from_sequence(vid, customers)
                            for vid, customers in orders.items()
                        ]
                    )
                )
                evaluation = engine.evaluate_keys(
                    encoded.keys, repair=True
                ).repaired_evaluation
                return FitnessResult(
                    fitness=float(evaluation.objective_value),
                    feasible=bool(evaluation.feasible),
                    route_signature=route.vehicles,
                )
            except Exception:
                checked = evaluate_scenario(
                    scenario, orders, closed_edge_ids=options.closed_edge_ids
                )
                elapsed = sum(veh.elapsed_time_s for veh in checked.vehicles.values())
                return FitnessResult(
                    fitness=elapsed if checked.feasible else elapsed + 1_000_000.0,
                    feasible=checked.feasible,
                    route_signature=route.vehicles,
                )

        started = time.perf_counter()
        result = ALNSReference(
            evaluator=alns_evaluator,
            customer_ids=list(c2i.values()),
            vehicle_ids=list(v2i.values()),
            initial_route=start_routes,
            config=ALNSConfig(
                max_evaluations=max(30, options.evaluations),
                time_limit_s=3.0,
            ),
            seed=options.seed,
        ).run()
        elapsed = time.perf_counter() - started
        if result.route is None:
            evaluation = initial.evaluation
        else:
            orders = {vid: [] for vid in v2i}
            for vehicle_int, customers in result.route.vehicles:
                orders[i2v[vehicle_int]] = [i2c[cid] for cid in customers]
            encoded = engine.encoder.encode(
                RoutePlan.from_routes(
                    [
                        VehicleRoute.from_sequence(vid, customers)
                        for vid, customers in orders.items()
                    ]
                )
            )
            evaluation = engine.evaluate_keys(encoded.keys, repair=True).repaired_evaluation
        return {
            "method": "ALNS",
            "status": "feasible" if evaluation.feasible else "no_feasible_incumbent",
            "elapsed_s": elapsed,
            "evaluations": result.evaluations,
            "plan": plan_orders(evaluation),
            "evaluation": evaluation_payload(evaluation),
            "trace": None,
            "closed_edge_ids": list(options.closed_edge_ids),
            "seed": options.seed,
        }

    def _solve_milp(self, scenario_id: str, options: SolveOptions) -> dict:
        scenario = self._scenario(scenario_id, options.closed_edge_ids)
        evaluator, engine = self._stack(scenario, options.closed_edge_ids)
        problem = self._snapshot_problem(scenario, options.closed_edge_ids)
        started = time.perf_counter()
        try:
            result = SnapshotMILPReference(
                problem=problem,
                time_limit_s=8.0,
                mip_gap=1e-4,
            ).solve()
        except ExactSolverUnavailable as exc:
            fallback = self._solve_constructive(scenario_id, options)
            fallback["method"] = "MILP"
            fallback["status"] = "unavailable"
            fallback["error"] = str(exc)
            return fallback
        elapsed = time.perf_counter() - started
        _c2i, i2c, _v2i, i2v = self._id_maps(scenario)
        if result.route is None:
            evaluation = InitialSolutionBuilder(scenario, evaluator).build().evaluation
        else:
            orders = {vehicle.vehicle_id: [] for vehicle in scenario.fleet}
            for vehicle_int, customers in result.route.vehicles:
                orders[i2v[vehicle_int]] = [i2c[cid] for cid in customers]
            encoded = engine.encoder.encode(
                RoutePlan.from_routes(
                    [
                        VehicleRoute.from_sequence(vid, customers)
                        for vid, customers in orders.items()
                    ]
                )
            )
            evaluation = engine.evaluate_keys(encoded.keys, repair=True).repaired_evaluation
        return {
            "method": "MILP",
            "status": str(result.status),
            "certified": bool(getattr(result, "certified_optimum", False)),
            "gap": getattr(result, "relative_gap", None),
            "lower_bound": result.lower_bound,
            "elapsed_s": elapsed,
            "evaluations": None,
            "plan": plan_orders(evaluation),
            "evaluation": evaluation_payload(evaluation),
            "trace": None,
            "closed_edge_ids": list(options.closed_edge_ids),
            "seed": options.seed,
            "note": (
                "Certified optimum only if the solver closed the gap. "
                "Otherwise this is a bound/reference, not a proven optimum."
            ),
        }

    def _snapshot_problem(
        self,
        scenario: Scenario,
        closed_edge_ids: Sequence[str],
    ) -> SnapshotCVRPTW:
        c2i, _i2c, v2i, _i2v = self._id_maps(scenario)
        depot = scenario.fleet[0].depot_node_id
        depot_start = 0
        depot_end = len(scenario.requests) + 1
        customers = []
        access = {depot_start: depot, depot_end: depot}
        for job in scenario.requests:
            node_id = c2i[job.request_id]
            access[node_id] = job.access_node_id
            customers.append(
                MILPNode(
                    node_id=node_id,
                    demand=float(job.demand),
                    service_time=float(job.service_duration_s),
                    earliest=float(job.earliest_service_start_s),
                    latest=float(job.latest_service_start_s),
                )
            )
        nodes = [depot_start, *[c2i[job.request_id] for job in scenario.requests], depot_end]
        arcs = []
        for origin in nodes:
            for destination in nodes:
                if origin == destination:
                    continue
                if origin == depot_end or destination == depot_start:
                    continue
                path = shortest_directed_path(
                    scenario,
                    access[origin],
                    access[destination],
                    closed_edge_ids,
                )
                if path is None:
                    continue
                _edges, travel = path
                distance = 0.0
                edge_lookup = {edge.edge_id: edge for edge in scenario.edges}
                for edge_id in _edges:
                    distance += float(edge_lookup[edge_id].length_m)
                arcs.append(
                    MILPArc(
                        origin=origin,
                        destination=destination,
                        travel_time=float(travel),
                        distance=distance,
                    )
                )
        return SnapshotCVRPTW(
            depot_start=depot_start,
            depot_end=depot_end,
            customers=tuple(customers),
            vehicles=tuple(
                MILPVehicle(vehicle_id=v2i[vehicle.vehicle_id], capacity=float(vehicle.capacity))
                for vehicle in scenario.fleet
            ),
            arcs=tuple(arcs),
        )


def _ensure_sumo_env() -> None:
    """Make SUMO visible to the API process on a typical Windows install."""
    existing = os.environ.get("SUMO_HOME")
    if existing and Path(existing, "bin").is_dir():
        return
    candidates = [
        Path(r"C:\Program Files (x86)\Eclipse\Sumo"),
        Path(r"C:\Program Files\Eclipse\Sumo"),
    ]
    for home in candidates:
        if (home / "bin" / "sumo.exe").is_file() or (home / "bin" / "sumo").is_file():
            os.environ["SUMO_HOME"] = str(home)
            os.environ["PATH"] = str(home / "bin") + os.pathsep + os.environ.get("PATH", "")
            return
