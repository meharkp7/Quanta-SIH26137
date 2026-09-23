"""Single service used by the UI, runtime loop, and demo scripts.

This is a facade. It does not reimplement QPSO, validation, or SUMO.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
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
    is_geo_scenario,
    plan_orders,
    scenario_payload,
    to_latlon,
    validator_payload,
)
from src.routing.initial_solution import (
    InitialSolutionBuilder,
    InitialSolutionConfig,
)
from src.routing.path_builder import DirectedPathBuilder, PathNotFoundError
from src.routing.route_encoding import Step7RouteEngine
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_plan import RoutePlan, VehicleRoute
from src.routing.validator import evaluate_scenario, shortest_directed_path

REFERENCE_ORDERS = {"V1": ["J1", "J2", "J3"], "V2": ["J4", "J5"]}


@dataclass(frozen=True)
class MockEpisodeResult:
    """Honest kinematic stand-in for a SUMO episode on large/geo networks.

    The committed plan is still the solver's validator-approved plan; only
    the microscopic lane/signal execution (SUMO) is replaced by a summary
    of what the plan would deliver when run to completion on the open
    network. `mode` is always "kinematic_mock" so callers can label it.
    """

    scenario_id: str
    duration_s: float
    total_steps: int
    delivered_count: int
    pending_count: int
    teleport_events: int
    mode: str = "kinematic_mock"

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "total_steps": self.total_steps,
            "duration_s": self.duration_s,
            "delivered_count": self.delivered_count,
            "pending_count": self.pending_count,
            "teleport_events": self.teleport_events,
            "mode": self.mode,
            "summary": (
                f"Kinematic mock of {self.duration_s:.0f}s: all "
                f"{self.delivered_count} deliveries complete on the open "
                "network; SUMO executes the 5-job fixture only."
            ),
        }


@dataclass(frozen=True)
class SolveOptions:
    method: str = "qpso"
    particles: int = 12
    evaluations: int = 40
    seed: int = 7
    closed_edge_ids: tuple[str, ...] = ()


class PlatformService:
    """Load fixtures, validate, solve, compare, and launch SUMO."""

    # In-memory solve cache: identical repeat solves (e.g. repeat demo
    # clicks) return instantly with `cached: true`. Capped at 128 entries
    # (FIFO eviction), no persistence.
    SOLVE_CACHE_MAX = 128
    _solve_cache: dict[tuple, dict] = {}

    @classmethod
    def _cache_key(
        cls,
        scenario_id: str,
        options: "SolveOptions",
    ) -> tuple:
        return (
            scenario_id,
            options.method.lower(),
            options.particles,
            options.evaluations,
            options.seed,
            tuple(sorted(options.closed_edge_ids)),
        )

    @classmethod
    def solve_cache_info(cls) -> dict:
        return {"entries": len(cls._solve_cache), "max_entries": cls.SOLVE_CACHE_MAX}

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

    # Networks above this edge count get a capped optimizer budget so the
    # UI never hangs on a multi-thousand-edge real-city graph. The cap is
    # reported back as `budget_note`; callers asking for less keep theirs.
    LARGE_GRAPH_EDGES = 1000
    LARGE_GRAPH_PARTICLES = 6
    LARGE_GRAPH_EVALUATIONS = 24

    # Large real-city graphs (the ~2000-edge Delhi OSM maps) previously
    # spent ~135 s per solve in exhaustive constructive insertion scanning
    # — every customer x vehicle x position got a full plan re-evaluation.
    # Bounded scanning keeps the same evaluator/validator truth but stops
    # each insertion round after a handful of feasible candidates, cutting
    # the initial build to a few seconds. Small fixtures and the step-4
    # acceptance tests keep the exhaustive default (config=None).
    INITIAL_FEASIBLE_TARGET = 8
    INITIAL_TRIALS_CAP = 300

    @classmethod
    def is_mock_scenario(cls, scenario_or_id) -> bool:
        """True when the network runs on the kinematic mock, not SUMO.

        SUMO execution (replay + loop) is reserved for the small step-3
        fixture; large or geo-projected graphs get the mock.
        """
        scenario = (
            load_scenario(scenario_or_id)
            if isinstance(scenario_or_id, str)
            else scenario_or_id
        )
        return (
            len(scenario.edges) > cls.LARGE_GRAPH_EDGES
            or is_geo_scenario(scenario)
        )

    @classmethod
    def _initial_config(cls, scenario) -> InitialSolutionConfig:
        if cls.is_mock_scenario(scenario):
            return InitialSolutionConfig(
                feasible_candidates_per_round=cls.INITIAL_FEASIBLE_TARGET,
                max_trials_per_round=cls.INITIAL_TRIALS_CAP,
            )
        return InitialSolutionConfig()

    @classmethod
    def _initial_builder(cls, scenario, evaluator) -> InitialSolutionBuilder:
        return InitialSolutionBuilder(
            scenario, evaluator, config=cls._initial_config(scenario)
        )

    def mock_episode(self, scenario, duration_s: float) -> MockEpisodeResult:
        return MockEpisodeResult(
            scenario_id=scenario.scenario_id,
            duration_s=float(duration_s),
            total_steps=max(1, int(duration_s)),
            delivered_count=len(scenario.requests),
            pending_count=0,
            teleport_events=0,
        )

    def _budget_for(self, scenario_id: str, options: SolveOptions) -> tuple[SolveOptions, str | None]:
        try:
            edge_count = len(load_scenario(scenario_id).edges)
        except Exception:
            return options, None
        if edge_count <= self.LARGE_GRAPH_EDGES:
            return options, None
        capped = SolveOptions(
            method=options.method,
            particles=min(options.particles, self.LARGE_GRAPH_PARTICLES),
            evaluations=min(options.evaluations, self.LARGE_GRAPH_EVALUATIONS),
            seed=options.seed,
            closed_edge_ids=options.closed_edge_ids,
        )
        if (capped.particles, capped.evaluations) == (options.particles, options.evaluations):
            return options, None
        return capped, (
            f"Large network ({edge_count} edges): budget capped to "
            f"{capped.particles} particles x {capped.evaluations} evaluations."
        )

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
        options, budget_note = self._budget_for(scenario_id, options)
        if not include_route_plan:
            hit = self._solve_cache.get(self._cache_key(scenario_id, options))
            if hit is not None:
                replay = dict(hit)
                replay["cached"] = True
                return replay
        method = options.method.lower()
        if method == "qpso":
            result = self._solve_swarm(scenario_id, options, algorithm="qpso", include_route_plan=include_route_plan)
        elif method == "pso":
            result = self._solve_swarm(scenario_id, options, algorithm="pso", include_route_plan=include_route_plan)
        elif method == "constructive":
            result = self._solve_constructive(scenario_id, options)
        elif method == "alns":
            result = self._solve_alns(scenario_id, options)
        elif method == "milp":
            result = self._solve_milp(scenario_id, options)
        else:
            raise ValueError(f"Unknown method: {options.method}")
        if budget_note:
            result["budget_note"] = budget_note
        result["cached"] = False
        # Cache only plain JSON-friendly solve payloads (the runtime loop
        # uses include_route_plan=True with live objects — never cached).
        if not include_route_plan:
            key = self._cache_key(scenario_id, options)
            if len(self._solve_cache) >= self.SOLVE_CACHE_MAX and key not in self._solve_cache:
                self._solve_cache.pop(next(iter(self._solve_cache)))
            self._solve_cache[key] = dict(result)
        return result

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

    def replay_sumo(
        self,
        scenario_id: str = "S3_BASE",
        plan: dict[str, list[str]] | None = None,
        closed_edge_ids: Sequence[str] = (),
    ) -> dict:
        """Return map frames for the UI.

        The small step-3 fixture runs in headless SUMO. Large or
        geo-projected networks (the Delhi OSM maps) are served by an honest
        kinematic mock: vehicles play back the validator-approved directed
        paths in lat/lon, closures are enforced per frame, and no SUMO or
        external key is involved.
        """
        scenario = self._scenario(scenario_id, closed_edge_ids)
        if self.is_mock_scenario(scenario):
            return self._mock_replay(
                scenario,
                scenario_id=scenario_id,
                plan=plan,
                closed_edge_ids=closed_edge_ids,
            )
        _ensure_sumo_env()
        from src.sim.fcd_replay import parse_fcd
        from src.sim.sumo_runner import run_episode

        output = PROJECT_ROOT / "artifacts" / "demo_sumo_ui"
        output.mkdir(parents=True, exist_ok=True)
        from src.routing.validator import evaluate_scenario as _evaluate_orders

        if plan:
            orders = {
                vehicle_id: list(stops)
                for vehicle_id, stops in plan.items()
            }
            source = "requested plan"
        else:
            orders = dict(REFERENCE_ORDERS)
            source = "reference plan"
        verdict = _evaluate_orders(scenario, orders)
        if not verdict.feasible and not plan:
            # The reference route can break under UI closures (e.g. a closed
            # road disconnects it). Fall back to a fresh constructive solve
            # under the same closures instead of crashing SUMO.
            fallback = self.solve(
                SolveOptions(
                    method="constructive",
                    particles=2,
                    evaluations=4,
                    seed=7,
                    closed_edge_ids=tuple(closed_edge_ids),
                ),
                scenario_id=scenario_id,
            )
            orders = {
                vehicle_id: list(stops)
                for vehicle_id, stops in fallback["plan"].items()
            }
            source = "constructive fallback plan"
            verdict = _evaluate_orders(scenario, orders)
        if not verdict.feasible:
            detail = ""
            try:
                unserved = list(getattr(verdict, "unserved_request_ids", None) or [])
                stranded = list(getattr(verdict, "disconnected_legs", None) or [])
                bits = []
                if unserved:
                    bits.append(f"unserved customers: {', '.join(map(str, unserved))}")
                if stranded:
                    bits.append(f"disconnected legs: {', '.join(map(str, stranded))}")
                if bits:
                    detail = " " + "; ".join(bits) + "."
            except Exception:
                detail = ""
            raise ValueError(
                f"The {source} is infeasible under the selected closures; "
                "solve a validated plan first." + detail
            )
        from src.sim.incidents import closure_schedule, configs_for_scenario

        incident_configs = configs_for_scenario(scenario)
        timed = [
            cfg
            for cfg in incident_configs
            if cfg.trigger_time_s > 0
        ]

        def _build_executable(current_orders):
            evaluator, engine = self._stack(scenario, closed_edge_ids)
            logical_plan = RoutePlan.from_routes(
                [VehicleRoute.from_sequence(vehicle.vehicle_id, current_orders.get(vehicle.vehicle_id, ())) for vehicle in scenario.fleet]
            )
            encoded = engine.encoder.encode(logical_plan)
            candidate = engine.evaluate_keys(encoded.keys, repair=True)
            if not candidate.repaired_evaluation.feasible:
                raise ValueError("The requested replay plan failed independent validation")
            from src.runtime.loop import DemoLoop
            return DemoLoop._to_executable_plan(
                candidate.repaired_evaluation,
                scenario_id=scenario_id,
                state_version=f"{scenario_id}:replay",
                route_version=f"ui-replay-{int(time.time())}",
            )

        notes: list[str] = []
        executable_plan = _build_executable(orders)
        if timed:
            # Beat-the-clock gate: a timed incident (e.g. E23 at t=50s) kills
            # any plan that arrives after the trigger. entry times use
            # length-proportional splits inside each leg with a 2 s margin.
            lengths = {
                edge.edge_id: float(edge.length_m)
                for edge in scenario.edges
            }

            def _late_edges(plan) -> list[tuple[str, float, float]]:
                late = []
                for route in plan.vehicle_routes:
                    for leg in route.legs:
                        total = sum(
                            lengths.get(eid, 0.0)
                            for eid in leg.physical_edge_ids
                        )
                        cursor = float(leg.departure_time_s)
                        for eid in leg.physical_edge_ids:
                            span = (
                                lengths.get(eid, 0.0) / total
                                * float(leg.travel_time_s)
                                if total > 0
                                else 0.0
                            )
                            for cfg in timed:
                                if (
                                    eid == cfg.edge_id
                                    and cursor >= cfg.trigger_time_s - 2.0
                                ):
                                    late.append(
                                        (eid, cursor, cfg.trigger_time_s)
                                    )
                            cursor += span
                return late

            late = _late_edges(executable_plan)
            if late:
                beaten = sorted({eid for eid, _, _ in late})
                extended = tuple(
                    dict.fromkeys(
                        list(closed_edge_ids) + beaten
                    )
                )
                retry = self.solve(
                    SolveOptions(
                        method="constructive",
                        particles=2,
                        evaluations=4,
                        seed=7,
                        closed_edge_ids=extended,
                    ),
                    scenario_id=scenario_id,
                )
                retry_orders = {
                    vehicle_id: list(stops)
                    for vehicle_id, stops in retry["plan"].items()
                }
                if _evaluate_orders(scenario, retry_orders).feasible:
                    retry_plan = _build_executable(retry_orders)
                    if not _late_edges(retry_plan):
                        executable_plan = retry_plan
                        orders = retry_orders
                        notes.append(
                            "Replay plan rerouted around "
                            + ", ".join(beaten)
                            + " to beat the incident clock."
                        )
                    else:
                        raise ValueError(
                            "The incident clock beats every feasible plan: "
                            + ", ".join(
                                f"{eid} needed at t={entry:.0f}s "
                                f"but closes at t={trig:.0f}s"
                                for eid, entry, trig in late
                            )
                            + ". Choose a different incident."
                        )
                else:
                    raise ValueError(
                        "The incident clock beats every feasible plan: "
                        + ", ".join(
                            f"{eid} needed at t={entry:.0f}s "
                            f"but closes at t={trig:.0f}s"
                            for eid, entry, trig in late
                        )
                        + ". Choose a different incident."
                    )
        episode = run_episode(scenario, executable_plan, output, gui=False)
        fcd_path = output / "sumo_output" / "fcd.xml"
        if not fcd_path.is_file():
            raise RuntimeError("SUMO finished but wrote no FCD trace")
        from src.sim.incidents import closure_schedule

        schedule = closure_schedule(incident_configs)
        frames = parse_fcd(
            fcd_path, closures=tuple(schedule) if schedule else ()
        )
        if schedule:
            incident_bits = ", ".join(
                f"{item['edge_id']} closes at t={item['from_s']:.0f}s"
                for item in schedule
            )
        else:
            incident_bits = "no closures scheduled"
        return {
            "mode": "replay",
            "episode": episode.to_dict(),
            "frames": frames,
            "duration_s": frames[-1]["t"] if frames else 0.0,
            "incidents": [
                {
                    "incident_id": cfg.incident_id,
                    "edge_id": cfg.edge_id,
                    "trigger_time_s": cfg.trigger_time_s,
                }
                for cfg in incident_configs
            ],
            "message": (
                "SUMO replay ready. Green/blue dots are delivery vans; "
                f"gray dots are background cars. {incident_bits}."
                + (" " + " ".join(notes) if notes else "")
            ),
            "notes": notes,
        }

    def _mock_replay(
        self,
        scenario: Scenario,
        *,
        scenario_id: str,
        plan: dict[str, list[str]] | None = None,
        closed_edge_ids: Sequence[str] = (),
    ) -> dict:
        """Kinematic mock episode for large/geo networks (no SUMO, no keys).

        Vehicles play back the independent validator's directed paths on the
        real graph: per-leg travel time is split across its edges by
        free-flow proportion, waits/services become stops at nodes, and the
        result is sampled into <=600 frames carrying x/y metres plus lat/lon
        for the tile map. One deterministic mid-run road blockage is picked
        from used edges whose entries all finish before the trigger, so the
        "vehicles already on the link clear it" rule holds and the plan
        stays legal. UI-selected closures are enforced from t=0.
        """
        notes: list[str] = []
        fleet_ids = [vehicle.vehicle_id for vehicle in scenario.fleet]

        if plan:
            orders = {
                vid: [str(x) for x in plan.get(vid, [])]
                for vid in fleet_ids
            }
            source = "requested plan"
        else:
            fallback = self.solve(
                SolveOptions(
                    method="constructive",
                    particles=2,
                    evaluations=4,
                    seed=7,
                    closed_edge_ids=tuple(closed_edge_ids),
                ),
                scenario_id=scenario_id,
            )
            orders = {
                vid: [str(x) for x in fallback["plan"].get(vid, [])]
                for vid in fleet_ids
            }
            source = "constructive plan"
        verdict = evaluate_scenario(
            scenario, orders, closed_edge_ids=closed_edge_ids
        )
        if not verdict.feasible:
            detail = ""
            try:
                unserved = list(getattr(verdict, "unserved_request_ids", None) or [])
                stranded = list(getattr(verdict, "disconnected_legs", None) or [])
                bits = []
                if unserved:
                    bits.append(f"unserved customers: {', '.join(map(str, unserved))}")
                if stranded:
                    bits.append(f"disconnected legs: {', '.join(map(str, stranded))}")
                if bits:
                    detail = " " + "; ".join(bits) + "."
            except Exception:
                detail = ""
            raise ValueError(
                f"The {source} is infeasible under the selected closures; "
                "solve a validated plan first." + detail
            )

        geo = is_geo_scenario(scenario)
        node_pos: dict[str, tuple[float, float, float | None, float | None]] = {}
        for node in scenario.nodes:
            lat = lon = None
            if geo:
                converted = to_latlon(node.x_m, node.y_m)
                if converted is not None:
                    lat, lon = converted
            node_pos[str(node.node_id)] = (
                float(node.x_m), float(node.y_m), lat, lon
            )
        edge_ff = {
            str(edge.edge_id): float(edge.free_flow_time_s)
            for edge in scenario.edges
        }
        edge_ends = {
            str(edge.edge_id): (str(edge.from_node), str(edge.to_node))
            for edge in scenario.edges
        }

        # --- per-vehicle timeline: (t0, t1, kind, payload) segments ---
        timelines: dict[str, list[tuple[float, float, str, str]]] = {}
        entries: dict[str, list[float]] = {}
        durations: dict[str, float] = {}
        depot_of = {
            vehicle.vehicle_id: str(vehicle.depot_node_id)
            for vehicle in scenario.fleet
        }
        for vid in fleet_ids:
            veh = verdict.vehicles.get(vid)
            segments: list[tuple[float, float, str, str]] = []
            cursor = 0.0

            def _drive(leg, start: float) -> float:
                edge_ids = [str(e) for e in leg.edge_ids]
                weights = [edge_ff.get(e, 1.0) for e in edge_ids]
                total = sum(weights)
                at = start
                for eid, w in zip(edge_ids, weights):
                    span = (
                        float(leg.travel_time_s) * (w / total)
                        if total > 0
                        else float(leg.travel_time_s) / max(1, len(edge_ids))
                    )
                    if span > 0:
                        segments.append((at, at + span, "edge", eid))
                        entries.setdefault(eid, []).append(at)
                        at += span
                return at

            if veh is not None:
                leg_iter = iter(veh.legs)
                for jid in veh.customer_order:
                    leg = next(leg_iter, None)
                    if leg is None:
                        break
                    cursor = _drive(leg, cursor)
                    start = float(veh.service_starts_s.get(jid, cursor))
                    end = float(veh.service_ends_s.get(jid, start))
                    stop_end = max(end, cursor, start)
                    segments.append((cursor, stop_end, "stop", str(leg.to_node)))
                    cursor = stop_end
                for leg in leg_iter:  # final depot return (or empty)
                    cursor = _drive(leg, cursor)
                    if leg.edge_ids:
                        segments.append(
                            (cursor, cursor, "stop", str(leg.to_node))
                        )
            timelines[vid] = segments
            durations[vid] = cursor if veh is not None else 0.0

        duration = max(durations.values(), default=0.0)
        if duration < 1.0:
            duration = 60.0

        def _locate(vid: str, t: float) -> tuple[float, float, float | None, float | None, bool]:
            segments = timelines[vid]
            if not segments:
                x, y, lat, lon = node_pos[depot_of[vid]]
                return x, y, lat, lon, True
            for t0, t1, kind, payload in segments:
                if t0 <= t < t1:
                    if kind == "stop":
                        x, y, lat, lon = node_pos[payload]
                        return x, y, lat, lon, True
                    frm, to = edge_ends.get(payload, (payload, payload))
                    fx, fy, flat, flon = node_pos[frm]
                    tx, ty, tlat, tlon = node_pos[to]
                    frac = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
                    lat = (
                        flat + (tlat - flat) * frac
                        if flat is not None and tlat is not None
                        else None
                    )
                    lon = (
                        flon + (tlon - flon) * frac
                        if flon is not None and tlon is not None
                        else None
                    )
                    return (
                        fx + (tx - fx) * frac,
                        fy + (ty - fy) * frac,
                        lat,
                        lon,
                        False,
                    )
            t0, t1, kind, payload = segments[-1]
            if kind == "edge":
                payload = edge_ends.get(payload, (payload, payload))[1]
            x, y, lat, lon = node_pos[payload]
            return x, y, lat, lon, True

        # --- mid-run blockage demo: only when the user picked no incident.
        # A user-selected road IS the incident (enforced from t=0 above) —
        # no extra hardcoded blockage is ever stacked on top of it.
        closed0 = [str(e) for e in closed_edge_ids]
        trigger = min(60.0, max(15.0, duration * 0.35))
        timed: dict | None = None
        closed0_set = set(closed0)
        candidates: list[tuple[float, str]] = []
        if not closed0:
            for eid, ets in entries.items():
                if eid in closed0_set:
                    continue
                if not ets or max(ets) >= trigger - 1.0:
                    continue  # someone would enter after the blockage
                candidates.append((abs(max(ets) - (trigger - 5.0)), eid))
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]))
                timed = {
                    "incident_id": "INC_MOCK_SCHEDULED_BLOCKAGE",
                    "edge_id": candidates[0][1],
                    "trigger_time_s": trigger,
                }
        incidents = [
            {
                "incident_id": f"INC_UI_CLOSURE_{eid}",
                "edge_id": eid,
                "trigger_time_s": 0.0,
            }
            for eid in sorted(closed0_set)
        ]
        if timed:
            incidents.append(timed)

        # --- sample frames ---
        step = max(1.0, duration / 600.0)
        count = int(duration / step) + 1
        frames: list[dict] = []
        for k in range(count):
            t = min(k * step, duration)
            vehicles = []
            for vid in fleet_ids:
                x, y, lat, lon, stopped = _locate(vid, t)
                entry: dict = {
                    "id": vid,
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "kind": "delivery",
                    "stopped": stopped,
                }
                if lat is not None and lon is not None:
                    entry["lat"] = round(lat, 6)
                    entry["lon"] = round(lon, 6)
                vehicles.append(entry)
            frame_closed = list(closed0)
            if timed and t >= trigger:
                frame_closed.append(timed["edge_id"])
            frames.append({"t": round(t, 1), "vehicles": vehicles, "closed": frame_closed})

        if timed:
            incident_bits = (
                f"{timed['edge_id']} blocks at t={trigger:.0f}s "
                "(vehicles already on it clear it)"
            )
        elif closed0:
            incident_bits = (
                f"selected incident: {len(closed0)} road(s) blocked from "
                "t=0 — no extra blockage added"
            )
        else:
            incident_bits = "no mid-run blockage candidate on the plan"
        notes.append(
            f"Kinematic mock (no SUMO): {len(fleet_ids)} vans replay the "
            f"validator-approved plan over {duration:.0f}s on the real "
            f"{len(scenario.edges)}-edge graph; {incident_bits}."
        )
        return {
            "mode": "kinematic_mock",
            "episode": self.mock_episode(scenario, duration).to_dict(),
            "frames": frames,
            "duration_s": duration,
            "incidents": incidents,
            "message": (
                "Mock replay ready: vehicles follow the validated directed "
                "paths in lat/lon on the real road graph; road blockages are "
                "enforced per frame. SUMO executes only the 5-job fixture, "
                "and no external map key is used."
            ),
            "notes": notes,
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
            initial = self._initial_builder(scenario, evaluator).build()
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
        initial = self._initial_builder(scenario, evaluator).build()
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
        initial = self._initial_builder(scenario, evaluator).build()
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
            evaluation = self._initial_builder(scenario, evaluator).build().evaluation
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

    # Joint GNN-Transformer forecaster training on artifacts/corpus_v2.
    #
    # The Step 13 model is one joint network (CausalGNNTransformer: two
    # directed edge-aware GNN layers for spatial encoding followed by a
    # per-road temporal Transformer). "GNN and Transformer" therefore means
    # training this single forecaster, not two separate models. Training is
    # launched as a background process so the API never blocks; progress is
    # tracked via artifacts/forecaster_v2/training_status.json.
    FORECASTER_CORPUS_DIR = PROJECT_ROOT / "artifacts" / "corpus_v2"
    FORECASTER_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "forecaster_v2"
    FORECASTER_LEGACY_DIR = PROJECT_ROOT / "artifacts" / "step13_forecaster_v1"
    FORECASTER_STATUS_FILE = "training_status.json"

    def forecaster_artifact_dirs(self) -> list[Path]:
        return [self.FORECASTER_OUTPUT_DIR, self.FORECASTER_LEGACY_DIR]

    def forecaster_status(self) -> dict:
        corpus = self.FORECASTER_CORPUS_DIR
        manifest_path = corpus / "corpus_manifest.json"
        corpus_info: dict[str, Any] = {
            "corpus_dir": str(corpus),
            "present": corpus.is_dir(),
            "manifest_present": manifest_path.is_file(),
        }
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                corpus_info["episodes"] = {
                    split: len(manifest.get(split, []))
                    for split in ("train", "validation", "test")
                }
                corpus_info["schema_version"] = manifest.get("schema_version")
                corpus_info["map_disjoint"] = manifest.get("map_disjoint")
            except Exception as exc:
                corpus_info["manifest_error"] = str(exc)
        cache_dir = corpus / ".window_cache"
        if cache_dir.is_dir():
            npz_files = list(cache_dir.glob("*.windows.npz"))
            corpus_info["window_cache"] = {
                "dir": str(cache_dir),
                "cached_episodes": len(npz_files),
            }
        else:
            corpus_info["window_cache"] = {"dir": str(cache_dir), "cached_episodes": 0}

        artifacts: dict[str, Any] = {}
        for directory in self.forecaster_artifact_dirs():
            entry: dict[str, Any] = {"dir": str(directory), "present": directory.is_dir()}
            manifest_file = directory / "manifest.json"
            entry["manifest_present"] = manifest_file.is_file()
            if manifest_file.is_file():
                try:
                    saved = json.loads(manifest_file.read_text(encoding="utf-8"))
                    entry["artifact"] = saved.get("artifact")
                    entry["metrics"] = saved.get("metrics")
                    entry["early_stopping"] = saved.get("early_stopping")
                except Exception as exc:
                    entry["manifest_error"] = str(exc)
            artifacts[directory.name] = entry

        training = self._read_forecaster_training_status()
        return {
            "corpus": corpus_info,
            "artifacts": artifacts,
            "training": training,
            "model": "CausalGNNTransformer (joint edge-aware GNN + temporal Transformer)",
        }

    def _read_forecaster_training_status(self) -> dict:
        output = self.FORECASTER_OUTPUT_DIR
        status_path = output / self.FORECASTER_STATUS_FILE
        if not status_path.is_file():
            return {"state": "idle", "status_file": str(status_path)}
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"state": "unknown", "error": str(exc), "status_file": str(status_path)}
        pid = status.get("pid")
        if isinstance(pid, int) and status.get("state") == "running":
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError, OSError):
                status = dict(status)
                status["state"] = "exited"
                status["alive"] = False
            else:
                status["alive"] = True
        log_path = status.get("log_file")
        if isinstance(log_path, str) and Path(log_path).is_file():
            try:
                lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
                status["log_tail"] = lines[-20:]
            except Exception:
                pass
        return status

    def start_forecaster_training(
        self,
        *,
        epochs: int = 60,
        width: int = 32,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        heads: int = 4,
        layers: int = 2,
        batch_size: int = 2,
        seed: int = 26137,
        device: str = "cpu",
        scheduler: str = "cosine",
        patience: int = 25,
        final_eval: bool = True,
        stream: bool = True,
        stride: int = 1,
    ) -> dict:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if scheduler not in ("cosine", "plateau", "none"):
            raise ValueError("scheduler must be cosine, plateau or none")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if stride > 1 and not stream:
            raise ValueError("stride>1 requires stream=True")
        corpus = self.FORECASTER_CORPUS_DIR
        if not (corpus / "corpus_manifest.json").is_file():
            raise ValueError(f"corpus_v2 manifest is missing at {corpus}")
        output = self.FORECASTER_OUTPUT_DIR
        output.mkdir(parents=True, exist_ok=True)

        current = self._read_forecaster_training_status()
        if current.get("state") == "running" and current.get("alive", True):
            result = dict(current)
            result["already_running"] = True
            return result

        cache_dir = corpus / ".window_cache"
        log_path = output / "training.log"
        command = [
            sys.executable,
            "-m",
            "src.learning.train_forecaster",
            str(corpus),
            str(output),
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--width", str(width),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--weight-decay", str(weight_decay),
            "--dropout", str(dropout),
            "--heads", str(heads),
            "--layers", str(layers),
            "--scheduler", scheduler,
            "--patience", str(patience),
            "--device", device,
            "--window-cache", str(cache_dir),
        ]
        if not final_eval:
            # Screening mode: skip the memory-heavy in-training final eval
            # (full-split [N,E,H] concats). Artifacts are finalized afterwards
            # with scripts/finalize_forecaster.py (streamed, O(batch) peak).
            command.append("--no-final-eval")
        if stream:
            # O(episode) RAM + zero padding waste; incompatible with the
            # in-training finale (see above).
            command.append("--stream")
        if stride > 1:
            command.extend(["--stride", str(stride)])
        log_handle = open(log_path, "ab")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        # The child keeps the inherited fd; the parent copy can close now.
        log_handle.close()
        status = {
            "state": "running",
            "pid": process.pid,
            "corpus": str(corpus),
            "output": str(output),
            "log_file": str(log_path),
            "status_file": str(output / self.FORECASTER_STATUS_FILE),
            "started_at": time.time(),
            "config": {
                "epochs": epochs,
                "width": width,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "heads": heads,
                "layers": layers,
                "batch_size": batch_size,
                "seed": seed,
                "device": device,
                "scheduler": scheduler,
                "patience": patience,
                "final_eval": final_eval,
                "stream": stream,
                "stride": stride,
            },
            "command": command,
            "already_running": False,
            "final_eval": final_eval,
        }
        (output / self.FORECASTER_STATUS_FILE).write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        return status


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
