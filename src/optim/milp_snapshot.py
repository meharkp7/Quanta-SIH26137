"""
Step 9 — five-job CVRPTW MILP reference.

This snapshot MILP is intentionally restricted to the static five-job
correctness fixture.  The road-to-stop snapshot is built with the project's
own DirectedPathBuilder, so every logical stop-to-stop arc uses the same
directed physical-path semantics as RouteEvaluator.

The MILP is independently checked by RouteEvaluator before an optimum is
labelled certified.

Run from the repository root:
    python -m src.optim.milp_snapshot
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.contracts.scenario import Scenario
from src.optim.references import ExactStatus, ReferenceMethod
from src.routing.route_evaluator import RouteEvaluationConfig, RouteEvaluator
from src.routing.path_builder import DirectedPathBuilder, PathNotFoundError
from src.routing.route_plan import RoutePlan, VehicleRoute


EPS = 1e-8
DEFAULT_FIXTURE = Path("fixtures/step3/base/scenario.json")
DEFAULT_ARTIFACT = Path(
    "artifacts/step9_research/five_job_milp_reference.json"
)


class MILPSnapshotError(RuntimeError):
    """Raised when the exact five-job snapshot cannot be constructed."""


@dataclass(frozen=True)
class SnapshotArc:
    origin: str
    destination: str
    travel_time_s: float
    distance_m: float


@dataclass(frozen=True)
class MILPSnapshotResult:
    method: str
    status: str
    solver_status_code: int
    certified_optimum: bool
    objective: float | None
    lower_bound: float | None
    relative_gap: float | None
    mip_gap_tolerance: float
    time_limit_s: float
    elapsed_s: float
    route: dict[str, list[str]] | None
    evaluator_objective: float | None
    evaluator_feasible: bool
    objective_difference: float | None
    snapshot_arc_count: int
    snapshot_path_source: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_scenario(path: Path) -> Scenario:
    if not path.exists():
        raise MILPSnapshotError(f"fixture not found: {path}")
    return Scenario.model_validate_json(path.read_text())


def build_snapshot(
    scenario: Scenario,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[SnapshotArc, ...]]:
    """
    Build the complete directed stop-level snapshot through DirectedPathBuilder.

    This is deliberately not a second Dijkstra implementation.  The path
    builder is the project's canonical physical-road path constructor.
    """
    if len(scenario.requests) != 5:
        raise MILPSnapshotError("expected exactly 5 requests")
    if len(scenario.fleet) != 2:
        raise MILPSnapshotError("expected exactly 2 vehicles")

    depots = {str(v.depot_node_id) for v in scenario.fleet}
    starts = {str(v.start_node_id) for v in scenario.fleet}

    if len(depots) != 1:
        raise MILPSnapshotError("expected one common depot")

    depot = next(iter(depots))

    if starts != {depot}:
        raise MILPSnapshotError(
            "expected every vehicle start_node_id to equal the depot"
        )

    customers = tuple(str(r.request_id) for r in scenario.requests)
    vehicles = tuple(str(v.vehicle_id) for v in scenario.fleet)

    request_node = {
        str(r.request_id): str(r.access_node_id)
        for r in scenario.requests
    }

    stop_ids = ("DEPOT",) + customers
    stop_nodes = (depot,) + tuple(request_node[c] for c in customers)

    builder = DirectedPathBuilder(scenario.edges)

    arcs: list[SnapshotArc] = []

    for oi, origin_node in enumerate(stop_nodes):
        for di, destination_node in enumerate(stop_nodes):
            if oi == di:
                continue

            try:
                path = builder.shortest_path(
                    origin_node,
                    destination_node,
                    departure_time_s=0.0,
                )
            except PathNotFoundError as exc:
                raise MILPSnapshotError(
                    f"no legal directed snapshot path from "
                    f"{origin_node!r} to {destination_node!r}"
                ) from exc

            arcs.append(
                SnapshotArc(
                    origin=stop_ids[oi],
                    destination=stop_ids[di],
                    travel_time_s=float(path.travel_time_s),
                    distance_m=float(path.distance_m),
                )
            )

    return customers, vehicles, tuple(arcs)


def _solve(
    *,
    scenario: Scenario,
    customers: tuple[str, ...],
    vehicles: tuple[str, ...],
    arcs: tuple[SnapshotArc, ...],
    config: RouteEvaluationConfig,
    time_limit_s: float,
    mip_gap: float,
) -> tuple[
    ExactStatus,
    int,
    bool,
    float | None,
    float | None,
    float | None,
    dict[str, list[str]] | None,
    float,
    str,
]:
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise MILPSnapshotError("SciPy MILP support is required") from exc

    started = time.perf_counter()

    n = len(customers)
    k_count = len(vehicles)
    arc_count = len(arcs)

    customer_index = {
        customer_id: i
        for i, customer_id in enumerate(customers)
    }

    vehicle_index = {
        vehicle_id: k
        for k, vehicle_id in enumerate(vehicles)
    }

    arc_index = {
        (arc.origin, arc.destination): a
        for a, arc in enumerate(arcs)
    }

    # Variable blocks.
    #
    # x[a,k] : vehicle k uses stop arc a
    # y[i,k] : customer i assigned to vehicle k
    # z[k]   : vehicle k is active
    # t[i,k] : service-start time at customer i
    # u[i,k] : MTZ visit order
    # w[i,k] : waiting time before service at customer i
    x: dict[tuple[int, int], int] = {}
    y: dict[tuple[int, int], int] = {}
    z: dict[int, int] = {}
    t: dict[tuple[int, int], int] = {}
    u: dict[tuple[int, int], int] = {}
    w: dict[tuple[int, int], int] = {}

    cursor = 0

    for a in range(arc_count):
        for k in range(k_count):
            x[a, k] = cursor
            cursor += 1

    for i in range(n):
        for k in range(k_count):
            y[i, k] = cursor
            cursor += 1

    for k in range(k_count):
        z[k] = cursor
        cursor += 1

    for i in range(n):
        for k in range(k_count):
            t[i, k] = cursor
            cursor += 1

    for i in range(n):
        for k in range(k_count):
            u[i, k] = cursor
            cursor += 1

    for i in range(n):
        for k in range(k_count):
            w[i, k] = cursor
            cursor += 1

    var_count = cursor

    # Conservative finite horizon for big-M constraints.
    max_latest = max(
        float(r.latest_service_start_s)
        for r in scenario.requests
    )
    max_travel = max(
        float(a.travel_time_s)
        for a in arcs
    )
    total_service = sum(
        float(r.service_duration_s)
        for r in scenario.requests
    )

    departure = float(config.default_start_time_s)

    horizon = max(
        departure + 1.0,
        max_latest + total_service + max_travel + 100.0,
    )

    # Objective = evaluator's static fixture objective:
    #
    #   distance + travel_time + waiting
    #
    # Service time is intentionally excluded by the fixture config.
    objective = np.zeros(var_count, dtype=float)

    for a, arc in enumerate(arcs):
        coefficient = (
            float(config.distance_weight) * float(arc.distance_m)
            + float(config.travel_time_weight) * float(arc.travel_time_s)
        )
        for k in range(k_count):
            objective[x[a, k]] = coefficient

    for i in range(n):
        for k in range(k_count):
            objective[w[i, k]] = float(config.waiting_time_weight)

    integrality = np.zeros(var_count, dtype=int)

    for index in x.values():
        integrality[index] = 1
    for index in y.values():
        integrality[index] = 1
    for index in z.values():
        integrality[index] = 1
    for index in u.values():
        integrality[index] = 1

    lower_bounds = np.full(var_count, -np.inf, dtype=float)
    upper_bounds = np.full(var_count, np.inf, dtype=float)

    for index in x.values():
        lower_bounds[index] = 0.0
        upper_bounds[index] = 1.0

    for index in y.values():
        lower_bounds[index] = 0.0
        upper_bounds[index] = 1.0

    for index in z.values():
        lower_bounds[index] = 0.0
        upper_bounds[index] = 1.0

    for index in t.values():
        lower_bounds[index] = 0.0
        upper_bounds[index] = horizon

    for index in u.values():
        lower_bounds[index] = 0.0
        upper_bounds[index] = float(n)

    for index in w.values():
        lower_bounds[index] = 0.0
        upper_bounds[index] = horizon

    rows: list[dict[int, float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add(
        coefficients: dict[int, float],
        lower: float = -np.inf,
        upper: float = np.inf,
    ) -> None:
        rows.append(coefficients)
        row_lower.append(lower)
        row_upper.append(upper)

    # --------------------------------------------------------------
    # Every customer is assigned exactly once.
    # --------------------------------------------------------------
    for i in range(n):
        add(
            {
                y[i, k]: 1.0
                for k in range(k_count)
            },
            1.0,
            1.0,
        )

    # --------------------------------------------------------------
    # Vehicle activation and depot flow.
    # --------------------------------------------------------------
    for k in range(k_count):
        outgoing_depot = [
            a
            for a, arc in enumerate(arcs)
            if arc.origin == "DEPOT"
            and arc.destination != "DEPOT"
        ]
        incoming_depot = [
            a
            for a, arc in enumerate(arcs)
            if arc.destination == "DEPOT"
            and arc.origin != "DEPOT"
        ]

        add(
            {
                x[a, k]: 1.0
                for a in outgoing_depot
            }
            | {z[k]: -1.0},
            0.0,
            0.0,
        )

        add(
            {
                x[a, k]: 1.0
                for a in incoming_depot
            }
            | {z[k]: -1.0},
            0.0,
            0.0,
        )

    # --------------------------------------------------------------
    # Customer assignment = one incoming and one outgoing arc per
    # assigned vehicle/customer.
    # --------------------------------------------------------------
    for i, customer in enumerate(customers):
        incoming = [
            a
            for a, arc in enumerate(arcs)
            if arc.destination == customer
        ]
        outgoing = [
            a
            for a, arc in enumerate(arcs)
            if arc.origin == customer
        ]

        for k in range(k_count):
            add(
                {
                    **{x[a, k]: 1.0 for a in incoming},
                    y[i, k]: -1.0,
                },
                0.0,
                0.0,
            )
            add(
                {
                    **{x[a, k]: 1.0 for a in outgoing},
                    y[i, k]: -1.0,
                },
                0.0,
                0.0,
            )

    # --------------------------------------------------------------
    # Vehicle flow conservation at every customer.
    # --------------------------------------------------------------
    for k in range(k_count):
        for customer in customers:
            incoming = [
                a
                for a, arc in enumerate(arcs)
                if arc.destination == customer
            ]
            outgoing = [
                a
                for a, arc in enumerate(arcs)
                if arc.origin == customer
            ]

            add(
                {
                    **{x[a, k]: 1.0 for a in incoming},
                    **{x[a, k]: -1.0 for a in outgoing},
                },
                0.0,
                0.0,
            )

    # --------------------------------------------------------------
    # Capacity.
    # --------------------------------------------------------------
    request_by_id = {
        str(r.request_id): r
        for r in scenario.requests
    }

    for k, vehicle_id in enumerate(vehicles):
        vehicle = next(
            v for v in scenario.fleet
            if str(v.vehicle_id) == vehicle_id
        )

        add(
            {
                y[i, k]: float(
                    request_by_id[customer].demand
                )
                for i, customer in enumerate(customers)
            },
            -np.inf,
            float(vehicle.capacity),
        )

    # --------------------------------------------------------------
    # Service-time windows and release times.
    # --------------------------------------------------------------
    for i, customer in enumerate(customers):
        request = request_by_id[customer]

        earliest = max(
            float(request.earliest_service_start_s),
            float(request.release_s),
        )
        latest = float(request.latest_service_start_s)

        for k in range(k_count):
            # t >= earliest when assigned.
            add(
                {
                    t[i, k]: 1.0,
                    y[i, k]: -earliest,
                },
                0.0,
                np.inf,
            )

            # t <= latest when assigned.
            add(
                {
                    t[i, k]: 1.0,
                    y[i, k]: -latest,
                },
                -np.inf,
                0.0,
            )

            # Unassigned customers have t=w=u=0.
            add(
                {
                    t[i, k]: 1.0,
                    y[i, k]: -horizon,
                },
                -np.inf,
                0.0,
            )

            add(
                {
                    w[i, k]: 1.0,
                    y[i, k]: -horizon,
                },
                -np.inf,
                0.0,
            )

            # MTZ bounds.
            add(
                {
                    u[i, k]: 1.0,
                    y[i, k]: -1.0,
                },
                0.0,
                np.inf,
            )
            add(
                {
                    u[i, k]: 1.0,
                    y[i, k]: -float(n),
                },
                -np.inf,
                0.0,
            )

    # --------------------------------------------------------------
    # MTZ subtour elimination.
    # --------------------------------------------------------------
    for k in range(k_count):
        for i, origin in enumerate(customers):
            for j, destination in enumerate(customers):
                if i == j:
                    continue

                a = arc_index.get((origin, destination))
                if a is None:
                    continue

                # u_j >= u_i + 1 - M(1-x_ij)
                M = float(n + 1)

                add(
                    {
                        u[j, k]: 1.0,
                        u[i, k]: -1.0,
                        x[a, k]: -M,
                    },
                    1.0 - M,
                    np.inf,
                )

    # --------------------------------------------------------------
    # Time propagation and waiting.
    #
    # For a selected customer->customer arc i->j:
    #   t_j >= t_i + service_i + travel_ij
    #
    # For depot->customer:
    #   t_j >= departure + travel
    #
    # Waiting is:
    #   w_j >= t_j - arrival_j
    #
    # Because waiting has a positive objective coefficient, the optimum
    # makes this lower bound tight.
    # --------------------------------------------------------------
    for a, arc in enumerate(arcs):
        if arc.destination == "DEPOT":
            continue

        destination = arc.destination
        j = customer_index[destination]

        if arc.origin == "DEPOT":
            for k in range(k_count):
                add(
                    {
                        t[j, k]: 1.0,
                        x[a, k]: -horizon,
                    },
                    departure + float(arc.travel_time_s) - horizon,
                    np.inf,
                )

                # w_j >= t_j - (departure + travel)
                add(
                    {
                        w[j, k]: 1.0,
                        t[j, k]: -1.0,
                        x[a, k]: -horizon,
                    },
                    -departure - float(arc.travel_time_s) - horizon,
                    np.inf,
                )
            continue

        origin = arc.origin
        if origin not in customer_index:
            continue

        i = customer_index[origin]
        service_i = float(
            request_by_id[origin].service_duration_s
        )

        for k in range(k_count):
            # t_j - t_i >= service_i + travel - M(1-x)
            M = horizon + service_i + float(arc.travel_time_s) + 1.0

            add(
                {
                    t[j, k]: 1.0,
                    t[i, k]: -1.0,
                    x[a, k]: -M,
                },
                service_i + float(arc.travel_time_s) - M,
                np.inf,
            )

            # w_j >= t_j - (t_i + service_i + travel)
            add(
                {
                    w[j, k]: 1.0,
                    t[j, k]: -1.0,
                    t[i, k]: 1.0,
                    x[a, k]: -M,
                },
                -service_i - float(arc.travel_time_s) - M,
                np.inf,
            )

    # --------------------------------------------------------------
    # Build sparse constraint matrix.
    # --------------------------------------------------------------
    matrix = lil_matrix(
        (len(rows), var_count),
        dtype=float,
    )

    for row_index, coefficients in enumerate(rows):
        for column_index, coefficient in coefficients.items():
            matrix[row_index, column_index] = coefficient

    constraints = LinearConstraint(
        matrix.tocsr(),
        np.asarray(row_lower, dtype=float),
        np.asarray(row_upper, dtype=float),
    )

    bounds = Bounds(
        lower_bounds,
        upper_bounds,
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": float(time_limit_s),
            "mip_rel_gap": float(mip_gap),
        },
    )

    elapsed = time.perf_counter() - started

    status = {
        0: ExactStatus.OPTIMAL,
        1: ExactStatus.TIME_LIMIT,
        2: ExactStatus.INFEASIBLE,
        3: ExactStatus.UNBOUNDED,
        4: ExactStatus.NUMERICAL,
    }.get(
        int(result.status),
        ExactStatus.UNKNOWN,
    )

    objective_value = (
        float(result.fun)
        if result.fun is not None
        and math.isfinite(float(result.fun))
        else None
    )

    lower_bound = None
    for name in ("mip_dual_bound", "dual_bound", "lower_bound"):
        value = getattr(result, name, None)
        if value is not None:
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate):
                lower_bound = candidate
                break

    relative_gap = None
    solver_gap = getattr(result, "mip_gap", None)
    if solver_gap is not None:
        try:
            candidate = float(solver_gap)
            if math.isfinite(candidate) and candidate >= 0.0:
                relative_gap = candidate
        except (TypeError, ValueError):
            pass

    if (
        relative_gap is None
        and objective_value is not None
        and lower_bound is not None
    ):
        relative_gap = max(
            0.0,
            (objective_value - lower_bound)
            / max(abs(objective_value), 1e-12),
        )

    route: dict[str, list[str]] | None = None

    if result.x is not None and status in (
        ExactStatus.OPTIMAL,
        ExactStatus.TIME_LIMIT,
    ):
        route = {
            vehicle_id: []
            for vehicle_id in vehicles
        }

        for k, vehicle_id in enumerate(vehicles):
            current = "DEPOT"
            seen: set[str] = set()

            while True:
                selected = None

                for a, arc in enumerate(arcs):
                    if (
                        arc.origin == current
                        and float(result.x[x[a, k]]) > 0.5
                    ):
                        selected = arc
                        break

                if selected is None:
                    break

                if selected.destination == "DEPOT":
                    current = "DEPOT"
                    break

                customer = selected.destination

                if customer in seen:
                    raise MILPSnapshotError(
                        f"decoded cycle for vehicle {vehicle_id}"
                    )

                seen.add(customer)
                route[vehicle_id].append(customer)
                current = customer

            if current != "DEPOT" and route[vehicle_id]:
                raise MILPSnapshotError(
                    f"decoded route for {vehicle_id} does not return to depot"
                )

    # Solver-level certification is deliberately provisional.  run() applies
    # the independent RouteEvaluator agreement check before final certification.
    certified = (
        status is ExactStatus.OPTIMAL
        and objective_value is not None
        and (
            relative_gap is None
            or relative_gap <= float(mip_gap) + 1e-12
        )
    )

    return (
        status,
        int(result.status),
        certified,
        objective_value,
        lower_bound,
        relative_gap,
        route,
        elapsed,
        str(result.message),
    )


def _route_to_plan(route: dict[str, list[str]]) -> RoutePlan:
    return RoutePlan.from_routes(
        [
            VehicleRoute.from_sequence(vehicle_id, customers)
            for vehicle_id, customers in route.items()
        ]
    )


def run(
    *,
    fixture: Path = DEFAULT_FIXTURE,
    artifact: Path = DEFAULT_ARTIFACT,
    time_limit_s: float = 30.0,
    mip_gap: float = 1e-9,
) -> MILPSnapshotResult:
    scenario = _load_scenario(fixture)

    config = RouteEvaluationConfig(
        distance_weight=1.0,
        travel_time_weight=1.0,
        waiting_time_weight=1.0,
        service_time_weight=0.0,
        default_start_time_s=0.0,
    )

    evaluator = RouteEvaluator(
        scenario,
        config=config,
    )

    customers, vehicles, arcs = build_snapshot(scenario)

    (
        status,
        solver_status_code,
        solver_certified,
        objective,
        lower_bound,
        relative_gap,
        route,
        elapsed,
        message,
    ) = _solve(
        scenario=scenario,
        customers=customers,
        vehicles=vehicles,
        arcs=arcs,
        config=config,
        time_limit_s=time_limit_s,
        mip_gap=mip_gap,
    )

    evaluator_objective = None
    evaluator_feasible = False
    objective_difference = None

    if route is not None:
        # Explicit coverage guard before handing the route to the evaluator.
        flattened = [
            customer
            for sequence in route.values()
            for customer in sequence
        ]

        if (
            len(flattened) != len(customers)
            or set(flattened) != set(customers)
            or len(flattened) != len(set(flattened))
        ):
            raise MILPSnapshotError(
                "MILP decoder did not return every customer exactly once"
            )

        evaluation = evaluator.evaluate(
            _route_to_plan(route)
        )

        evaluator_objective = float(
            evaluation.objective_value
        )
        evaluator_feasible = bool(
            evaluation.feasible
        )

        if objective is not None:
            objective_difference = (
                evaluator_objective - objective
            )

    independently_verified = (
        solver_certified
        and evaluator_feasible
        and evaluator_objective is not None
        and objective is not None
        and abs(
            float(evaluator_objective)
            - float(objective)
        ) <= EPS
    )

    certified = bool(
        independently_verified
    )

    result = MILPSnapshotResult(
        method=ReferenceMethod.MILP.value,
        status=status.value,
        solver_status_code=solver_status_code,
        certified_optimum=certified,
        objective=objective,
        lower_bound=lower_bound,
        relative_gap=relative_gap,
        mip_gap_tolerance=float(mip_gap),
        time_limit_s=float(time_limit_s),
        elapsed_s=float(elapsed),
        route=route,
        evaluator_objective=evaluator_objective,
        evaluator_feasible=evaluator_feasible,
        objective_difference=objective_difference,
        snapshot_arc_count=len(arcs),
        snapshot_path_source="DirectedPathBuilder",
        message=message,
    )

    artifact.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        )
    )

    return result


def main() -> None:
    result = run()

    print("Step 9 — five-job MILP exact reference")
    print("=" * 50)
    print(f"Status:              {result.status}")
    print(f"Certified optimum:   {result.certified_optimum}")
    print(
        f"Solver objective:    {result.objective:.6f}"
        if result.objective is not None
        else "Solver objective:    None"
    )
    print(f"Lower bound:         {result.lower_bound}")
    print(f"Relative gap:        {result.relative_gap}")
    print(f"Elapsed:             {result.elapsed_s:.3f} s")
    print(f"Snapshot arcs:       {result.snapshot_arc_count}")
    print(f"Snapshot paths:      {result.snapshot_path_source}")
    print(f"Evaluator feasible:  {result.evaluator_feasible}")
    print(f"Evaluator objective: {result.evaluator_objective}")
    print(
        f"Objective difference:{result.objective_difference}"
    )

    if result.route is not None:
        print("Route:")
        for vehicle_id, customers in result.route.items():
            print(
                f"  {vehicle_id}: "
                + (
                    " -> ".join(customers)
                    if customers
                    else "(unused)"
                )
            )

    print(f"Artifact:            {DEFAULT_ARTIFACT}")


if __name__ == "__main__":
    main()