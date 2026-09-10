"""End-to-end routing pipeline orchestration for Step 4.

The pipeline is the integration boundary between:

    Scenario
        ↓
    InitialSolutionBuilder
        ↓
    RouteRepairer
        ↓
    RouteEvaluator
        ↓
    RoutingPipelineResult

The pipeline intentionally does not implement QPSO, PSO, ALNS, DRL, GNN,
Transformer or SUMO logic. It provides the deterministic baseline execution
path that those later components can consume.

Supported execution modes
-------------------------
* static routing;
* dynamic traffic;
* incident-aware routing;
* rolling-horizon re-routing;
* optimizer warm starts;
* benchmark evaluation;
* external candidate-plan evaluation;
* SUMO-facing baseline evaluation.

All scale is determined by the supplied Scenario. No artificial vehicle,
customer, edge or route-count limits are imposed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.contracts.core_types import RoadEdgeId, TimeS
from src.routing.initial_solution import (
    InitialSolutionBuilder,
    InitialSolutionConfig,
    InitialSolutionResult,
)
from src.routing.route_evaluator import (
    RouteEvaluationConfig,
    RouteEvaluator,
    RoutePlanEvaluation,
    TravelTimeProvider,
)
from src.routing.route_plan import RoutePlan
from src.routing.route_repair import (
    RouteRepairConfig,
    RouteRepairResult,
    RouteRepairer,
)


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class RoutingPipelineConfig:
    """Configuration for one complete baseline-routing execution."""

    initial_solution: InitialSolutionConfig = (
        InitialSolutionConfig()
    )

    evaluator: RouteEvaluationConfig = (
        RouteEvaluationConfig()
    )

    repair: RouteRepairConfig = (
        RouteRepairConfig()
    )

    enable_repair: bool = True

    # If True, the pipeline reports the final result as an operational
    # failure when feasibility could not be recovered.
    #
    # The underlying evaluation is still returned in full either way.
    require_final_feasibility: bool = False

    # If True, an incomplete candidate is considered an execution error in
    # addition to being represented through RoutePlanEvaluation.
    require_complete_assignment: bool = False


# ============================================================================
# Result
# ============================================================================


@dataclass(frozen=True)
class RoutingPipelineResult:
    """Structured result produced by one pipeline execution."""

    initial_solution: InitialSolutionResult

    repaired_solution: RouteRepairResult | None

    final_route_plan: RoutePlan

    final_evaluation: RoutePlanEvaluation

    feasible: bool

    complete: bool

    repair_changed_solution: bool

    total_distance_m: float

    total_travel_time_s: TimeS

    total_waiting_time_s: TimeS

    total_service_time_s: TimeS

    total_elapsed_time_s: TimeS

    total_lateness_s: TimeS

    objective_value: float

    errors: tuple[str, ...]

    @property
    def route_plan(self) -> RoutePlan:
        """Alias for the final logical route plan."""

        return self.final_route_plan

    @property
    def evaluation(self) -> RoutePlanEvaluation:
        """Alias for the final route-plan evaluation."""

        return self.final_evaluation

    @property
    def repair_applied(self) -> bool:
        """Whether the repair stage was enabled."""

        return self.repaired_solution is not None

    @property
    def successful(self) -> bool:
        """Whether the final pipeline result is complete and feasible."""

        return self.feasible and self.complete


# ============================================================================
# Pipeline
# ============================================================================


class RoutingPipeline:
    """Coordinate construction, repair and evaluation.

    This class is intentionally thin. Business logic belongs to the
    underlying routing components; the pipeline only coordinates them.

    A single RouteEvaluator instance is shared by the initializer and
    repairer so that all three stages use identical network state and
    objective semantics.
    """

    def __init__(
        self,
        scenario,
        *,
        config: RoutingPipelineConfig | None = None,
        travel_time_provider: TravelTimeProvider | None = None,
        closed_edge_ids: Iterable[RoadEdgeId] = (),
    ) -> None:
        self.scenario = scenario

        self.config = (
            config
            if config is not None
            else RoutingPipelineConfig()
        )

        self.travel_time_provider = (
            travel_time_provider
        )

        self.closed_edge_ids = frozenset(
            closed_edge_ids
        )

        # ---------------------------------------------------------------
        # One authoritative evaluator for the entire pipeline.
        # ---------------------------------------------------------------

        self.evaluator = RouteEvaluator(
            scenario,
            config=self.config.evaluator,
            travel_time_provider=(
                travel_time_provider
            ),
            closed_edge_ids=self.closed_edge_ids,
        )

        # ---------------------------------------------------------------
        # Constructive initialization.
        # ---------------------------------------------------------------

        self.initial_solution_builder = (
            InitialSolutionBuilder(
                scenario,
                evaluator=self.evaluator,
                config=self.config.initial_solution,
            )
        )

        # ---------------------------------------------------------------
        # Feasibility repair.
        # ---------------------------------------------------------------

        self.repairer = RouteRepairer(
            scenario,
            self.evaluator,
            config=self.config.repair,
        )

    # =========================================================================
    # Main execution
    # =========================================================================

    def solve(
        self,
        *,
        planning_time_s: TimeS = 0.0,
    ) -> RoutingPipelineResult:
        """Build, optionally repair, and evaluate a baseline route plan.

        ``planning_time_s`` is propagated unchanged through construction,
        repair and final evaluation so dynamic traffic providers observe a
        consistent simulation/planning epoch.
        """

        initial_solution = (
            self.initial_solution_builder.build(
                planning_time_s=planning_time_s,
            )
        )

        initial_plan = (
            initial_solution.route_plan
        )

        if self.config.enable_repair:
            repaired_solution = (
                self.repairer.repair(
                    initial_plan,
                    planning_time_s=planning_time_s,
                )
            )

            final_plan = (
                repaired_solution.repaired_route_plan
            )

            final_evaluation = (
                repaired_solution.repaired_evaluation
            )

            repair_changed = (
                repaired_solution.changed
            )

        else:
            repaired_solution = None

            final_plan = initial_plan

            final_evaluation = (
                self.evaluator.evaluate(
                    final_plan,
                    planning_time_s=planning_time_s,
                )
            )

            repair_changed = False

        return self._build_result(
            initial_solution=initial_solution,
            repaired_solution=repaired_solution,
            final_plan=final_plan,
            final_evaluation=final_evaluation,
            repair_changed=repair_changed,
        )

    # =========================================================================
    # External-plan evaluation
    # =========================================================================

    def evaluate(
        self,
        route_plan: RoutePlan,
        *,
        planning_time_s: TimeS = 0.0,
    ) -> RoutePlanEvaluation:
        """Evaluate an externally supplied logical route plan.

        This is useful for:
        * QPSO particles decoded into RoutePlans;
        * PSO/ALNS candidates;
        * DRL actions;
        * benchmark baselines;
        * manually supplied routes;
        * SUMO-generated candidate routes.
        """

        return self.evaluator.evaluate(
            route_plan,
            planning_time_s=planning_time_s,
        )

    # =========================================================================
    # Explicit repair API
    # =========================================================================

    def repair(
        self,
        route_plan: RoutePlan,
        *,
        planning_time_s: TimeS = 0.0,
    ) -> RouteRepairResult:
        """Repair an externally supplied route plan."""

        return self.repairer.repair(
            route_plan,
            planning_time_s=planning_time_s,
        )

    # =========================================================================
    # Explicit initial-solution API
    # =========================================================================

    def initial_solution(
        self,
        *,
        planning_time_s: TimeS = 0.0,
        request_ids=None,
    ) -> InitialSolutionResult:
        """Construct an initial solution without running repair."""

        return self.initial_solution_builder.build(
            planning_time_s=planning_time_s,
            request_ids=request_ids,
        )

    # =========================================================================
    # Network-state cloning
    # =========================================================================

    def with_network_state(
        self,
        *,
        closed_edge_ids: Iterable[RoadEdgeId] = (),
        travel_time_provider: TravelTimeProvider | None = None,
    ) -> RoutingPipeline:
        """Create a pipeline bound to a new network-state snapshot.

        The Scenario itself remains unchanged.

        This is the intended interface for rolling-horizon routing:

            t0 → pipeline(snapshot_0)
            t1 → pipeline(snapshot_1)
            t2 → pipeline(snapshot_2)
            ...

        The same logical scenario can therefore be evaluated against
        different traffic/incident states without mutating shared state.
        """

        provider = (
            self.travel_time_provider
            if travel_time_provider is None
            else travel_time_provider
        )

        return RoutingPipeline(
            self.scenario,
            config=self.config,
            travel_time_provider=provider,
            closed_edge_ids=closed_edge_ids,
        )

    # =========================================================================
    # Result construction
    # =========================================================================

    def _build_result(
        self,
        *,
        initial_solution: InitialSolutionResult,
        repaired_solution: RouteRepairResult | None,
        final_plan: RoutePlan,
        final_evaluation: RoutePlanEvaluation,
        repair_changed: bool,
    ) -> RoutingPipelineResult:
        """Build the public pipeline result and aggregate diagnostics."""

        errors = list(
            initial_solution.errors
        )

        if repaired_solution is not None:
            errors.extend(
                repaired_solution.unresolved_reasons
            )

        errors.extend(
            final_evaluation.errors
        )

        complete = (
            final_evaluation.all_requests_served
            and final_evaluation.customer_uniqueness_feasible
            and not final_evaluation.unknown_vehicle_ids
        )

        feasible = (
            final_evaluation.feasible
        )

        if (
            self.config.require_final_feasibility
            and not feasible
        ):
            errors.append(
                "Final routing pipeline result "
                "is infeasible."
            )

        if (
            self.config.require_complete_assignment
            and not complete
        ):
            errors.append(
                "Final routing pipeline result "
                "does not contain a complete customer assignment."
            )

        # dict.fromkeys preserves deterministic first occurrence while
        # eliminating repeated diagnostics from the different pipeline stages.
        unique_errors = tuple(
            dict.fromkeys(
                str(error)
                for error in errors
                if error
            )
        )

        return RoutingPipelineResult(
            initial_solution=initial_solution,
            repaired_solution=repaired_solution,
            final_route_plan=final_plan,
            final_evaluation=final_evaluation,
            feasible=feasible,
            complete=complete,
            repair_changed_solution=repair_changed,
            total_distance_m=(
                final_evaluation.total_distance_m
            ),
            total_travel_time_s=(
                final_evaluation.total_travel_time_s
            ),
            total_waiting_time_s=(
                final_evaluation.total_waiting_time_s
            ),
            total_service_time_s=(
                final_evaluation.total_service_time_s
            ),
            total_elapsed_time_s=(
                final_evaluation.total_elapsed_time_s
            ),
            total_lateness_s=(
                final_evaluation.total_lateness_s
            ),
            objective_value=(
                final_evaluation.objective_value
            ),
            errors=unique_errors,
        )


__all__ = [
    "RoutingPipelineConfig",
    "RoutingPipelineResult",
    "RoutingPipeline",
]