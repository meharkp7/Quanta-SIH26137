from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

from src.contracts.core_types import ScopeAction
from src.optim.qpso import (
    AdaptiveQPSO,
    AdaptiveQPSOConfig,
    RouteFitnessOracle,
)
from src.routing.route_encoding import Step7RouteEngine

from src.runtime.scope_actions import (
    ScopeSelection,
    build_commitment_snapshot,
)


@dataclass(frozen=True)
class ScopeReplanConfig:
    """
    Initial Step-15 development budgets.

    These are proposed budgets from the methodology, not measured
    performance claims.
    """

    local_budget_s: float = 0.25
    vehicle_budget_s: float = 0.50
    regional_budget_s: float = 1.00
    global_budget_s: float = 2.00

    particles: int = 20
    evaluations: int = 100

    seed: int = 26137


@dataclass(frozen=True)
class ScopeReplanResult:
    requested_action: ScopeAction
    executed_action: ScopeAction

    mutable_request_ids: tuple[str, ...]
    affected_vehicle_ids: tuple[str, ...]

    route_plan: object | None
    evaluation: object | None

    feasible: bool
    qpso_called: bool

    elapsed_s: float
    evaluations: int

    overridden: bool
    override_reason: str | None

    reason: str


class ScopeReplanner:
    """
    Execute Step-15 scope decisions through the real Step-7/QPSO stack.

    Scope selection and route optimization remain separate:
        ScopeActionSelector -> scope
        CommitmentSnapshot  -> immutable state
        Step7RouteEngine     -> decode/repair/evaluate
        AdaptiveQPSO         -> optimize
    """

    def __init__(
        self,
        scenario,
        *,
        config: ScopeReplanConfig | None = None,
    ) -> None:
        self.scenario = scenario
        self.config = (
            config
            if config is not None
            else ScopeReplanConfig()
        )

    def replan(
        self,
        current_plan,
        selection: ScopeSelection,
        *,
        planning_time_s: float = 0.0,
        closed_edge_ids: Sequence[str] = (),
    ) -> ScopeReplanResult:

        # --------------------------------------------------------------
        # KEEP means no QPSO.
        # --------------------------------------------------------------

        if selection.action == ScopeAction.KEEP:
            return ScopeReplanResult(
                requested_action=ScopeAction.KEEP,
                executed_action=ScopeAction.KEEP,
                mutable_request_ids=(),
                affected_vehicle_ids=selection.affected_vehicle_ids,
                route_plan=current_plan,
                evaluation=None,
                feasible=True,
                qpso_called=False,
                elapsed_s=0.0,
                evaluations=0,
                overridden=selection.overridden,
                override_reason=selection.override_reason,
                reason=selection.reason,
            )

        start = perf_counter()

        # --------------------------------------------------------------
        # Build the REAL immutable commitment snapshot.
        # --------------------------------------------------------------

        commitments = build_commitment_snapshot(
            self.scenario,
            current_plan,
            selection,
            planning_time_s=planning_time_s,
        )

        # --------------------------------------------------------------
        # Build the existing Step-7 engine.
        # --------------------------------------------------------------

        from src.routing.route_evaluator import RouteEvaluator

        evaluator = RouteEvaluator(
            self.scenario,
            closed_edge_ids=closed_edge_ids,
        )

        engine = Step7RouteEngine(
            self.scenario,
            evaluator,
        )

        # --------------------------------------------------------------
        # Existing QPSO oracle.
        # --------------------------------------------------------------

        oracle = RouteFitnessOracle(
            engine,
            commitments=commitments,
            planning_time_s=planning_time_s,
            repair=True,
        )

        budget = self._budget_for(
            selection.action
        )

        qpso_config = AdaptiveQPSOConfig(
            dimensions=engine.encoder.dimension,
            lower_bound=0.0,
            upper_bound=1.0,
            population_size=self.config.particles,
            max_evaluations=self.config.evaluations,
            seed=self.config.seed,
        )

        # --------------------------------------------------------------
        # Warm start from current route.
        # --------------------------------------------------------------

        encoded = engine.encoder.encode(
            current_plan,
            commitments=commitments,
        )

        initial_population = self._initial_population(
            encoded.keys,
            engine.encoder.dimension,
        )

        optimizer = AdaptiveQPSO(
            qpso_config,
            oracle,
            initial_population=initial_population,
        )

        result = optimizer.optimize()

        # --------------------------------------------------------------
        # Re-evaluate final position through the REAL engine.
        # --------------------------------------------------------------

        candidate = engine.evaluate_keys(
            result.best_position,
            commitments=commitments,
            planning_time_s=planning_time_s,
            repair=True,
        )

        evaluation = getattr(
            candidate,
            "repaired_evaluation",
            None,
        )

        if evaluation is None:
            evaluation = getattr(
                candidate,
                "evaluation",
                None,
            )

        route_plan = getattr(
            candidate,
            "repaired_plan",
            None,
        )

        if route_plan is None:
            route_plan = getattr(
                candidate,
                "route_plan",
                None,
            )

        elapsed = perf_counter() - start

        # --------------------------------------------------------------
        # Hard safety condition:
        # never dispatch an invalid result.
        # --------------------------------------------------------------

        feasible = bool(
            evaluation is not None
            and evaluation.feasible
        )

        if not feasible:
            return ScopeReplanResult(
                requested_action=selection.action,
                executed_action=ScopeAction.KEEP,
                mutable_request_ids=selection.request_ids,
                affected_vehicle_ids=selection.affected_vehicle_ids,
                route_plan=current_plan,
                evaluation=evaluation,
                feasible=False,
                qpso_called=True,
                elapsed_s=elapsed,
                evaluations=result.evaluations,
                overridden=True,
                override_reason=(
                    "QPSO did not produce an independently validated "
                    "feasible scoped route."
                ),
                reason=(
                    "Retained the current legal incumbent because the "
                    "scoped candidate was not feasible."
                ),
            )

        return ScopeReplanResult(
            requested_action=selection.action,
            executed_action=selection.action,
            mutable_request_ids=selection.request_ids,
            affected_vehicle_ids=selection.affected_vehicle_ids,
            route_plan=route_plan,
            evaluation=evaluation,
            feasible=True,
            qpso_called=True,
            elapsed_s=elapsed,
            evaluations=result.evaluations,
            overridden=selection.overridden,
            override_reason=selection.override_reason,
            reason=selection.reason,
        )

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def _budget_for(
        self,
        action: ScopeAction,
    ) -> float:
        if action == ScopeAction.LOCAL:
            return self.config.local_budget_s

        if action == ScopeAction.VEHICLE:
            return self.config.vehicle_budget_s

        if action == ScopeAction.REGIONAL:
            return self.config.regional_budget_s

        if action == ScopeAction.GLOBAL:
            return self.config.global_budget_s

        return 0.0

    # ------------------------------------------------------------------
    # Warm-start population
    # ------------------------------------------------------------------

    @staticmethod
    def _initial_population(
        incumbent_keys,
        dimensions: int,
    ) -> list[list[float]]:
        """
        Seed QPSO with the incumbent plus deterministic perturbations.

        The optimizer still owns its stochastic evolution.
        """

        base = [
            min(
                1.0,
                max(
                    0.0,
                    float(value),
                ),
            )
            for value in incumbent_keys
        ]

        population = [base]

        # Deterministic small perturbations.
        for scale in (0.01, 0.03, 0.07, 0.15):
            plus = [
                min(1.0, value + scale)
                for value in base
            ]

            minus = [
                max(0.0, value - scale)
                for value in base
            ]

            population.append(plus)
            population.append(minus)

        # Fill remaining population with incumbent copies.
        # QPSO itself supplies the subsequent diversity.
        while len(population) < 20:
            population.append(list(base))

        return population[:20]