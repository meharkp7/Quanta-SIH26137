from __future__ import annotations

from pydantic import Field, model_validator

from ._base import Contract
from ._immutable import ImmutableStrMap
from .core_types import SolveStatus
from .routing import RoutePlan


class ConstraintViolation(Contract):
    name: str
    magnitude: float = Field(ge=0)
    count: int = Field(ge=0)


class ValidationResult(Contract):
    feasible: bool

    capacity_feasible: bool
    time_window_feasible: bool
    route_continuity_feasible: bool
    legal_road_feasible: bool
    commitment_feasible: bool
    depot_feasible: bool
    subtour_free: bool

    violations: tuple[ConstraintViolation, ...] = ()

    @model_validator(mode="after")
    def _feasible_iff_all_components_pass(self) -> "ValidationResult":
        components = (
            self.capacity_feasible,
            self.time_window_feasible,
            self.route_continuity_feasible,
            self.legal_road_feasible,
            self.commitment_feasible,
            self.depot_feasible,
            self.subtour_free,
        )
        all_pass = all(components)
        if self.feasible != all_pass:
            raise ValueError(
                f"feasible={self.feasible} is inconsistent with the "
                f"individual constraint checks (all_pass={all_pass}) -- "
                "'feasible' must be exactly the AND of every component, "
                "never asserted independently"
            )
        if self.feasible and self.violations:
            raise ValueError(
                "feasible=True but violations is nonempty"
            )
        return self


class SearchTrace(Contract):
    timestamp_s: tuple[float, ...]
    best_feasible_objective: tuple[float | None, ...]
    best_penalized_objective: tuple[float | None, ...]

    evaluator_calls: tuple[int, ...]
    feasible_counts: tuple[int, ...]

    local_search_improvement: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _series_are_aligned(self) -> "SearchTrace":
        n = len(self.timestamp_s)
        for name, series in (
            ("best_feasible_objective", self.best_feasible_objective),
            ("best_penalized_objective", self.best_penalized_objective),
            ("evaluator_calls", self.evaluator_calls),
            ("feasible_counts", self.feasible_counts),
        ):
            if len(series) != n:
                raise ValueError(
                    f"{name} has {len(series)} entries but timestamp_s has "
                    f"{n} -- every logged series must align to the same "
                    "timeline"
                )
        if list(self.timestamp_s) != sorted(self.timestamp_s):
            raise ValueError("timestamp_s must be nondecreasing")
        return self


class SolveResult(Contract):
    scenario_id: str
    state_version: str

    method: str
    solver_version: str

    status: SolveStatus

    route_plan: RoutePlan | None

    # These two are kept explicitly separate and must never be collapsed:
    # `objective_value` is the objective of the best *independently
    # validated feasible* incumbent (the only thing that may actually be
    # dispatched). `best_penalized_objective` is the best objective among
    # possibly-infeasible particles/candidates, used only to guide search.
    # See CONTRACTS.md / master plan: "only independently validated
    # feasible plans may be dispatched."
    objective_value: float | None
    best_penalized_objective: float | None = None

    objective_time: float | None = Field(default=None, ge=0)
    objective_distance: float | None = Field(default=None, ge=0)
    objective_congestion: float | None = Field(default=None, ge=0)
    route_change_penalty: float | None = Field(default=None, ge=0)

    validation: ValidationResult

    elapsed_time_s: float = Field(ge=0)

    search_trace: SearchTrace | None

    diagnostics: ImmutableStrMap = Field(default_factory=ImmutableStrMap)

    @model_validator(mode="after")
    def _feasible_status_implies_dispatchable_plan(self) -> "SolveResult":
        if self.status == SolveStatus.FEASIBLE:
            if self.route_plan is None:
                raise ValueError("status=FEASIBLE but route_plan is None")
            if not self.validation.feasible:
                raise ValueError(
                    "status=FEASIBLE but validation.feasible=False -- a "
                    "solver's own status claim never overrides the "
                    "independent validator"
                )
            if self.objective_value is None:
                raise ValueError("status=FEASIBLE but objective_value is None")
        if self.status in (SolveStatus.INFEASIBLE, SolveStatus.NO_FEASIBLE_INCUMBENT):
            if self.route_plan is not None and self.validation.feasible:
                raise ValueError(
                    f"status={self.status.value} but a feasible route_plan "
                    "and passing validation are both present"
                )
        return self
