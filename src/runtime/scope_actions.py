from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from src.contracts.decision import ScopeAction, ScopeDecision
from dataclasses import dataclass
from typing import Any
from src.routing.evaluator_state import (
    CommitmentSnapshot,
    VehicleCommitment,
)
from src.routing.evaluator_state import (
    CommitmentSnapshot,
    VehicleCommitment,
)

@dataclass(frozen=True)
class JobImpact:
    """
    Impact information for one mutable request/job.

    Higher impact means the job should be considered earlier when
    constructing LOCAL / REGIONAL scopes.
    """

    request_id: str
    vehicle_id: str
    zone_id: str | None = None
    affected: bool = False
    deadline_slack_s: float = float("inf")
    route_overlap_fraction: float = 0.0
    congestion_exposure: float = 0.0

    @property
    def impact_score(self) -> float:
        """Deterministic impact ranking used by scope selection."""

        slack_pressure = (
            0.0
            if self.deadline_slack_s == float("inf")
            else 1.0 / max(self.deadline_slack_s, 1.0)
        )

        return (
            4.0 * float(self.affected)
            + 3.0 * max(0.0, min(1.0, self.route_overlap_fraction))
            + 2.0 * max(0.0, self.congestion_exposure)
            + 5.0 * slack_pressure
        )


@dataclass(frozen=True)
class ScopeSelection:
    """
    Result of translating an action into an exact set of mutable jobs.

    `request_ids` and `vehicle_ids` are intentionally explicit.
    The planner must never infer the scope from the action name later.
    """

    action: ScopeAction
    request_ids: tuple[str, ...]
    vehicle_ids: tuple[str, ...]
    reason: str
    affected_vehicle_ids: tuple[str, ...] = ()
    overridden: bool = False
    override_reason: str | None = None


@dataclass
class ScopeActionConfig:
    """Initial Step-15 scope policy."""

    local_max_jobs: int = 10

    # A vehicle can only be selected for VEHICLE if it has at least
    # one affected/mutable request.
    require_affected_vehicle_for_vehicle: bool = True

    # Rule-based safety policy.
    safety_override_to_global: bool = True

    # KEEP is invalid if an affected mutable request exists and there is
    # no legal way to preserve the current plan.
    keep_requires_no_affected_mutable_jobs: bool = True


@dataclass
class ScopeActionSelector:
    """
    Deterministic Step-15 scope selector.

    This component decides WHICH jobs are mutable.
    Actual route optimization remains the responsibility of Step 7/QPSO.
    """

    config: ScopeActionConfig = field(default_factory=ScopeActionConfig)

    def select(
        self,
        action: ScopeAction,
        *,
        jobs: Sequence[JobImpact],
        affected_vehicle_ids: Iterable[str] = (),
        affected_zone_ids: Iterable[str] = (),
        legally_mutable_request_ids: Iterable[str] | None = None,
        legally_mutable_vehicle_ids: Iterable[str] | None = None,
    ) -> ScopeSelection:
        """
        Build an exact scope for the requested action.

        The selector is deliberately conservative:
        - only legally mutable jobs can enter a replan scope;
        - KEEP never claims to replan jobs;
        - duplicate IDs are removed deterministically;
        - invalid safety situations can override KEEP.
        """

        jobs = tuple(jobs)

        affected_vehicles = self._unique(affected_vehicle_ids)
        affected_zones = set(self._unique(affected_zone_ids))

        mutable_requests = (
            set(legally_mutable_request_ids)
            if legally_mutable_request_ids is not None
            else {job.request_id for job in jobs}
        )

        mutable_vehicles = (
            set(legally_mutable_vehicle_ids)
            if legally_mutable_vehicle_ids is not None
            else {job.vehicle_id for job in jobs}
        )

        eligible = tuple(
            job
            for job in jobs
            if job.request_id in mutable_requests
            and job.vehicle_id in mutable_vehicles
        )

        affected_mutable = tuple(
            job
            for job in eligible
            if job.affected or job.vehicle_id in affected_vehicles
        )

        if action == ScopeAction.KEEP:
            return self._select_keep(
                eligible=eligible,
                affected_mutable=affected_mutable,
                affected_vehicles=affected_vehicles,
            )

        if action == ScopeAction.LOCAL:
            selected = self._local_scope(affected_mutable)
            return self._selection(
                ScopeAction.LOCAL,
                selected,
                reason=(
                    "Selected up to "
                    f"{self.config.local_max_jobs} highest-impact mutable "
                    "jobs across affected vehicles."
                ),
                affected_vehicles=affected_vehicles,
            )

        if action == ScopeAction.VEHICLE:
            return self._vehicle_scope(
                eligible=eligible,
                affected_mutable=affected_mutable,
                affected_vehicles=affected_vehicles,
            )

        if action == ScopeAction.REGIONAL:
            selected = self._regional_scope(
                eligible,
                affected_mutable,
                affected_zones,
            )

            return self._selection(
                ScopeAction.REGIONAL,
                selected,
                reason=(
                    "Selected legally mutable jobs in affected vehicles "
                    "and neighboring/affected zones."
                ),
                affected_vehicles=affected_vehicles,
            )

        if action == ScopeAction.GLOBAL:
            selected = self._global_scope(eligible)
            return self._selection(
                ScopeAction.GLOBAL,
                selected,
                reason="Selected all legally mutable jobs.",
                affected_vehicles=affected_vehicles,
            )

        raise ValueError(f"Unsupported scope action: {action!r}")

    # ------------------------------------------------------------------
    # KEEP
    # ------------------------------------------------------------------

    def _select_keep(
        self,
        *,
        eligible: Sequence[JobImpact],
        affected_mutable: Sequence[JobImpact],
        affected_vehicles: tuple[str, ...],
    ) -> ScopeSelection:
        """
        KEEP means no mutable jobs are handed to QPSO.

        If the current assignment is unsafe because affected mutable work
        exists, override KEEP to the smallest safe scope.
        """

        if (
            self.config.safety_override_to_global
            and self.config.keep_requires_no_affected_mutable_jobs
            and affected_mutable
        ):
            selected = self._local_scope(affected_mutable)

            # If LOCAL cannot provide a safe scope, escalate globally.
            if not selected:
                selected = self._global_scope(eligible)
                override_action = ScopeAction.GLOBAL
            else:
                override_action = ScopeAction.LOCAL

            return self._selection(
                override_action,
                selected,
                reason="KEEP was unsafe because mutable affected work exists.",
                affected_vehicles=affected_vehicles,
                overridden=True,
                override_reason=(
                    "Requested KEEP was overridden by the rule-based safety "
                    "policy because affected mutable work requires replanning."
                ),
            )

        return ScopeSelection(
            action=ScopeAction.KEEP,
            request_ids=(),
            vehicle_ids=(),
            reason="No mutable work requires replanning; preserve current plan.",
            affected_vehicle_ids=affected_vehicles,
        )

    # ------------------------------------------------------------------
    # LOCAL
    # ------------------------------------------------------------------

    def _local_scope(
        self,
        jobs: Sequence[JobImpact],
    ) -> tuple[JobImpact, ...]:
        ranked = sorted(
            jobs,
            key=lambda job: (
                -job.impact_score,
                job.request_id,
            ),
        )

        return tuple(ranked[: self.config.local_max_jobs])

    # ------------------------------------------------------------------
    # VEHICLE
    # ------------------------------------------------------------------

    def _vehicle_scope(
        self,
        *,
        eligible: Sequence[JobImpact],
        affected_mutable: Sequence[JobImpact],
        affected_vehicles: tuple[str, ...],
    ) -> ScopeSelection:
        candidate_vehicle_ids = set(affected_vehicles)

        if self.config.require_affected_vehicle_for_vehicle:
            candidate_vehicle_ids.update(
                job.vehicle_id for job in affected_mutable
            )
        else:
            candidate_vehicle_ids.update(job.vehicle_id for job in eligible)

        if not candidate_vehicle_ids:
            return ScopeSelection(
                action=ScopeAction.VEHICLE,
                request_ids=(),
                vehicle_ids=(),
                reason="No affected vehicle has legally mutable work.",
                affected_vehicle_ids=affected_vehicles,
            )

        # Select the most impacted vehicle deterministically.
        vehicle_scores: dict[str, float] = {}

        for job in eligible:
            if job.vehicle_id not in candidate_vehicle_ids:
                continue

            vehicle_scores[job.vehicle_id] = (
                vehicle_scores.get(job.vehicle_id, 0.0)
                + job.impact_score
            )

        if not vehicle_scores:
            return ScopeSelection(
                action=ScopeAction.VEHICLE,
                request_ids=(),
                vehicle_ids=(),
                reason="Affected vehicles have no legally mutable requests.",
                affected_vehicle_ids=affected_vehicles,
            )

        selected_vehicle = min(
            vehicle_scores,
            key=lambda vehicle_id: (
                -vehicle_scores[vehicle_id],
                vehicle_id,
            ),
        )

        selected = tuple(
            job
            for job in eligible
            if job.vehicle_id == selected_vehicle
        )

        return self._selection(
            ScopeAction.VEHICLE,
            selected,
            reason=(
                "Selected the mutable suffix of the most affected vehicle "
                "according to deterministic impact ranking."
            ),
            affected_vehicles=affected_vehicles,
        )

    # ------------------------------------------------------------------
    # REGIONAL
    # ------------------------------------------------------------------

    def _regional_scope(
        self,
        eligible: Sequence[JobImpact],
        affected_mutable: Sequence[JobImpact],
        affected_zones: set[str],
    ) -> tuple[JobImpact, ...]:
        affected_vehicle_set = {
            job.vehicle_id for job in affected_mutable
        }

        selected: list[JobImpact] = []

        for job in eligible:
            if (
                job.vehicle_id in affected_vehicle_set
                or (
                    job.zone_id is not None
                    and job.zone_id in affected_zones
                )
            ):
                selected.append(job)

        return tuple(
            sorted(
                selected,
                key=lambda job: (-job.impact_score, job.request_id),
            )
        )

    # ------------------------------------------------------------------
    # GLOBAL
    # ------------------------------------------------------------------

    @staticmethod
    def _global_scope(
        eligible: Sequence[JobImpact],
    ) -> tuple[JobImpact, ...]:
        return tuple(
            sorted(
                eligible,
                key=lambda job: (
                    job.vehicle_id,
                    job.request_id,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selection(
        self,
        action: ScopeAction,
        jobs: Sequence[JobImpact],
        *,
        reason: str,
        affected_vehicles: tuple[str, ...],
        overridden: bool = False,
        override_reason: str | None = None,
    ) -> ScopeSelection:
        request_ids = self._unique(job.request_id for job in jobs)
        vehicle_ids = self._unique(job.vehicle_id for job in jobs)

        return ScopeSelection(
            action=action,
            request_ids=request_ids,
            vehicle_ids=vehicle_ids,
            reason=reason,
            affected_vehicle_ids=affected_vehicles,
            overridden=overridden,
            override_reason=override_reason,
        )

    @staticmethod
    def _unique(values: Iterable[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []

        for value in values:
            value = str(value)

            if value in seen:
                continue

            seen.add(value)
            result.append(value)

        return tuple(result)

@dataclass(frozen=True)
class ScopeCommitmentPlan:
    """
    Converts a scope decision into immutable customer prefixes.

    Requests outside the mutable scope remain committed to their current
    vehicle and ordering. Requests inside the scope remain eligible for
    QPSO replanning.
    """

    mutable_request_ids: tuple[str, ...]
    mutable_vehicle_ids: tuple[str, ...]
    committed_by_vehicle: dict[str, tuple[str, ...]]


def build_scope_commitment_plan(
        current_plan: Any,
        selection: ScopeSelection,
    ) -> ScopeCommitmentPlan:
        """
        Build commitment information from the CURRENT route plan.

        The current assignment is never modified here.

        For each vehicle:
        prefix = customers before the first mutable customer.

        Customers outside the selected mutable scope remain fixed.
        """

        mutable_requests = set(
            selection.request_ids
        )

        mutable_vehicles = set(
            selection.vehicle_ids
        )

        committed_by_vehicle: dict[str, tuple[str, ...]] = {}

        for vehicle_route in current_plan.vehicle_routes:
            vehicle_id = str(
                vehicle_route.vehicle_id
            )

            customers = tuple(
                str(customer_id)
                for customer_id in vehicle_route.customer_ids
            )

            # VEHICLE / LOCAL / REGIONAL / GLOBAL:
            # only explicitly selected requests may move.
            #
            # Everything before the first selected request remains a committed
            # prefix. For vehicles with no mutable request, the whole route is
            # committed.
            if vehicle_id not in mutable_vehicles:
                committed_by_vehicle[vehicle_id] = customers
                continue

            first_mutable_index = None

            for index, customer_id in enumerate(customers):
                if customer_id in mutable_requests:
                    first_mutable_index = index
                    break

            if first_mutable_index is None:
                committed_by_vehicle[vehicle_id] = customers
            else:
                committed_by_vehicle[vehicle_id] = customers[
                    :first_mutable_index
                ]

        return ScopeCommitmentPlan(
            mutable_request_ids=tuple(
                selection.request_ids
            ),
            mutable_vehicle_ids=tuple(
                selection.vehicle_ids
            ),
            committed_by_vehicle=committed_by_vehicle,
        )

def build_commitment_snapshot(
        scenario,
        current_plan,
        selection: ScopeSelection,
        *,
        planning_time_s: float = 0.0,
    ) -> CommitmentSnapshot:
        """
        Convert a scope selection into the evaluator's real immutable
        CommitmentSnapshot.

        Rules:
        - current vehicle state is preserved;
        - onboard cargo remains on its current vehicle;
        - executed physical prefix remains frozen;
        - customers before the first mutable customer remain committed;
        - vehicles with no mutable scope remain completely committed.

        The resulting snapshot is consumed directly by Step7RouteEngine/QPSO.
        """

        mutable_requests = set(selection.request_ids)
        mutable_vehicles = set(selection.vehicle_ids)

        commitments: list[VehicleCommitment] = []

        for vehicle in scenario.fleet:
            vehicle_id = str(vehicle.vehicle_id)

            current_node_id = (
                getattr(vehicle, "current_node_id", None)
                or vehicle.start_node_id
            )

            current_time_s = float(planning_time_s)

            current_load_units = float(
                getattr(vehicle, "remaining_load", 0.0)
                or 0.0
            )

            onboard_request_ids = tuple(
                getattr(vehicle, "onboard_request_ids", ())
                or ()
            )

            frozen_prefix_edge_ids = tuple(
                getattr(vehicle, "executed_prefix_edge_ids", ())
                or ()
            )

            current_route = current_plan.route_for(vehicle_id)
            customer_ids = tuple(
                str(customer_id)
                for customer_id in current_route.customer_ids
            )

            # A vehicle outside the selected scope is completely locked.
            if vehicle_id not in mutable_vehicles:
                committed_customer_ids = customer_ids

            else:
                # Preserve everything before the first mutable request.
                first_mutable_index = None

                for index, customer_id in enumerate(customer_ids):
                    if customer_id in mutable_requests:
                        first_mutable_index = index
                        break

                if first_mutable_index is None:
                    committed_customer_ids = customer_ids
                else:
                    committed_customer_ids = customer_ids[
                        :first_mutable_index
                    ]

            commitments.append(
                VehicleCommitment(
                    vehicle_id=vehicle_id,
                    current_node_id=current_node_id,
                    current_time_s=current_time_s,
                    current_load_units=current_load_units,
                    onboard_request_ids=onboard_request_ids,
                    frozen_prefix_edge_ids=frozen_prefix_edge_ids,
                    committed_customer_ids=committed_customer_ids,
                )
            )

        return CommitmentSnapshot(
            tuple(commitments)
        )

def build_commitment_snapshot(
    scenario,
    current_plan,
    selection: ScopeSelection,
    *,
    planning_time_s: float = 0.0,
) -> CommitmentSnapshot:
    """
    Convert a Step-15 scope selection into the repository's real
    CommitmentSnapshot.

    The current RoutePlan is never mutated.

    For every vehicle:
      - vehicles outside the selected scope are completely locked;
      - vehicles inside the scope keep the prefix before the first
        mutable request;
      - current simulation state is preserved;
      - onboard requests remain onboard;
      - executed physical prefixes remain frozen.
    """

    mutable_requests = set(selection.request_ids)
    mutable_vehicles = set(selection.vehicle_ids)

    commitments: list[VehicleCommitment] = []

    for vehicle in scenario.fleet:
        vehicle_id = vehicle.vehicle_id

        current_node_id = getattr(
            vehicle,
            "current_node_id",
            None,
        )

        if current_node_id is None:
            current_node_id = vehicle.start_node_id

        current_time_s = float(
            getattr(
                vehicle,
                "current_time_s",
                planning_time_s,
            )
            or planning_time_s
        )

        current_load_units = float(
            getattr(
                vehicle,
                "remaining_load",
                0.0,
            )
            or 0.0
        )

        onboard_request_ids = tuple(
            getattr(
                vehicle,
                "onboard_request_ids",
                (),
            )
            or ()
        )

        frozen_prefix_edge_ids = tuple(
            getattr(
                vehicle,
                "executed_prefix_edge_ids",
                (),
            )
            or ()
        )

        route = current_plan.route_for(vehicle_id)

        customer_ids = tuple(
            route.customer_ids
        )

        # Vehicle is outside the mutable scope:
        # preserve its complete current customer sequence.
        if vehicle_id not in mutable_vehicles:
            committed_customer_ids = customer_ids

        else:
            # Preserve the prefix before the first mutable request.
            first_mutable_index = None

            for index, customer_id in enumerate(customer_ids):
                if customer_id in mutable_requests:
                    first_mutable_index = index
                    break

            if first_mutable_index is None:
                committed_customer_ids = customer_ids
            else:
                committed_customer_ids = customer_ids[
                    :first_mutable_index
                ]

        # Onboard work must not simultaneously appear as future committed
        # customer work.
        committed_customer_ids = tuple(
            customer_id
            for customer_id in committed_customer_ids
            if customer_id not in onboard_request_ids
        )

        commitments.append(
            VehicleCommitment(
                vehicle_id=vehicle_id,
                current_node_id=current_node_id,
                current_time_s=current_time_s,
                current_load_units=current_load_units,
                onboard_request_ids=onboard_request_ids,
                frozen_prefix_edge_ids=frozen_prefix_edge_ids,
                committed_customer_ids=committed_customer_ids,
            )
        )

    return CommitmentSnapshot(
        tuple(commitments)
    )