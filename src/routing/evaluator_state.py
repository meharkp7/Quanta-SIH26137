from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Iterable, Mapping

from src.contracts.core_types import (
    CustomerId,
    RoadEdgeId,
    RoadNodeId,
    TimeS,
    VehicleId,
)


@dataclass(frozen=True)
class EvaluationConstraints:
    """
    Immutable policy controlling what the evaluator enforces.

    This object deliberately contains policy only. It does not contain
    optimization logic and does not mutate evaluator/network state.
    """

    enforce_capacity: bool = True
    enforce_time_windows: bool = True
    enforce_releases: bool = True
    require_depot_return: bool = True
    require_commitments: bool = True
    allow_waiting: bool = True
    require_all_requests_served: bool = False

    load_tolerance: float = 1e-9
    time_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        for name in ("load_tolerance", "time_tolerance"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative"
                )

    @classmethod
    def from_config(cls, config: Any) -> "EvaluationConstraints":
        return cls(
            allow_waiting=bool(config.allow_waiting),
        )


@dataclass(frozen=True)
class VehicleCommitment:
    """
    Immutable snapshot of the state that replanning is not allowed to undo.

    current_node_id/current_time_s/current_load_units describe the state
    from which the mutable continuation begins.

    frozen_prefix_edge_ids are historical/committed physical edges. They are
    validated for consistency but are NOT re-executed during continuation.

    committed_customer_ids are the customer visits whose relative prefix
    commitment must be preserved.

    onboard_request_ids represent requests already onboard at the snapshot.
    Their load is assumed to already be included in current_load_units.
    """

    vehicle_id: VehicleId
    current_node_id: RoadNodeId
    current_time_s: TimeS
    current_load_units: float

    onboard_request_ids: tuple[CustomerId, ...] = ()
    frozen_prefix_edge_ids: tuple[RoadEdgeId, ...] = ()
    committed_customer_ids: tuple[CustomerId, ...] = ()

    def __post_init__(self) -> None:
        current_time = float(self.current_time_s)
        current_load = float(self.current_load_units)

        if not isfinite(current_time):
            raise ValueError("current_time_s must be finite")

        if current_time < 0.0:
            raise ValueError("current_time_s must be non-negative")

        if not isfinite(current_load):
            raise ValueError("current_load_units must be finite")

        if current_load < 0.0:
            raise ValueError(
                "current_load_units must be non-negative"
            )

        if len(set(self.onboard_request_ids)) != len(
            self.onboard_request_ids
        ):
            raise ValueError(
                "onboard_request_ids contains duplicates"
            )

        if len(set(self.frozen_prefix_edge_ids)) != len(
            self.frozen_prefix_edge_ids
        ):
            raise ValueError(
                "frozen_prefix_edge_ids contains duplicates"
            )

        if len(set(self.committed_customer_ids)) != len(
            self.committed_customer_ids
        ):
            raise ValueError(
                "committed_customer_ids contains duplicates"
            )

        onboard = set(self.onboard_request_ids)
        committed = set(self.committed_customer_ids)

        overlap = onboard & committed
        if overlap:
            raise ValueError(
                "a request cannot simultaneously be represented as "
                "onboard and as a future committed customer: "
                f"{sorted(overlap, key=str)}"
            )


@dataclass(frozen=True)
class CommitmentSnapshot:
    """
    Immutable deterministic collection of vehicle commitments.
    """

    vehicles: tuple[VehicleCommitment, ...] = ()

    _by_vehicle: Mapping[
        VehicleId,
        VehicleCommitment,
    ] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        by_vehicle: dict[
            VehicleId,
            VehicleCommitment,
        ] = {}

        for commitment in self.vehicles:
            if commitment.vehicle_id in by_vehicle:
                raise ValueError(
                    "duplicate commitment for vehicle "
                    f"{commitment.vehicle_id!r}"
                )

            by_vehicle[commitment.vehicle_id] = commitment

        object.__setattr__(
            self,
            "_by_vehicle",
            by_vehicle,
        )

    def for_vehicle(
        self,
        vehicle_id: VehicleId,
    ) -> VehicleCommitment | None:
        return self._by_vehicle.get(vehicle_id)

    @classmethod
    def from_scenario(
        cls,
        vehicles: Iterable[Any],
        *,
        default_time_s: TimeS = 0.0,
    ) -> "CommitmentSnapshot":
        """
        Build a commitment snapshot from V1.2 vehicle state.

        Important:
        default_time_s is the evaluator's current planning epoch. We do not
        silently force current_time_s to zero when evaluating a later
        simulation snapshot.
        """

        evaluation_time = float(default_time_s)

        if not isfinite(evaluation_time):
            raise ValueError(
                "default_time_s must be finite"
            )

        if evaluation_time < 0.0:
            raise ValueError(
                "default_time_s must be non-negative"
            )

        result: list[VehicleCommitment] = []

        for vehicle in vehicles:
            current_node = getattr(
                vehicle,
                "current_node_id",
                None,
            )

            if current_node is None:
                current_node = vehicle.start_node_id

            current_load = float(
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

            result.append(
                VehicleCommitment(
                    vehicle_id=vehicle.vehicle_id,
                    current_node_id=current_node,
                    current_time_s=evaluation_time,
                    current_load_units=current_load,
                    onboard_request_ids=onboard_request_ids,
                    frozen_prefix_edge_ids=frozen_prefix_edge_ids,
                )
            )

        return cls(tuple(result))