from __future__ import annotations

from pydantic import Field, model_validator

from ._base import Contract
from .core_types import CustomerId, DistanceM, RoadEdgeId, RouteVersion, TimeS, VehicleId


class StopLeg(Contract):
    from_stop_id: str
    to_stop_id: str

    # Physical route corresponding to THIS exact stop leg. Distance, time,
    # and congestion below MUST all refer to this same edge sequence --
    # CONTRACTS.md's Routing Rule exists specifically to forbid combining
    # the shortest-distance path from one route with the shortest-time
    # cost from a different one.
    physical_edge_ids: tuple[RoadEdgeId, ...]

    travel_time_s: TimeS = Field(ge=0)
    distance_m: DistanceM = Field(ge=0)
    congestion_exposure: float = Field(ge=0)

    departure_time_s: TimeS = Field(ge=0)
    arrival_time_s: TimeS = Field(ge=0)

    @model_validator(mode="after")
    def _timing_and_path_consistent(self) -> "StopLeg":
        if self.arrival_time_s < self.departure_time_s:
            raise ValueError(
                f"leg {self.from_stop_id!r}->{self.to_stop_id!r}: "
                f"arrival_time_s ({self.arrival_time_s}) precedes "
                f"departure_time_s ({self.departure_time_s})"
            )
        if len(self.physical_edge_ids) == 0 and self.from_stop_id != self.to_stop_id:
            raise ValueError(
                f"leg {self.from_stop_id!r}->{self.to_stop_id!r}: no "
                "physical_edge_ids given for a leg between distinct stops"
            )
        implied = self.arrival_time_s - self.departure_time_s
        if abs(implied - self.travel_time_s) > 1e-6:
            raise ValueError(
                f"leg {self.from_stop_id!r}->{self.to_stop_id!r}: "
                f"arrival - departure ({implied}s) does not match "
                f"travel_time_s ({self.travel_time_s}s)"
            )
        return self


class VehicleRoute(Contract):
    vehicle_id: VehicleId

    # Customer order decided by optimizer.
    customer_order: tuple[CustomerId, ...]

    # Exact physical road path executed between stops.
    legs: tuple[StopLeg, ...]

    expected_departure_s: TimeS = Field(ge=0)
    expected_return_s: TimeS = Field(ge=0)

    expected_driving_time_s: TimeS = Field(ge=0)
    expected_waiting_time_s: TimeS = Field(ge=0)
    expected_service_time_s: TimeS = Field(ge=0)

    served_request_ids: tuple[CustomerId, ...]

    # Prefix is immutable during replanning.
    frozen_prefix_edge_ids: tuple[RoadEdgeId, ...]

    @model_validator(mode="after")
    def _return_after_departure(self) -> "VehicleRoute":
        if self.expected_return_s < self.expected_departure_s:
            raise ValueError(
                f"vehicle {self.vehicle_id!r}: expected_return_s "
                f"({self.expected_return_s}) precedes expected_departure_s "
                f"({self.expected_departure_s})"
            )
        return self

    @model_validator(mode="after")
    def _served_and_ordered_customers_are_consistent(self) -> "VehicleRoute":
        order_ids = set(self.customer_order)
        if len(order_ids) != len(self.customer_order):
            raise ValueError(
                f"vehicle {self.vehicle_id!r}: customer_order contains "
                "duplicate customer IDs"
            )
        served_not_ordered = set(self.served_request_ids) - order_ids
        if served_not_ordered:
            raise ValueError(
                f"vehicle {self.vehicle_id!r}: served_request_ids contains "
                f"IDs absent from customer_order: {sorted(served_not_ordered)}"
            )
        return self

    @model_validator(mode="after")
    def _legs_chain_and_end_at_or_before_return(self) -> "VehicleRoute":
        if not self.legs:
            return self
        for prev_leg, next_leg in zip(self.legs, self.legs[1:]):
            if prev_leg.to_stop_id != next_leg.from_stop_id:
                raise ValueError(
                    f"vehicle {self.vehicle_id!r}: leg chain is broken "
                    f"between {prev_leg.to_stop_id!r} and "
                    f"{next_leg.from_stop_id!r}"
                )
        if self.legs[0].departure_time_s < self.expected_departure_s - 1e-6:
            raise ValueError(
                f"vehicle {self.vehicle_id!r}: first leg departs before "
                "expected_departure_s"
            )
        if self.legs[-1].arrival_time_s > self.expected_return_s + 1e-6:
            raise ValueError(
                f"vehicle {self.vehicle_id!r}: last leg arrives after "
                "expected_return_s"
            )
        return self


class RoutePlan(Contract):
    scenario_id: str
    state_version: str
    route_version: RouteVersion

    generated_at_s: TimeS = Field(ge=0)

    vehicle_routes: tuple[VehicleRoute, ...]

    objective_value: float

    objective_time: float = Field(ge=0)
    objective_distance: float = Field(ge=0)
    objective_congestion: float = Field(ge=0)
    route_change_penalty: float = Field(ge=0)

    changed_vehicle_ids: tuple[VehicleId, ...]

    # Human-readable method/provenance.
    generated_by: str

    @model_validator(mode="after")
    def _unique_vehicles_and_valid_changed_set(self) -> "RoutePlan":
        route_vehicle_ids = [r.vehicle_id for r in self.vehicle_routes]
        if len(route_vehicle_ids) != len(set(route_vehicle_ids)):
            raise ValueError("vehicle_routes contains duplicate vehicle_id")
        known = set(route_vehicle_ids)
        dangling = [v for v in self.changed_vehicle_ids if v not in known]
        if dangling:
            raise ValueError(
                f"changed_vehicle_ids references vehicles absent from "
                f"vehicle_routes: {dangling}"
            )
        return self
