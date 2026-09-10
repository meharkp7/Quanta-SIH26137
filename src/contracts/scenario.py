from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from ._base import Contract
from ._immutable import ImmutableStrMap
from .core_types import (
    CustomerId,
    DemandUnits,
    DistanceM,
    GraphVersion,
    NodeKind,
    RequestStatus,
    RoadClass,
    RoadEdgeId,
    RoadNodeId,
    ScenarioId,
    SchemaVersion,
    SpeedMps,
    TimeS,
    VehicleId,
    ParentRoadId,
)


class Units(Contract):
    distance: str = "m"
    time: str = "s"
    speed: str = "m/s"
    demand: str = "load_units"


class CoordinateTransform(Contract):
    scale: float = Field(gt=0)
    translation_x_m: float
    translation_y_m: float
    description: str


class RandomSeeds(Contract):
    """Every seed is a *separate* stream by design (CONTRACTS.md: "Every
    asynchronous candidate must carry a state version"; the analogous rule
    for randomness is that one global seed is insufficient because
    different algorithms/components consume randomness differently)."""

    scenario_seed: int
    road_seed: int
    traffic_seed: int
    incident_seed: int
    window_seed: int
    optimizer_seed: int
    learning_seed: int


class RoadNode(Contract):
    node_id: RoadNodeId
    x_m: float
    y_m: float
    kind: NodeKind
    zone_id: str
    signalized: bool = False


class RoadEdge(Contract):
    """
    One physical directed road segment.

    Traversal is legal ONLY from `from_node` to `to_node`.

    Reverse traversal requires a separate RoadEdge whose
    from_node/to_node are reversed.

    `lane_count` refers only to this directed movement.
    """

    edge_id: RoadEdgeId
    parent_road_id: ParentRoadId

    from_node: RoadNodeId
    to_node: RoadNodeId

    length_m: DistanceM
    road_class: RoadClass

    speed_limit_mps: SpeedMps

    lane_count: int

    capacity_veh_per_hour: float

    source_edge_id: str | None = None
    provenance: str = "synthetic"

    open_by_default: bool = True

    @property
    def free_flow_time_s(self) -> float:
        """Free-flow traversal time implied by length and speed limit."""
        return float(self.length_m) / float(self.speed_limit_mps)

    @model_validator(mode="after")
    def validate_directed_edge(self):
        if self.from_node == self.to_node:
            raise ValueError(
                "RoadEdge cannot be a self-loop."
            )

        if self.length_m <= 0:
            raise ValueError(
                "length_m must be positive."
            )

        if self.speed_limit_mps <= 0:
            raise ValueError(
                "speed_limit_mps must be positive."
            )

        if self.lane_count <= 0:
            raise ValueError(
                "lane_count must be at least 1."
            )

        if self.capacity_veh_per_hour <= 0:
            raise ValueError(
                "capacity_veh_per_hour must be positive."
            )

        return self
    

class Request(Contract):
    request_id: CustomerId

    original_customer_id: str
    original_x: float
    original_y: float

    access_node_id: RoadNodeId
    access_distance_m: DistanceM = Field(ge=0)

    demand: DemandUnits = Field(ge=0)

    known_at_s: TimeS = Field(ge=0)
    release_s: TimeS = Field(ge=0)

    earliest_service_start_s: TimeS = Field(ge=0)
    latest_service_start_s: TimeS = Field(ge=0)
    service_duration_s: TimeS = Field(ge=0)

    status: RequestStatus = RequestStatus.PENDING

    @model_validator(mode="after")
    def _window_and_release_are_consistent(self) -> "Request":
        if self.latest_service_start_s < self.earliest_service_start_s:
            raise ValueError(
                f"request {self.request_id!r}: latest_service_start_s "
                f"({self.latest_service_start_s}) < earliest_service_start_s "
                f"({self.earliest_service_start_s})"
            )
        if self.release_s < self.known_at_s:
            raise ValueError(
                f"request {self.request_id!r}: release_s cannot precede "
                "known_at_s"
            )
        if self.earliest_service_start_s < self.release_s:
            raise ValueError(
                f"request {self.request_id!r}: earliest_service_start_s "
                "cannot precede release_s (a job cannot be served before "
                "it is released)"
            )
        return self


class Vehicle(Contract):
    vehicle_id: VehicleId

    capacity: DemandUnits = Field(gt=0)

    start_node_id: RoadNodeId
    depot_node_id: RoadNodeId

    # Dynamic fields are included here because the fleet contract must
    # preserve state across replanning.
    current_edge_id: RoadEdgeId | None = None
    current_node_id: RoadNodeId | None = None

    distance_remaining_m: DistanceM = Field(default=0.0, ge=0)

    onboard_request_ids: tuple[CustomerId, ...] = ()
    remaining_load: DemandUnits = Field(default=0.0, ge=0)

    executed_prefix_edge_ids: tuple[RoadEdgeId, ...] = ()

    @model_validator(mode="after")
    def _load_within_capacity(self) -> "Vehicle":
        if self.remaining_load > self.capacity + 1e-9:
            raise ValueError(
                f"vehicle {self.vehicle_id!r}: remaining_load "
                f"({self.remaining_load}) exceeds capacity ({self.capacity})"
            )
        return self


class Scenario(Contract):
    schema_version: SchemaVersion
    scenario_id: ScenarioId

    source_name: str
    source_checksum: str | None

    units: Units
    coordinate_transform: CoordinateTransform | None

    graph_version: GraphVersion

    nodes: tuple[RoadNode, ...]
    edges: tuple[RoadEdge, ...]
    requests: tuple[Request, ...]
    fleet: tuple[Vehicle, ...]

    seeds: RandomSeeds
    generator_version: str

    dataset_split: str
    configuration_version: str

    field_provenance: ImmutableStrMap = Field(default_factory=ImmutableStrMap)

    # Optional parent benchmark identity for derived scenarios. Presence of
    # this field is what CONTRACTS.md's Benchmark Integrity rule requires
    # for any road-network/traffic transformation of an original instance;
    # this contract cannot force callers to set it, but it can (and does,
    # below) forbid duplicate/dangling identifiers within the scenario.
    parent_instance_id: str | None = None

    @field_validator("nodes")
    @classmethod
    def _unique_node_ids(cls, v: tuple[RoadNode, ...]) -> tuple[RoadNode, ...]:
        ids = [n.node_id for n in v]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"duplicate node_id(s): {sorted(dupes)}")
        return v

    @field_validator("edges")
    @classmethod
    def _unique_edge_ids(cls, v: tuple[RoadEdge, ...]) -> tuple[RoadEdge, ...]:
        ids = [e.edge_id for e in v]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"duplicate edge_id(s): {sorted(dupes)}")
        return v

    @field_validator("requests")
    @classmethod
    def _unique_request_ids(cls, v: tuple[Request, ...]) -> tuple[Request, ...]:
        ids = [r.request_id for r in v]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"duplicate request_id(s): {sorted(dupes)}")
        return v

    @field_validator("fleet")
    @classmethod
    def _unique_vehicle_ids(cls, v: tuple[Vehicle, ...]) -> tuple[Vehicle, ...]:
        ids = [f.vehicle_id for f in v]
        if len(ids) != len(set(ids)):
            dupes = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"duplicate vehicle_id(s): {sorted(dupes)}")
        return v

    @model_validator(mode="after")
    def _edges_and_requests_reference_known_nodes(self) -> "Scenario":
        node_ids = {n.node_id for n in self.nodes}
        for e in self.edges:
            if e.from_node not in node_ids or e.to_node not in node_ids:
                raise ValueError(
                    f"edge {e.edge_id!r} references an unknown node "
                    f"({e.from_node!r} -> {e.to_node!r})"
                )
        for r in self.requests:
            if r.access_node_id not in node_ids:
                raise ValueError(
                    f"request {r.request_id!r} access_node_id "
                    f"{r.access_node_id!r} is not a known node"
                )
        for veh in self.fleet:
            if veh.start_node_id not in node_ids or veh.depot_node_id not in node_ids:
                raise ValueError(
                    f"vehicle {veh.vehicle_id!r} references an unknown "
                    "start/depot node"
                )
        return self
