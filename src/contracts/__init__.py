"""SIH26137 contract layer -- V1.1 (Pydantic, runtime-validated, frozen).

Every object in this package is:
  - typed        (dedicated ID types from core_types.py, no bare str
                   fallbacks for anything that already has an ID type)
  - validated    (invalid construction raises pydantic.ValidationError,
                   not "raises eventually, somewhere downstream")
  - immutable    (frozen=True, and Mapping-typed fields use ImmutableStrMap
                   instead of plain dict so nested mutation is also blocked)
  - serializable (model_dump_json() / model_validate_json() on every model,
                   inherited from Pydantic -- no bespoke JSON code needed
                   per owner)

Supersedes the V1 dataclass-based contract layer. The architecture (one
frozen contract per pipeline stage, environment truth kept structurally
separate from Observation) is unchanged; only the enforcement mechanism is
new.
"""

from .core_types import (
    CongestionIntensity,
    CustomerId,
    DecisionId,
    DemandUnits,
    DistanceM,
    EpisodeId,
    EventId,
    ForecastVersion,
    GraphVersion,
    ModelPromotionStatus,
    ModelVersion,
    NodeKind,
    ParentRoadId,
    PolicyVersion,
    RequestStatus,
    RoadClass,
    RoadEdgeId,
    RoadNodeId,
    RouteVersion,
    ScalerVersion,
    ScenarioId,
    SchemaVersion,
    ScopeAction,
    SolveStatus,
    SpeedMps,
    TargetKind,
    TimeS,
    VehicleId,
)
from .decision import ScopeDecision
from .environment import EdgeTruth, EnvironmentTruth, HiddenEvent
from .forecast import Forecast
from .model_pair import ModelPair
from .observation import (
    EdgeObservation,
    Observation,
    VehicleObservation,
    VisibleEvent,
    VisibleJob,
)
from .routing import RoutePlan, StopLeg, VehicleRoute
from .scenario import (
    CoordinateTransform,
    RandomSeeds,
    Request,
    RoadEdge,
    RoadNode,
    Scenario,
    Units,
    Vehicle,
)
from .solve import ConstraintViolation, SearchTrace, SolveResult, ValidationResult

__all__ = [
    # Core types
    "ScenarioId", "RoadEdgeId", "RoadNodeId", "ParentRoadId", "CustomerId",
    "VehicleId", "EpisodeId", "EventId", "DecisionId", "RouteVersion",
    "GraphVersion", "ForecastVersion", "PolicyVersion", "ModelVersion",
    "SchemaVersion", "ScalerVersion", "DistanceM", "TimeS", "SpeedMps",
    "DemandUnits", "CongestionIntensity",
    "NodeKind", "RoadClass", "RequestStatus", "TargetKind", "ScopeAction",
    "SolveStatus", "ModelPromotionStatus",

    # Scenario
    "Scenario", "RoadNode", "RoadEdge", "Request", "Vehicle", "Units",
    "CoordinateTransform", "RandomSeeds",

    # Observation
    "Observation", "EdgeObservation", "VehicleObservation", "VisibleJob",
    "VisibleEvent",

    # Forecast
    "Forecast",

    # Routing
    "RoutePlan", "StopLeg", "VehicleRoute",

    # Decision
    "ScopeDecision",

    # Solve
    "SolveResult", "ValidationResult", "SearchTrace", "ConstraintViolation",

    # Model pair
    "ModelPair",

    # Environment truth (kept import-isolated from Observation above)
    "HiddenEvent", "EdgeTruth", "EnvironmentTruth",
]
