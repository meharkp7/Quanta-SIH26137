"""Stable identifiers, unit aliases, and enumerations.

Renamed from ``types.py`` in the V1 contract layer: a module literally named
``types.py`` shadows the standard-library ``types`` module the moment this
package's directory (rather than only the package itself) ends up on
``sys.path`` -- a real failure mode, not a theoretical one. Nothing else
about this file's content needed to change for that reason alone.

This module intentionally does NOT get re-exported with ``from .core_types
import *`` anywhere in this package. ``__all__`` is still declared below so
that IF a wildcard import is ever used, it does not also leak ``dataclass``/
``Enum``/``NewType`` style helper names into a consumer's namespace.
"""

from __future__ import annotations

from enum import Enum
from typing import NewType

# ---------------------------------------------------------------------------
# Stable identifiers
#
# These are structural (mypy-time) distinctions only. They do NOT validate
# format at runtime -- a ScenarioId is, underneath, still a str. Runtime
# uniqueness/consistency of identifiers is enforced where it matters inside
# the Pydantic models in this package (e.g. Scenario rejects duplicate node
# IDs), not inside this module.
# ---------------------------------------------------------------------------

ScenarioId = NewType("ScenarioId", str)
RoadEdgeId = NewType("RoadEdgeId", str)
RoadNodeId = NewType("RoadNodeId", str)
ParentRoadId = NewType("ParentRoadId", str)
CustomerId = NewType("CustomerId", str)
VehicleId = NewType("VehicleId", str)
EpisodeId = NewType("EpisodeId", str)
EventId = NewType("EventId", str)
DecisionId = NewType("DecisionId", str)
RouteVersion = NewType("RouteVersion", str)
GraphVersion = NewType("GraphVersion", str)
ForecastVersion = NewType("ForecastVersion", str)
PolicyVersion = NewType("PolicyVersion", str)
ModelVersion = NewType("ModelVersion", str)
SchemaVersion = NewType("SchemaVersion", str)
ScalerVersion = NewType("ScalerVersion", str)


# ---------------------------------------------------------------------------
# Internal units
#
# Project-wide convention:
#   distance -> meters
#   time     -> seconds
#   speed    -> meters / second
#   demand   -> abstract load units unless a physical conversion exists
# ---------------------------------------------------------------------------

DistanceM = float
TimeS = float
SpeedMps = float
DemandUnits = float
CongestionIntensity = float


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeKind(str, Enum):
    JUNCTION = "junction"
    DEPOT = "depot"
    CUSTOMER_ACCESS = "customer_access"
    OTHER = "other"


class RoadClass(str, Enum):
    LOCAL = "local"
    COLLECTOR = "collector"
    ARTERIAL = "arterial"
    ACCESS_CONNECTOR = "access_connector"


class RequestStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    ONBOARD = "onboard"
    SERVED = "served"
    UNASSIGNED = "unassigned"
    CANCELLED = "cancelled"


class TargetKind(str, Enum):
    SPEED_PROXY = "speed_proxy"
    REALIZED_TRAVERSAL = "realized_traversal"


class ScopeAction(str, Enum):
    KEEP = "KEEP"
    LOCAL = "LOCAL"
    VEHICLE = "VEHICLE"
    REGIONAL = "REGIONAL"
    GLOBAL = "GLOBAL"


class SolveStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIME_LIMIT = "time_limit"
    NO_FEASIBLE_INCUMBENT = "no_feasible_incumbent"
    ERROR = "error"


class ModelPromotionStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


__all__ = [
    "ScenarioId", "RoadEdgeId", "RoadNodeId", "ParentRoadId", "CustomerId",
    "VehicleId", "EpisodeId", "EventId", "DecisionId", "RouteVersion",
    "GraphVersion", "ForecastVersion", "PolicyVersion", "ModelVersion",
    "SchemaVersion", "ScalerVersion",
    "DistanceM", "TimeS", "SpeedMps", "DemandUnits", "CongestionIntensity",
    "NodeKind", "RoadClass", "RequestStatus", "TargetKind", "ScopeAction",
    "SolveStatus", "ModelPromotionStatus",
]
