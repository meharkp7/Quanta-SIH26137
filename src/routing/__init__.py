"""Routing package.

Step 3
------
Independent directed-road feasibility validation.

Step 4
------
Generic initial route construction, physical path construction, route
evaluation, deterministic repair, and pipeline orchestration.
"""

from src.routing.initial_solution import (
    InitialSolutionBuilder,
    InitialSolutionConfig,
    InitialSolutionResult,
    InsertionCandidate,
)

from src.routing.path_builder import (
    DirectedPathBuilder,
    DirectedRoadGraph,
    InvalidTravelTimeError,
    PathNotFoundError,
    PathResult,
    TravelTimeProvider as PathTravelTimeProvider,
)

from src.routing.pipeline import (
    RoutingPipeline,
    RoutingPipelineConfig,
    RoutingPipelineResult,
)

from src.routing.route_evaluator import (
    RouteEvaluationConfig,
    RouteEvaluator,
    RoutePlanEvaluation,
    StopEvaluation,
    VehicleRouteEvaluation,
)

from src.routing.route_plan import (
    RoutePlan,
    StopLocation,
    VehicleRoute,
)

from src.routing.route_repair import (
    RepairAction,
    RouteRepairConfig,
    RouteRepairResult,
    RouteRepairer,
)

from src.routing.route_types import (
    PhysicalRoute,
    PhysicalRoutePlan,
    RouteLeg,
)

from src.routing.validator import (
    EvaluationResult,
    evaluate_scenario,
    shortest_directed_path,
)


__all__ = [
    # Step 3 validator
    "EvaluationResult",
    "evaluate_scenario",
    "shortest_directed_path",

    # Logical routes
    "RoutePlan",
    "VehicleRoute",
    "StopLocation",

    # Physical routes
    "RouteLeg",
    "PhysicalRoute",
    "PhysicalRoutePlan",

    # Path construction
    "DirectedRoadGraph",
    "DirectedPathBuilder",
    "PathResult",
    "PathNotFoundError",
    "InvalidTravelTimeError",
    "PathTravelTimeProvider",

    # Evaluation
    "RouteEvaluator",
    "RouteEvaluationConfig",
    "RoutePlanEvaluation",
    "VehicleRouteEvaluation",
    "StopEvaluation",

    # Initial solution
    "InitialSolutionBuilder",
    "InitialSolutionConfig",
    "InitialSolutionResult",
    "InsertionCandidate",

    # Repair
    "RouteRepairer",
    "RouteRepairConfig",
    "RouteRepairResult",
    "RepairAction",

    # Pipeline
    "RoutingPipeline",
    "RoutingPipelineConfig",
    "RoutingPipelineResult",
]