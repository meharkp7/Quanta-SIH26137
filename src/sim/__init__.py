"""
src/sim — SUMO simulation adapter package.

Public surface
--------------
- SumoExporter   : scenario → net.xml
- RouteBuilder   : RoutePlan → vehicles.rou.xml
- TraciAdapter   : TraCI step loop
- SimState       : delivery state machine
- IncidentManager: road closure management
- run_episode    : one-call orchestrator (sumo_runner)
"""

from .sumo_exporter import SumoExporter, load_mapping
from .route_builder import RouteBuilder
from .traci_adaptor import TraciAdapter, SimStepOutput, VehicleStepData
from .sim_state import SimState, RequestState, VehicleSimState
from .incidents import IncidentManager, IncidentConfig, STEP6_CLOSURE_CONFIG
from .sumo_runner import run_episode, EpisodeResult

__all__ = [
    "SumoExporter",
    "load_mapping",
    "RouteBuilder",
    "TraciAdapter",
    "SimStepOutput",
    "VehicleStepData",
    "SimState",
    "RequestState",
    "VehicleSimState",
    "IncidentManager",
    "IncidentConfig",
    "STEP6_CLOSURE_CONFIG",
    "run_episode",
    "EpisodeResult",
]
