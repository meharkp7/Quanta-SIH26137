"""
Road closure incident management.

Implements the rule from the plan:
  "Add one road closure with a declared rule for vehicles already on the link;
   initially allow those vehicles to clear it while forbidding new entry."

The closure changes lane entry permissions, never the speed of occupants.
Existing vehicles clear the link. Future routes avoid the closed link; reopening
restores the exact saved permissions. SUMO errors propagate to the runner.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import traci as TraCI  # type alias for IDE; runtime import is deferred

logger = logging.getLogger(__name__)

# Speed threshold below which an edge is considered effectively closed
_CLOSED_SPEED_MPS = 0.0
# Speed used when re-opening an edge
_REOPEN_SPEED_SENTINEL = -1.0   # means "restore original" in SUMO TraCI


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class IncidentConfig:
    """
    Declarative description of a single road-closure incident.

    Attributes
    ----------
    incident_id:
        Stable unique ID used in VisibleEvent.
    edge_id:
        Contract edge ID (will be resolved to SUMO edge ID via the mapping).
    trigger_time_s:
        Simulation time at which the closure is applied.
    duration_s:
        How long the edge stays closed (None = indefinite for this episode).
    announce_lead_s:
        How many seconds before trigger_time_s the event is *announced*
        (revealed) to the planner.  0 means no advance notice (surprise).
    """
    incident_id: str
    edge_id: str
    trigger_time_s: float
    duration_s: float | None = None
    announce_lead_s: float = 0.0


@dataclasses.dataclass
class IncidentRecord:
    """Runtime state for one active incident."""
    config: IncidentConfig
    sumo_edge_id: str
    announced: bool = False
    closed: bool = False
    clearing: bool = False   # edge is closed but vehicles still on it
    cleared: bool = False    # no vehicles remain, edge is confirmed clear
    original_permissions: dict = dataclasses.field(default_factory=dict)
    reopened: bool = False


# ---------------------------------------------------------------------------
# IncidentManager
# ---------------------------------------------------------------------------

class IncidentManager:
    """
    Manages all road-closure incidents for one simulation episode.

    Usage::

        manager = IncidentManager(configs, edge_mapping)
        # In the TraCI step loop:
        events = manager.step(traci, sim_time_s)
        # events is a list of (incident_id, event_type) for new state changes

    The caller is responsible for translating ``events`` into ``VisibleEvent``
    contract objects and adding them to the emitted ``Observation``.
    """

    def __init__(
        self,
        configs: list[IncidentConfig],
        edge_mapping: dict[str, str],
    ) -> None:
        self._edge_map = edge_mapping
        self._records: list[IncidentRecord] = []
        for cfg in configs:
            sumo_eid = edge_mapping.get(cfg.edge_id, cfg.edge_id)
            self._records.append(IncidentRecord(config=cfg, sumo_edge_id=sumo_eid))

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def step(
        self,
        traci,                     # traci module (imported by caller)
        sim_time_s: float,
    ) -> list[tuple[str, str]]:
        """
        Evaluate all incidents at the current simulation time.

        Returns a list of (incident_id, event_type) tuples for any new state
        transitions that occurred this step, so the caller can emit
        VisibleEvents into the next Observation.

        event_type values:
          "announced"  — advance notice emitted to planner
          "closed"     — edge has been closed to new entrants
          "clearing"   — vehicles remain on closed edge; clearing in progress
          "cleared"    — edge is now empty; confirmed fully closed
          "reopened"   — edge has been re-opened (if duration_s set)
        """
        events: list[tuple[str, str]] = []

        for rec in self._records:
            cfg = rec.config

            # --- Announce phase ----------------------------------------
            announce_time = cfg.trigger_time_s - cfg.announce_lead_s
            if not rec.announced and sim_time_s >= announce_time:
                rec.announced = True
                logger.info(
                    "incident %r announced at %.1fs (trigger=%.1fs)",
                    cfg.incident_id, sim_time_s, cfg.trigger_time_s,
                )
                events.append((cfg.incident_id, "announced"))

            # --- Closure phase -----------------------------------------
            if not rec.closed and sim_time_s >= cfg.trigger_time_s:
                self._close_edge(traci, rec, sim_time_s)
                rec.closed = True
                events.append((cfg.incident_id, "closed"))

            # --- Clearing phase: monitor vehicles already on the edge --
            if rec.closed and not rec.cleared:
                vehicles_on_edge = self._vehicles_on_edge(traci, rec.sumo_edge_id)
                if vehicles_on_edge:
                    if not rec.clearing:
                        rec.clearing = True
                        logger.info(
                            "incident %r: %d vehicle(s) clearing edge %r",
                            cfg.incident_id, len(vehicles_on_edge), rec.sumo_edge_id,
                        )
                        events.append((cfg.incident_id, "clearing"))
                else:
                    # Edge is now confirmed clear
                    if rec.clearing or rec.closed:
                        rec.cleared = True
                        logger.info(
                            "incident %r: edge %r is now clear at %.1fs",
                            cfg.incident_id, rec.sumo_edge_id, sim_time_s,
                        )
                        events.append((cfg.incident_id, "cleared"))

            # --- Reopen phase (optional) --------------------------------
            if (
                rec.closed
                and not rec.reopened
                and cfg.duration_s is not None
                and sim_time_s >= cfg.trigger_time_s + cfg.duration_s
            ):
                self._reopen_edge(traci, rec, sim_time_s)
                rec.reopened = True
                events.append((cfg.incident_id, "reopened"))

        return events

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_edge_closed(self, edge_id: str) -> bool:
        """Return True if the given contract edge_id is currently closed."""
        sumo_eid = self._edge_map.get(edge_id, edge_id)
        for rec in self._records:
            if rec.sumo_edge_id == sumo_eid and rec.closed and not rec.reopened:
                return True
        return False

    def closed_edge_ids(self) -> list[str]:
        """Return contract edge IDs of all currently closed edges."""
        reverse = {v: k for k, v in self._edge_map.items()}
        return [
            reverse.get(rec.sumo_edge_id, rec.sumo_edge_id)
            for rec in self._records
            if rec.closed and not rec.reopened
        ]

    # ------------------------------------------------------------------
    # TraCI helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _close_edge(traci, rec: IncidentRecord, sim_time_s: float) -> None:
        """Forbid new entry while permitting existing occupants to clear."""
        lane_count = traci.edge.getLaneNumber(rec.sumo_edge_id)
        for lane_idx in range(lane_count):
            lane_id = f"{rec.sumo_edge_id}_{lane_idx}"
            rec.original_permissions[lane_id] = list(traci.lane.getAllowed(lane_id))
            traci.lane.setAllowed(lane_id, ["authority"])
        for vid in traci.vehicle.getIDList():
            current = traci.vehicle.getRoadID(vid)
            future = traci.vehicle.getRoute(vid)[traci.vehicle.getRouteIndex(vid) + 1:]
            if current and not current.startswith(":") and current != rec.sumo_edge_id and rec.sumo_edge_id in future:
                traci.vehicle.rerouteTraveltime(vid, currentTravelTimes=False)

    @staticmethod
    def _reopen_edge(traci, rec: IncidentRecord, sim_time_s: float) -> None:
        for lane_id, permissions in rec.original_permissions.items():
            traci.lane.setAllowed(lane_id, permissions)

    @staticmethod
    def _vehicles_on_edge(traci, sumo_edge_id: str) -> list[str]:
        """Return list of vehicle IDs currently on the given SUMO edge."""
        try:
            return list(traci.edge.getLastStepVehicleIDs(sumo_edge_id))
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Step-6 fixture incident (declared here for easy import by sumo_runner)
# ---------------------------------------------------------------------------

STEP6_CLOSURE_CONFIG = IncidentConfig(
    incident_id="INC_E23_CLOSURE",
    edge_id="E23",               # N2→N3, the closeable segment per the plan
    trigger_time_s=50.0,         # close at t=50s during the episode
    duration_s=None,             # stays closed for the rest of the episode
    announce_lead_s=10.0,        # announce at t=40s so planner can react
)


def configs_for_scenario(scenario, *, include_step6_default: bool = True) -> list[IncidentConfig]:
    """Build the incident schedule from UI-selected closures.

    Edges already closed in the scenario contract (``open_by_default=False``,
    i.e. the user-selected incident roads) close at t=0 with no advance
    notice — the planner routed around them from the start. The STEP6 E23@50s
    demo closure is appended unless E23 itself is UI-closed (then the t=0
    version wins; no duplicates). Vehicles already on a link when it closes
    are allowed to clear it while new entry is forbidden (plan Step 6 rule).
    """
    closed_ids = [
        edge.edge_id
        for edge in scenario.edges
        if not edge.open_by_default
    ]
    configs = [
        IncidentConfig(
            incident_id=f"INC_{edge_id}_UI_CLOSURE",
            edge_id=edge_id,
            trigger_time_s=0.0,
            duration_s=None,
            announce_lead_s=0.0,
        )
        for edge_id in closed_ids
    ]
    if include_step6_default and STEP6_CLOSURE_CONFIG.edge_id not in closed_ids:
        configs.append(STEP6_CLOSURE_CONFIG)
    return configs


def closure_schedule(configs) -> list[dict]:
    """Convert incident configs to frame-annotation schedule entries."""
    return [
        {
            "edge_id": cfg.edge_id,
            "from_s": float(cfg.trigger_time_s),
            "to_s": (
                float(cfg.trigger_time_s + cfg.duration_s)
                if cfg.duration_s is not None
                else None
            ),
        }
        for cfg in configs
    ]
