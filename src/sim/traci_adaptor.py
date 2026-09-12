"""
TraCI stepping loop: records vehicle progress, captures Observation snapshots,
and detects stop/service events.

Bridges SUMO's low-level TraCI API to the application's SimState and the
Observation contract.  All service-state changes flow through SimState;
this module only *detects* events and delegates.

Key invariants:
  * No teleportation: vehicles that are absent from SUMO but not finished
    are logged as errors.
  * Stop detection: a vehicle is considered to be "at a stop" when TraCI
    reports stopState & STOP_FLAG (bit 1 set).
  * Service events: begin_service fires when a vehicle enters a stop;
    complete_service fires when the stop ends (vehicle leaves stopped state).
  * Observations are emitted at every step but callers can subsample.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# SUMO stop-state bit flag (bit 1 = currently stopped at a programmed stop)
_SUMO_STOP_FLAG: int = 1

def _find_sumo_binary(gui: bool = False) -> Path:
    """Resolve the requested SUMO executable cross-platform.

    Resolution order:
      1. SUMO_HOME/bin/<sumo-name> (or .exe on Windows)
      2. executable discovered on PATH
    """
    name = "sumo-gui" if gui else "sumo"
    env_home = os.environ.get("SUMO_HOME")

    if env_home:
        home = Path(env_home)
        for candidate in (
            home / "bin" / name,
            home / "bin" / f"{name}.exe",
        ):
            if candidate.is_file():
                return candidate.resolve()

    executable = shutil.which(name)
    if executable:
        return Path(executable).resolve()

    if not name.endswith(".exe"):
        executable = shutil.which(f"{name}.exe")
        if executable:
            return Path(executable).resolve()

    raise RuntimeError(
        f"{name!r} executable not found. Install SUMO, add its bin "
        "directory to PATH, or set SUMO_HOME to the SUMO root directory."
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VehicleStepData:
    """Raw per-vehicle data collected in one TraCI step."""
    vehicle_id: str
    edge_id: str | None
    lane_id: str | None
    position_m: float          # distance along current edge
    distance_remaining_m: float
    speed_mps: float
    stop_state: int            # raw SUMO stop-state flags
    is_stopped: bool           # (stop_state & _SUMO_STOP_FLAG) != 0


@dataclass
class EdgeStepData:
    """Raw per-edge traffic data collected in one TraCI step."""
    edge_id: str
    mean_speed_mps: float
    occupancy: float           # 0–1
    vehicle_count: int
    halting_count: int


@dataclass
class SimStepOutput:
    """Everything collected in a single simulation step."""
    sim_time_s: float
    vehicles: list[VehicleStepData] = field(default_factory=list)
    edges: list[EdgeStepData] = field(default_factory=list)
    # (vehicle_id, request_id) pairs where service just began this step
    service_begin_events: list[tuple[str, str]] = field(default_factory=list)
    # (vehicle_id, request_id) pairs where service just completed this step
    service_complete_events: list[tuple[str, str]] = field(default_factory=list)
    # Vehicles that teleported (error)
    teleported_vehicle_ids: list[str] = field(default_factory=list)
    arrived_vehicle_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TraCI Adapter
# ---------------------------------------------------------------------------

class TraciAdapter:
    """
    Manages one SUMO simulation run via TraCI.

    Usage::

        adapter = TraciAdapter(
            net_path=net_path,
            rou_path=rou_path,
            sumo_cfg_path=cfg_path,    # optional; generated if None
            edge_ids=list_of_edge_ids, # contract edge IDs to monitor
            stop_to_request_map=...,   # {vehicle_id: [(stop_index, request_id)]}
        )
        with adapter:
            while not adapter.done:
                output = adapter.step()
                # process output ...

    ``stop_to_request_map`` maps each vehicle_id to a list of
    (stop_index, request_id) pairs so the adapter knows which SUMO stop
    corresponds to which delivery request.
    """

    def __init__(
        self,
        net_path: Path,
        rou_path: Path,
        output_dir: Path,
        edge_ids: list[str],
        stop_to_request_map: dict[str, list[tuple[int, str]]],
        *,
        end_time_s: float = 600.0,
        step_length_s: float = 1.0,
        sumo_cfg_path: Path | None = None,
        gui: bool = False,
        service_earliest: dict[str, float] | None = None,
    ) -> None:
        self.net_path = Path(net_path).resolve()
        self.rou_path = Path(rou_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.edge_ids = list(edge_ids)
        self.stop_to_request_map = stop_to_request_map  # vid -> [(idx, rid)]

        self.end_time_s = end_time_s
        self.step_length_s = step_length_s
        self.gui = gui
        self.service_earliest = service_earliest or {}

        # Unique label prevents 'Connection already active' when multiple
        # TraciAdapter instances run in the same Python process (e.g. pytest)
        self._label = f"sumo_{uuid.uuid4().hex[:8]}"

        # Will be set in __enter__
        self._traci = None

        # Stop-state tracking: vehicle_id -> last seen active tripId (or None)
        # Used to detect service BEGIN (None → tripId) and COMPLETE (tripId → None)
        self._prev_active_trip: dict[str, str | None] = {}

        # Set of vehicle IDs SUMO has ever seen (for teleport detection)
        self._known_vehicles: set[str] = set()
        self._previous_active: set[str] = set()

        # Generated or provided config path
        self._sumo_cfg = sumo_cfg_path or self._generate_sumo_config()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TraciAdapter":
        import traci

        binary = str(_find_sumo_binary(self.gui))

        command = [
            binary,
            "-c", str(self._sumo_cfg),
            "--step-length", str(self.step_length_s),
            "--no-step-log",
            "--duration-log.disable",
            "--tripinfo-output", str(self.output_dir / "tripinfo.xml"),
            "--fcd-output", str(self.output_dir / "fcd.xml"),
            "--log", str(self.output_dir / "sumo.log"),
        ]
        if self.gui:
            command.extend(["--start", "--delay", "80"])

        traci.start(
            command,
            label=self._label,
        )
        self._traci = traci.getConnection(self._label)
        logger.info("SUMO started via TraCI (label=%s)", self._label)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._traci is not None:
            try:
                # close() shuts the socket; getConnection(label).close() removes
                # the label from traci's registry so it can be reused.
                self._traci.close()
                logger.info("SUMO connection closed (label=%s)", self._label)
            except Exception as e:
                logger.warning("Error closing TraCI (label=%s): %s", self._label, e)
        return False  # don't suppress exceptions

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        if self._traci is None:
            return True
        return self._traci.simulation.getTime() >= self.end_time_s

    def step(self, on_step: Callable[[SimStepOutput], None] | None = None) -> SimStepOutput:
        """
        Advance simulation by one step and return collected data.

        Optionally call ``on_step(output)`` immediately after collecting
        (useful for inline processing without storing all step outputs).
        """
        traci = self._traci
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        output = SimStepOutput(sim_time_s=sim_time)
        output.arrived_vehicle_ids = list(traci.simulation.getArrivedIDList())

        # ---- Teleportation detection ----------------------------------
        teleporting = set(traci.simulation.getStartingTeleportIDList())
        if teleporting:
            for vid in teleporting:
                logger.error(
                    "TELEPORTATION detected for vehicle %r at %.1fs — "
                    "check route and network for missing edges or impossible turns",
                    vid, sim_time,
                )
            output.teleported_vehicle_ids = list(teleporting)

        # ---- Vehicle data --------------------------------------------
        active_ids = set(traci.vehicle.getIDList())
        lost = self._previous_active - active_ids - set(output.arrived_vehicle_ids) - teleporting
        if lost:
            raise RuntimeError(f"Vehicles disappeared without arrival: {sorted(lost)}")
        self._previous_active = active_ids
        self._known_vehicles.update(active_ids)

        for vid in active_ids:
            edge_id = traci.vehicle.getRoadID(vid) or None
            lane_id = traci.vehicle.getLaneID(vid) or None
            position_m = traci.vehicle.getLanePosition(vid)
            speed_mps = traci.vehicle.getSpeed(vid)
            stop_state = traci.vehicle.getStopState(vid)
            is_stopped = bool(stop_state & _SUMO_STOP_FLAG)

            remaining = 0.0
            if lane_id:
                remaining = max(0.0, traci.lane.getLength(lane_id) - position_m)

            output.vehicles.append(VehicleStepData(
                vehicle_id=vid,
                edge_id=edge_id,
                lane_id=lane_id,
                position_m=position_m,
                distance_remaining_m=remaining,
                speed_mps=speed_mps,
                stop_state=stop_state,
                is_stopped=is_stopped,
            ))

            # ---- Stop/service event detection via tripId -----------------
            # We read the active stop's tripId from SUMO directly, which is the
            # request_id we embedded in the stop XML.  This avoids any
            # stop-index accounting and works regardless of network topology.
            active_trip_id = self._get_active_stop_trip_id(traci, vid)
            if active_trip_id and sim_time < self.service_earliest.get(active_trip_id, 0):
                active_trip_id = None
            prev_trip_id = self._prev_active_trip.get(vid)

            if active_trip_id and active_trip_id != prev_trip_id:
                # Vehicle has arrived at a new stop
                output.service_begin_events.append((vid, active_trip_id))
                logger.debug(
                    "service BEGIN: vehicle=%r request=%r t=%.1f",
                    vid, active_trip_id, sim_time,
                )
            elif prev_trip_id and not active_trip_id:
                # Vehicle just departed from a stop
                output.service_complete_events.append((vid, prev_trip_id))
                logger.debug(
                    "service COMPLETE: vehicle=%r request=%r t=%.1f",
                    vid, prev_trip_id, sim_time,
                )
            self._prev_active_trip[vid] = active_trip_id

        # ---- Edge data -----------------------------------------------
        for eid in self.edge_ids:
            try:
                output.edges.append(EdgeStepData(
                    edge_id=eid,
                    mean_speed_mps=traci.edge.getLastStepMeanSpeed(eid),
                    occupancy=traci.edge.getLastStepOccupancy(eid) / 100.0,
                    vehicle_count=traci.edge.getLastStepVehicleNumber(eid),
                    halting_count=traci.edge.getLastStepHaltingNumber(eid),
                ))
            except Exception as exc:
                logger.debug("Edge %r not found in SUMO: %s", eid, exc)

        if on_step is not None:
            on_step(output)

        return output

    # ------------------------------------------------------------------
    # Stop index detection helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_active_stop_trip_id(traci, vehicle_id: str) -> str | None:
        """Read the current programmed stop only when SUMO reports stopped."""
        if not traci.vehicle.getStopState(vehicle_id) & _SUMO_STOP_FLAG:
            return None
        return traci.vehicle.getStopParameter(vehicle_id, 0, "tripId") or None

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def _generate_sumo_config(self) -> Path:
        """
        Write a minimal .sumocfg file referencing the net and route files.
        """
        cfg_path = self.output_dir / "simulation.sumocfg"
        cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="{self.net_path.as_posix()}"/>
    <route-files value="{self.rou_path.as_posix()}"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="{self.end_time_s}"/>
    <step-length value="{self.step_length_s}"/>
  </time>
  <processing>
    <ignore-route-errors value="false"/>
    <time-to-teleport value="-1"/>
  </processing>
  <report>
    <no-step-log value="true"/>
    <no-warnings value="false"/>
  </report>
</configuration>"""
        cfg_path.write_text(cfg_content)
        return cfg_path