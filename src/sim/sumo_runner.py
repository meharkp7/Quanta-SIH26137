"""
Wires together:
  1. SumoExporter    — scenario → net.xml
  2. RouteBuilder    — RoutePlan → vehicles.rou.xml
  3. TraciAdapter    — TraCI step loop
  4. SimState        — delivery state machine
  5. IncidentManager — road closure logic

This is the entry point for running one complete episode of the fixture.
Import and call ``run_episode()`` from tests or the CLI.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Summary of one completed simulation episode."""
    scenario_id: str
    total_steps: int
    sim_end_time_s: float
    wall_clock_s: float

    delivered_count: int
    failed_count: int
    pending_count: int

    teleport_events: int
    service_begin_events: int
    service_complete_events: int

    incident_events: list[tuple[str, str]]   # (incident_id, event_type)

    vehicle_summaries: list[dict]
    request_summaries: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_episode(
    scenario: Any,                 # src.contracts.scenario.Scenario
    route_plan: Any,               # src.contracts.routing.RoutePlan
    output_dir: Path,
    *,
    end_time_s: float = 400.0,
    step_length_s: float = 1.0,
    gui: bool = False,
    log_every_n_steps: int = 10,
    on_step: Callable[[Any], None] | None = None,
) -> EpisodeResult:
    """
    Run one simulation episode for the given scenario and route plan.

    Parameters
    ----------
    scenario:
        Loaded Scenario contract (the step-3 fixture or any compatible one).
    route_plan:
        A RoutePlan produced by P2/P3's optimizer for this scenario.
    output_dir:
        Directory where SUMO network files, logs and outputs are written.
    end_time_s:
        Simulation end time in seconds.
    step_length_s:
        SUMO step length in seconds.
    gui:
        Launch sumo-gui instead of headless sumo (for debugging).
    log_every_n_steps:
        How often to emit a summary log line.

    Returns
    -------
    EpisodeResult with delivery outcomes and event counts.
    """
    from .sumo_exporter import SumoExporter
    from .route_builder import RouteBuilder
    from .traci_adaptor import TraciAdapter
    from .sim_state import SimState
    from .incidents import IncidentManager, STEP6_CLOSURE_CONFIG

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.monotonic()

    # ------------------------------------------------------------------ 1
    # Export network
    # ------------------------------------------------------------------ 1
    logger.info("Step 6 — exporting SUMO network …")
    exporter = SumoExporter(scenario, output_dir=output_dir / "network")
    net_path = exporter.export()
    sumo_mapping = {
        "edge_mapping": exporter.edge_mapping,
        "node_mapping": exporter.node_mapping,
    }
    logger.info("Network written to %s", net_path)

    # ------------------------------------------------------------------ 2
    # Build routes
    # ------------------------------------------------------------------ 2
    logger.info("Step 6 — building vehicle routes …")
    builder = RouteBuilder(scenario, route_plan, sumo_mapping)
    rou_path = builder.build(output_dir=output_dir / "routes")
    logger.info("Routes written to %s", rou_path)

    # ------------------------------------------------------------------ 3
    # Build stop→request mapping for TraCI adapter
    # ------------------------------------------------------------------ 3
    stop_to_request_map = _build_stop_request_map(route_plan)

    # ------------------------------------------------------------------ 4
    # Initialise state and incidents
    # ------------------------------------------------------------------ 4
    sim_state = SimState(scenario)
    sim_state.load_route_cargo(route_plan)

    incident_mgr = IncidentManager(
        configs=[STEP6_CLOSURE_CONFIG],
        edge_mapping=exporter.edge_mapping,
    )

    # Edge IDs to monitor (all contract edges)
    edge_ids = [e.edge_id for e in scenario.edges]

    # ------------------------------------------------------------------ 5
    # Run the TraCI loop
    # ------------------------------------------------------------------ 5
    logger.info("Step 6 — starting SUMO …")
    step_count = 0
    all_incident_events: list[tuple[str, str]] = []
    total_teleports = 0
    total_service_begins = 0
    total_service_completes = 0

    progress = []
    sumo_log = output_dir / "sumo_output" / "sumo.log"
    try:
        with TraciAdapter(
            net_path=net_path,
            rou_path=rou_path,
            output_dir=output_dir / "sumo_output",
            edge_ids=edge_ids,
            stop_to_request_map=stop_to_request_map,
            end_time_s=end_time_s,
            step_length_s=step_length_s,
            gui=gui,
            service_earliest={r.request_id: max(r.earliest_service_start_s, r.release_s) for r in scenario.requests},
        ) as adapter:
            while not adapter.done:
                step_out = adapter.step()
                progress.append(asdict(step_out))
                if on_step is not None:
                    on_step(step_out)
                sim_time = step_out.sim_time_s
                step_count += 1

                # ---- Update delivery state from step output ----------------
                for vdata in step_out.vehicles:
                    if vdata.vehicle_id not in stop_to_request_map:
                        continue
                    sim_state.update_vehicle_position(
                        vdata.vehicle_id,
                        current_edge_id=vdata.edge_id,
                        current_node_id=None,   # TraCI gives edge, not node
                        distance_remaining_m=vdata.distance_remaining_m,
                        sim_time_s=sim_time,
                    )

                for vid, rid in step_out.service_begin_events:
                    sim_state.begin_service(vid, rid, sim_time)
                    total_service_begins += 1

                for vid, rid in step_out.service_complete_events:
                    sim_state.complete_service(vid, rid, sim_time)
                    total_service_completes += 1

                for vid in step_out.arrived_vehicle_ids:
                    sim_state.mark_vehicle_finished(vid)

                total_teleports += len(step_out.teleported_vehicle_ids)

                # ---- Road closure incidents --------------------------------
                inc_events = incident_mgr.step(adapter._traci, sim_time)
                all_incident_events.extend(inc_events)
                for inc_id, evt_type in inc_events:
                    logger.info(
                        "INCIDENT %r → %s at t=%.1fs", inc_id, evt_type, sim_time
                    )

                # ---- Periodic status log ----------------------------------
                if step_count % log_every_n_steps == 0:
                    summary = sim_state.summary()
                    logger.info(
                        "t=%.1fs | delivered=%d | pending=%d | failed=%d | "
                        "vehicles=%s",
                        sim_time,
                        summary["delivered"],
                        summary["pending"],
                        summary["failed"],
                        summary["vehicles"],
                    )

                # ---- Early exit if all delivered -------------------------
                if sim_state.all_delivered() and all(v.state.value == "finished" for v in sim_state.snapshot_vehicles()):
                    logger.info(
                        "All requests delivered at t=%.1fs — terminating early",
                        sim_time,
                    )
                    break

    except Exception as exc:
        # Diagnose SUMO startup/runtime failures clearly
        diag = ""
        if sumo_log.exists():
            try:
                diag = sumo_log.read_text(errors="replace")[-2000:]
            except Exception:
                pass
        raise RuntimeError(
            f"SUMO episode failed: {exc}\n"
            f"SUMO log (last 2000 chars):\n{diag}"
        ) from exc

    wall_end = time.monotonic()

    # ------------------------------------------------------------------ 6
    # Collect results
    # ------------------------------------------------------------------ 6
    req_snapshots = sim_state.snapshot_requests()
    veh_snapshots = sim_state.snapshot_vehicles()

    result = EpisodeResult(
        scenario_id=scenario.scenario_id,
        total_steps=step_count,
        sim_end_time_s=step_count * step_length_s,
        wall_clock_s=round(wall_end - wall_start, 3),
        delivered_count=sim_state.delivered_count(),
        failed_count=sim_state.failed_count(),
        pending_count=sim_state.pending_count(),
        teleport_events=total_teleports,
        service_begin_events=total_service_begins,
        service_complete_events=total_service_completes,
        incident_events=all_incident_events,
        vehicle_summaries=[
            {
                "vehicle_id": v.vehicle_id,
                "state": v.state.value,
                "delivered_count": v.delivered_count,
                "remaining_load": v.remaining_load,
            }
            for v in veh_snapshots
        ],
        request_summaries=[
            {
                "request_id": r.request_id,
                "state": r.state.value,
                "assigned_vehicle_id": r.assigned_vehicle_id,
                "service_start_s": r.service_start_s,
                "service_end_s": r.service_end_s,
                "was_late": r.was_late,
            }
            for r in req_snapshots
        ],
    )

    (output_dir / "progress.json").write_text(json.dumps(progress, indent=2))

    # Write results JSON
    results_path = output_dir / "episode_result.json"
    results_path.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("Episode complete. Results → %s", results_path)

    _log_episode_summary(result)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_stop_request_map(route_plan: Any) -> dict[str, list[tuple[int, str]]]:
    """
    Build the stop_index → request_id mapping needed by TraciAdapter.

    For each vehicle route, the customer_order gives the sequence of stops
    (index 0 = first stop, index 1 = second stop, …).
    """
    mapping: dict[str, list[tuple[int, str]]] = {}
    for vr in route_plan.vehicle_routes:
        mapping[vr.vehicle_id] = [
            (idx, rid) for idx, rid in enumerate(vr.customer_order)
        ]
    return mapping


def _check_window_violations(sim_state: Any, scenario: Any, sim_time: float) -> None:
    """
    Mark requests as FAILED if their latest_service_start_s has passed and
    they are still PENDING.
    """
    from .sim_state import RequestState

    req_map = {r.request_id: r for r in scenario.requests}
    for snap in sim_state.snapshot_requests():
        if snap.state == RequestState.PENDING:
            req = req_map.get(snap.request_id)
            if req and sim_time > req.latest_service_start_s + req.service_duration_s:
                sim_state.mark_failed(
                    snap.request_id,
                    reason=f"window expired at t={sim_time:.1f}s",
                    sim_time_s=sim_time,
                )


def _log_episode_summary(result: EpisodeResult) -> None:
    logger.info(
        "\n"
        "═══════════════════════════════════════\n"
        " EPISODE COMPLETE — %s\n"
        "═══════════════════════════════════════\n"
        " Steps:           %d\n"
        " Sim time:        %.1f s\n"
        " Wall clock:      %.1f s\n"
        " Delivered:       %d\n"
        " Failed:          %d\n"
        " Pending (unserved): %d\n"
        " Teleport events: %d\n"
        " Incident events: %d\n"
        "═══════════════════════════════════════",
        result.scenario_id,
        result.total_steps,
        result.sim_end_time_s,
        result.wall_clock_s,
        result.delivered_count,
        result.failed_count,
        result.pending_count,
        result.teleport_events,
        len(result.incident_events),
    )
