"""SUMO-backed causal episode generator for Step 11.

The runner executes the declared event tape inside SUMO and records only
real TraCI outcomes as observations/trajectories. Forecasts and matured labels
are produced afterwards from those observations, never from simulator weights.
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Iterable

from src.contracts.scenario import Scenario
from src.data.dynamic_episodes import DynamicEpisodeConfig, DynamicEvent
from src.sim.causal_logger import CausalSumoLogger
from src.sim.fixture import fixture_plan
from src.sim.route_builder import RouteBuilder
from src.sim.sumo_exporter import SumoExporter
from src.sim.traci_adaptor import TraciAdapter

logger = logging.getLogger(__name__)


def _event_edges(scenario: Scenario, event: DynamicEvent) -> list[str]:
    """Resolve event parent roads to stable contract edge IDs."""
    affected = set(event.affected_parent_road_ids)
    return [edge.edge_id for edge in scenario.edges if edge.parent_road_id in affected]


class _RuntimeEventTape:
    """Apply causal incident/closure events directly to SUMO lanes."""

    def __init__(self, scenario: Scenario, events: Iterable[DynamicEvent], edge_mapping: dict[str, str]):
        self.edge_mapping = edge_mapping
        self.records = []
        for event in events:
            for edge_id in _event_edges(scenario, event):
                sumo_edge = edge_mapping.get(edge_id, edge_id)
                self.records.append({
                    "event": event,
                    "edge_id": edge_id,
                    "sumo_edge": sumo_edge,
                    "active": False,
                    "announced": False,
                    "restored": False,
                    "lanes": {},
                })
    def active_closed_edges(self) -> set[str]:
        return {
            rec["edge_id"]
            for rec in self.records
            if rec["active"]
            and not rec["restored"]
            and rec["event"].event_type in {"closure", "scheduled_closure"}
        }

    def step(self, traci, sim_time_s: float) -> list[tuple[str, str]]:
        emitted: list[tuple[str, str]] = []
        for rec in self.records:
            event = rec["event"]
            if not rec["announced"] and sim_time_s >= event.reveal_time_s:
                rec["announced"] = True
                emitted.append((event.event_id, "announced"))

            if not rec["active"] and sim_time_s >= event.effect_start_s:
                lane_count = traci.edge.getLaneNumber(rec["sumo_edge"])
                for lane_idx in range(lane_count):
                    lane_id = f"{rec['sumo_edge']}_{lane_idx}"
                    rec["lanes"][lane_id] = {
                        "allowed": list(traci.lane.getAllowed(lane_id)),
                        "disallowed": list(traci.lane.getDisallowed(lane_id)),
                        "max_speed": float(traci.lane.getMaxSpeed(lane_id)),
                    }
                    if event.event_type in {"closure", "scheduled_closure"}:
                        # Existing occupants may clear; new entries are blocked.
                        traci.lane.setAllowed(lane_id, ["authority"])
                    else:
                        # Incident = capacity/speed degradation, not a fabricated label.
                        factor = max(0.10, 1.0 - 0.75 * float(event.severity))
                        traci.lane.setMaxSpeed(lane_id, max(0.1, rec["lanes"][lane_id]["max_speed"] * factor))
                rec["active"] = True
                emitted.append((event.event_id, "closed" if event.event_type in {"closure", "scheduled_closure"} else "incident_active"))

            if rec["active"] and not rec["restored"] and sim_time_s >= event.effect_end_s:
                for lane_id, original in rec["lanes"].items():
                    if event.event_type in {"closure", "scheduled_closure"}:
                        traci.lane.setAllowed(lane_id, original["allowed"])
                        traci.lane.setDisallowed(lane_id, original["disallowed"])
                    else:
                        traci.lane.setMaxSpeed(lane_id, original["max_speed"])
                rec["restored"] = True
                emitted.append((event.event_id, "reopened"))
        return emitted


def run_sumo_causal_episode(
    *,
    scenario: Scenario,
    output_dir: Path,
    config: DynamicEpisodeConfig,
    episode_id: str,
    split: str = "train",
    events: list[DynamicEvent] | None = None,
    traffic_only: bool = False,
) -> Path:
    """Run one full causal episode with the event tape applied inside SUMO."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = SumoExporter(scenario, output_dir=output_dir / "network")
    net_path = exporter.export()
    mapping = exporter.edge_mapping
    route_plan = None if traffic_only else fixture_plan(scenario)

    affected_sumo_edges = []

    for event in events or []:
        for edge_id in _event_edges(scenario, event):
            sumo_edge = mapping.get(edge_id, edge_id)
            if sumo_edge not in affected_sumo_edges:
                affected_sumo_edges.append(sumo_edge)

    blocked_intervals: list[tuple[float, float]] = []

    for event in events or []:
        if event.event_type in {"closure", "scheduled_closure"}:
            blocked_intervals.append(
                (
                    float(event.effect_start_s),
                    float(event.effect_end_s),
                )
            )

    stop_map = {}
    if traffic_only:
        closure_edges = {
            edge_id
            for event in (events or [])
            if event.event_type in {"closure", "scheduled_closure"}
            for edge_id in _event_edges(scenario, event)
        }
        rou_path = _traffic_routes(
            scenario,
            config,
            output_dir / "routes",
            blocked_edge_ids=closure_edges,
        )
    else:
        rou_path = RouteBuilder(
            scenario,
            route_plan,
            {
                "edge_mapping": mapping,
                "node_mapping": exporter.node_mapping,
            },
            background_duration_s=float(config.duration_s),
            background_interval_s=float(config.interval_s),
            background_start_s=0.0,
            background_target_edges=affected_sumo_edges,
            background_blocked_intervals=blocked_intervals,
        ).build(
            output_dir=output_dir / "routes"
        )

        stop_map = {}
        for vehicle in route_plan.vehicle_routes:
            stop_map[vehicle.vehicle_id] = []
            for index, request_id in enumerate(vehicle.customer_order):
                stop_map[vehicle.vehicle_id].append((index, request_id))

    logger_obj = CausalSumoLogger(scenario, interval_s=config.interval_s)
    tape = _RuntimeEventTape(scenario, events or [], mapping)

    runtime_events: list[dict] = []
    teleports = 0
    with TraciAdapter(
        net_path=net_path,
        rou_path=rou_path,
        output_dir=output_dir / "sumo_output",
        edge_ids=[edge.edge_id for edge in scenario.edges],
        stop_to_request_map=stop_map,
        end_time_s=float(config.duration_s),
        step_length_s=1.0,
        gui=False,
        random_seed=config.seed + 211,
        use_subscriptions=traffic_only,
        service_earliest={r.request_id: max(r.earliest_service_start_s, r.release_s) for r in scenario.requests},
    ) as adapter:
        while not adapter.done:
            next_time_s = adapter._traci.simulation.getTime() + 1.0

            for event_id, event_type in tape.step(adapter._traci, next_time_s):
                runtime_events.append({
                    "event_id": event_id,
                    "event_type": event_type,
                    "timestamp_s": float(next_time_s),
                })

            step = adapter.step()

            logger_obj.on_step(
                step,
                known_closed_edges=tape.active_closed_edges(),
            )

            teleports += len(step.teleported_vehicle_ids)

            if step.teleported_vehicle_ids:
                raise RuntimeError(
                    f"SUMO teleportation detected: {step.teleported_vehicle_ids}"
            )

    if teleports:
        raise RuntimeError(f"SUMO episode had {teleports} teleport events")

    logger_obj.write(
        output_dir,
        episode_id=episode_id,
        config=config,
        events=[
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "generation_time_s": e.generation_time_s,
                "reveal_time_s": e.reveal_time_s,
                "effect_start_s": e.effect_start_s,
                "effect_end_s": e.effect_end_s,
                "affected_parent_road_ids": list(e.affected_parent_road_ids),
                "severity": e.severity,
            }
            for e in (events or [])
        ]
    )

    (output_dir / "runtime_events.json").write_text(
        __import__("json").dumps(runtime_events, indent=2), encoding="utf-8"
    )
    return output_dir


def _traffic_routes(scenario, config, output_dir, *, blocked_edge_ids=()):
    """Seeded legal walks, independent of the hidden incident tape."""
    import random
    import xml.etree.ElementTree as ET
    from src.data.causal_episodes import stream_seeds
    from src.data.dynamic_episodes import _daily_profile
    rng = random.Random(stream_seeds(config.seed).trajectory)
    traffic_rng = random.Random(stream_seeds(config.seed).traffic)
    blocked = set(blocked_edge_ids)
    edges = [e for e in scenario.edges if e.open_by_default and e.edge_id not in blocked]
    if not edges:
        raise ValueError("No legal traffic edges remain after applying closures")
    outgoing = {}
    for edge in edges:
        outgoing.setdefault(edge.from_node, []).append(edge)
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="probe_car", sigma="0.3", speedFactor="1", speedDev="0.05")
    departure, index = 0.0, 0
    while departure < config.duration_s - 120:
        edge = rng.choice(edges)
        route = [edge.edge_id]
        for _ in range(5):
            legal = [
                e for e in outgoing.get(edge.to_node, ())
                if e.edge_id not in blocked and e.to_node != edge.from_node
            ]
            if not legal:
                break
            edge = rng.choice(legal)
            route.append(edge.edge_id)
        vehicle = ET.SubElement(root, "vehicle", id=f"traffic_{index}", type="probe_car", depart=f"{departure:.3f}")
        ET.SubElement(vehicle, "route", edges=" ".join(route))
        # Separate RNG for arrival spacing. Congestion arises inside SUMO.
        profile = _daily_profile(departure, config.duration_s, config.regime)
        departure += traffic_rng.uniform(5.0, 10.0) / max(0.5, profile)
        index += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "traffic.rou.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path
