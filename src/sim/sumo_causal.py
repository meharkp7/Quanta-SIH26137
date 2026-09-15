"""SUMO-backed causal episode generator for Step 11.

Runs declared dynamic events inside real SUMO/TraCI and records only measured
simulation outcomes.  The route plan is supplied by the routing pipeline for
arbitrary generated scenarios; the Step-3 fixture planner is never used here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from src.contracts.scenario import Scenario
from src.data.dynamic_episodes import DynamicEpisodeConfig, DynamicEvent
from src.sim.causal_logger import CausalSumoLogger
from src.sim.route_builder import RouteBuilder
from src.sim.sumo_exporter import SumoExporter
from src.sim.traci_adaptor import TraciAdapter

logger = logging.getLogger(__name__)


def _event_edges(scenario: Scenario, event: DynamicEvent) -> list[str]:
    affected = set(event.affected_parent_road_ids)
    return [
        edge.edge_id
        for edge in scenario.edges
        if edge.parent_road_id in affected
    ]


class _RuntimeEventTape:
    """Apply the declared incident/closure tape directly through TraCI."""

    def __init__(
        self,
        scenario: Scenario,
        events: Iterable[DynamicEvent],
        edge_mapping: dict[str, str],
    ) -> None:
        self.edge_mapping = edge_mapping
        self.records: list[dict] = []

        for event in events:
            for edge_id in _event_edges(scenario, event):
                sumo_edge = edge_mapping.get(edge_id, edge_id)
                self.records.append(
                    {
                        "event": event,
                        "edge_id": edge_id,
                        "sumo_edge": sumo_edge,
                        "active": False,
                        "announced": False,
                        "restored": False,
                        "lanes": {},
                    }
                )

    def active_closed_edges(self) -> set[str]:
        return {
            rec["edge_id"]
            for rec in self.records
            if rec["active"]
            and not rec["restored"]
            and rec["event"].event_type
            in {"closure", "scheduled_closure"}
        }

    def step(self, traci, sim_time_s: float) -> list[tuple[str, str]]:
        emitted: list[tuple[str, str]] = []

        for rec in self.records:
            event = rec["event"]

            if (
                not rec["announced"]
                and sim_time_s >= event.reveal_time_s
            ):
                rec["announced"] = True
                emitted.append((event.event_id, "announced"))

            if (
                not rec["active"]
                and sim_time_s >= event.effect_start_s
            ):
                lane_count = traci.edge.getLaneNumber(
                    rec["sumo_edge"]
                )

                for lane_idx in range(lane_count):
                    lane_id = f"{rec['sumo_edge']}_{lane_idx}"
                    rec["lanes"][lane_id] = {
                        "allowed": list(
                            traci.lane.getAllowed(lane_id)
                        ),
                        "disallowed": list(
                            traci.lane.getDisallowed(lane_id)
                        ),
                        "max_speed": float(
                            traci.lane.getMaxSpeed(lane_id)
                        ),
                    }

                    if event.event_type in {
                        "closure",
                        "scheduled_closure",
                    }:
                        traci.lane.setAllowed(
                            lane_id,
                            ["authority"],
                        )
                    else:
                        factor = max(
                            0.10,
                            1.0 - 0.75 * float(event.severity),
                        )
                        traci.lane.setMaxSpeed(
                            lane_id,
                            max(
                                0.1,
                                rec["lanes"][lane_id][
                                    "max_speed"
                                ] * factor,
                            ),
                        )

                rec["active"] = True
                emitted.append(
                    (
                        event.event_id,
                        (
                            "closed"
                            if event.event_type
                            in {"closure", "scheduled_closure"}
                            else "incident_active"
                        ),
                    )
                )

            if (
                rec["active"]
                and not rec["restored"]
                and sim_time_s >= event.effect_end_s
            ):
                for lane_id, original in rec["lanes"].items():
                    if event.event_type in {
                        "closure",
                        "scheduled_closure",
                    }:
                        traci.lane.setAllowed(
                            lane_id,
                            original["allowed"],
                        )
                        traci.lane.setDisallowed(
                            lane_id,
                            original["disallowed"],
                        )
                    else:
                        traci.lane.setMaxSpeed(
                            lane_id,
                            original["max_speed"],
                        )

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
    route_plan=None,
    step_length_s: float = 2.0,
    network_cache_dir: Path | None = None,
) -> Path:
    """Run one full causal episode in SUMO/TraCI."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not traffic_only and route_plan is None:
        raise ValueError(
            "route_plan is required for non-traffic-only causal episodes; "
            "fixture_plan is intentionally not supported"
        )

    if network_cache_dir is not None:
        network_cache_dir = Path(network_cache_dir).resolve()
        cached_net = network_cache_dir / "net.xml"
        if cached_net.exists():
            net_path = cached_net
            mapping = {edge.edge_id: edge.edge_id for edge in scenario.edges}
            node_mapping = {node.node_id: node.node_id for node in scenario.nodes}
        else:
            exporter = SumoExporter(
                scenario,
                output_dir=network_cache_dir,
            )
            net_path = exporter.export()
            mapping = exporter.edge_mapping
            node_mapping = exporter.node_mapping
    else:
        exporter = SumoExporter(
            scenario,
            output_dir=output_dir / "network",
        )
        net_path = exporter.export()
        mapping = exporter.edge_mapping
        node_mapping = exporter.node_mapping

    affected_sumo_edges: list[str] = []
    for event in events or []:
        for edge_id in _event_edges(scenario, event):
            sumo_edge = mapping.get(edge_id, edge_id)
            if sumo_edge not in affected_sumo_edges:
                affected_sumo_edges.append(sumo_edge)

    blocked_intervals: list[tuple[float, float]] = [
        (
            float(event.effect_start_s),
            float(event.effect_end_s),
        )
        for event in events or []
        if event.event_type
        in {"closure", "scheduled_closure"}
    ]

    if traffic_only:
        closure_edges = {
            edge_id
            for event in events or []
            if event.event_type
            in {"closure", "scheduled_closure"}
            for edge_id in _event_edges(scenario, event)
        }

        rou_path = _traffic_routes(
            scenario,
            config,
            output_dir / "routes",
            blocked_edge_ids=closure_edges,
        )
        stop_map = {}
        use_subscriptions = True
    else:
        rou_path = RouteBuilder(
            scenario,
            route_plan,
            {
                "edge_mapping": mapping,
                "node_mapping": node_mapping,
            },
            background_duration_s=float(config.duration_s),
            background_interval_s=float(config.interval_s),
            background_start_s=0.0,
            background_target_edges=affected_sumo_edges,
            background_blocked_intervals=blocked_intervals,
            background_seed=config.seed,
        ).build(
            output_dir=output_dir / "routes"
        )

        stop_map = {
            vehicle.vehicle_id: [
                (index, request_id)
                for index, request_id
                in enumerate(vehicle.customer_ids)
            ]
            for vehicle in route_plan.vehicle_routes
        }
        use_subscriptions = True

    logger_obj = CausalSumoLogger(
        scenario,
        interval_s=config.interval_s,
    )
    tape = _RuntimeEventTape(
        scenario,
        events or [],
        mapping,
    )

    runtime_events: list[dict] = []
    teleports = 0

    with TraciAdapter(
        net_path=net_path,
        rou_path=rou_path,
        output_dir=output_dir / "sumo_output",
        edge_ids=[edge.edge_id for edge in scenario.edges],
        stop_to_request_map=stop_map,
        end_time_s=float(config.duration_s),
        step_length_s=float(step_length_s),
        gui=False,
        random_seed=config.seed + 211,
        use_subscriptions=use_subscriptions,
        service_earliest={
            request.request_id: max(
                request.earliest_service_start_s,
                request.release_s,
            )
            for request in scenario.requests
        },
    ) as adapter:
        while not adapter.done:
            next_time_s = (
                adapter._traci.simulation.getTime() + float(step_length_s)
            )

            for event_id, event_type in tape.step(
                adapter._traci,
                next_time_s,
            ):
                runtime_events.append(
                    {
                        "event_id": event_id,
                        "event_type": event_type,
                        "timestamp_s": float(next_time_s),
                    }
                )

            step = adapter.step()

            logger_obj.on_step(
                step,
                known_closed_edges=tape.active_closed_edges(),
            )

            teleports += len(
                step.teleported_vehicle_ids
            )

            if step.teleported_vehicle_ids:
                raise RuntimeError(
                    "SUMO teleportation detected: "
                    f"{step.teleported_vehicle_ids}"
                )

    if teleports:
        raise RuntimeError(
            f"SUMO episode had {teleports} teleport events"
        )

    logger_obj.write(
        output_dir,
        episode_id=episode_id,
        config=config,
        events=[
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "generation_time_s": event.generation_time_s,
                "reveal_time_s": event.reveal_time_s,
                "effect_start_s": event.effect_start_s,
                "effect_end_s": event.effect_end_s,
                "affected_parent_road_ids": list(
                    event.affected_parent_road_ids
                ),
                "severity": event.severity,
            }
            for event in events or []
        ],
    )

    (output_dir / "runtime_events.json").write_text(
        json.dumps(runtime_events, indent=2),
        encoding="utf-8",
    )

    return output_dir


def _traffic_routes(
    scenario,
    config,
    output_dir,
    *,
    blocked_edge_ids=(),
):
    """Generate legal SUMO background traffic from the scenario graph."""

    import random
    import xml.etree.ElementTree as ET

    from src.data.causal_episodes import stream_seeds
    from src.data.dynamic_episodes import _daily_profile

    rng = random.Random(
        stream_seeds(config.seed).trajectory
    )
    traffic_rng = random.Random(
        stream_seeds(config.seed).traffic
    )

    blocked = set(blocked_edge_ids)
    edges = [
        edge
        for edge in scenario.edges
        if edge.open_by_default
        and edge.edge_id not in blocked
    ]

    if not edges:
        raise ValueError(
            "No legal traffic edges remain after applying closures"
        )

    outgoing: dict[str, list] = {}
    for edge in edges:
        outgoing.setdefault(edge.from_node, []).append(edge)

    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="probe_car",
        sigma="0.0",
        tau="2.0",
        speedFactor="1",
        speedDev="0.05",
    )

    departure = 0.0
    index = 0

    while departure < config.duration_s - 120:
        edge = rng.choice(edges)
        route = [edge.edge_id]

        for _ in range(5):
            legal = [
                candidate
                for candidate in outgoing.get(
                    edge.to_node,
                    (),
                )
                if (
                    candidate.edge_id not in blocked
                    and candidate.to_node != edge.from_node
                )
            ]

            if not legal:
                break

            edge = rng.choice(legal)
            route.append(edge.edge_id)

        vehicle = ET.SubElement(
            root,
            "vehicle",
            id=f"traffic_{index}",
            type="probe_car",
            depart=f"{departure:.3f}",
        )
        ET.SubElement(
            vehicle,
            "route",
            edges=" ".join(route),
        )

        profile = _daily_profile(
            departure,
            config.duration_s,
            config.regime,
        )

        departure += (
            traffic_rng.uniform(5.0, 10.0)
            / max(0.5, profile)
        )
        index += 1

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / "traffic.rou.xml"
    ET.ElementTree(root).write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )
    return path