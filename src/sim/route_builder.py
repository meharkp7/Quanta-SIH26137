"""
route_builder.py — Step 6 SUMO route XML builder.

Supports both repository route representations:
1. logical routes from ``src.routing.route_plan`` (customer IDs only);
2. physical contract routes from ``src.contracts.routing`` (StopLegs).

The routing layer intentionally owns logical plans while the simulation layer
needs physical edge sequences.  When a logical RoutePlan is supplied, this
builder resolves it through the repository RouteEvaluator/DirectedPathBuilder
instead of assuming that logical VehicleRoute has ``legs``.
"""

from __future__ import annotations

import heapq
import logging
from math import inf
from pathlib import Path
from xml.etree import ElementTree as ET

from src.contracts.routing import (
    RoutePlan as PhysicalRoutePlanContract,
    VehicleRoute as PhysicalVehicleRoute,
)
from src.contracts.scenario import Request, Scenario
from src.routing.route_evaluator import RouteEvaluator
from src.routing.route_plan import (
    RoutePlan as LogicalRoutePlan,
    VehicleRoute as LogicalVehicleRoute,
)

logger = logging.getLogger(__name__)


class RouteBuilder:
    """Build SUMO vehicle/background route XML from a Quanta route plan."""

    def __init__(
        self,
        scenario: Scenario,
        route_plan,
        sumo_mapping: dict,
        *,
        background_duration_s: float = 0.0,
        background_interval_s: float = 60.0,
        background_start_s: float = 0.0,
        background_target_edges: list[str] | None = None,
        background_blocked_intervals: list[tuple[float, float]] | None = None,
    ) -> None:
        self.scenario = scenario
        self.route_plan = route_plan
        self._edge_map: dict[str, str] = sumo_mapping.get(
            "edge_mapping",
            {},
        )

        if background_duration_s < 0:
            raise ValueError(
                "background_duration_s must be non-negative"
            )
        if background_interval_s <= 0:
            raise ValueError(
                "background_interval_s must be positive"
            )
        if background_start_s < 0:
            raise ValueError(
                "background_start_s must be non-negative"
            )

        self._background_duration_s = float(
            background_duration_s
        )
        self._background_interval_s = float(
            background_interval_s
        )
        self._background_start_s = float(
            background_start_s
        )
        self._background_target_edges = list(
            background_target_edges or []
        )
        self._background_blocked_intervals = tuple(
            (float(start), float(end))
            for start, end in (background_blocked_intervals or [])
            if float(end) > float(start)
        )

        self._requests: dict[str, Request] = {
            r.request_id: r
            for r in scenario.requests
        }

        self._node_to_request: dict[
            str, list[Request]
        ] = {}

        for req in scenario.requests:
            self._node_to_request.setdefault(
                req.access_node_id,
                [],
            ).append(req)

        # Logical Step-4/5 plans contain customer_ids only.  Resolve those
        # once into the existing physical representation used by SUMO.
        self._logical_evaluations = None

    def build(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        root = ET.Element("routes")
        self._add_vtypes(root)

        if self._is_logical_plan():
            self._build_logical_routes(root)
        else:
            for veh_route in self.route_plan.vehicle_routes:
                self._add_contract_delivery_vehicle(
                    root,
                    veh_route,
                )

        self._add_background_trips(root)

        path = output_dir / "vehicles.rou.xml"
        _write_xml(root, path)
        return path

    def _is_logical_plan(self) -> bool:
        routes = getattr(
            self.route_plan,
            "vehicle_routes",
            (),
        )
        if not routes:
            return False

        first = routes[0]
        return isinstance(
            first,
            LogicalVehicleRoute,
        ) or hasattr(first, "customer_ids")

    def _build_logical_routes(
        self,
        parent: ET.Element,
    ) -> None:
        """Resolve logical customer sequences into physical road paths."""

        evaluator = RouteEvaluator(self.scenario)
        evaluation = evaluator.evaluate(
            self.route_plan,
            planning_time_s=0.0,
        )

        if not evaluation.feasible:
            raise ValueError(
                "Cannot build SUMO routes from an infeasible logical "
                f"RoutePlan: {evaluation.errors}"
            )

        self._logical_evaluations = {
            item.vehicle_id: item
            for item in evaluation.vehicle_evaluations
        }

        for logical_route in self.route_plan.vehicle_routes:
            vehicle_eval = self._logical_evaluations.get(
                logical_route.vehicle_id
            )
            if vehicle_eval is None:
                raise ValueError(
                    "No evaluator result for vehicle "
                    f"{logical_route.vehicle_id!r}"
                )

            physical_route = vehicle_eval.physical_route
            if physical_route is None:
                if logical_route.customer_ids:
                    raise ValueError(
                        "Logical vehicle route has customers but no "
                        f"physical route: {logical_route.vehicle_id!r}"
                    )
                continue

            # RouteEvaluator produces physically connected directed legs, but
            # its generic graph may choose an immediate U-turn.  SUMO's
            # compiled network intentionally forbids U-turn connections, so
            # repair only the affected legs here.
            edge_lookup = {
                edge.edge_id: edge
                for edge in self.scenario.edges
            }
            all_sumo_edges: list[str] = []
            stop_specs = []
            previous_physical_edge = None

            physical_legs = list(physical_route.legs)
            stop_evaluations = list(vehicle_eval.stops)
            previous_stop_until = 0.0

            for leg_index, leg in enumerate(physical_legs):
                leg_edges = list(leg.edge_ids)

                needs_repair = self._contains_u_turn(
                    leg_edges,
                    edge_lookup,
                    previous_edge_id=previous_physical_edge,
                )

                if needs_repair:
                    leg_edges = self._path_without_u_turns(
                        leg.from_node,
                        leg.to_node,
                        previous_edge_id=previous_physical_edge,
                    )

                translated = self._translate_edges(leg_edges)
                for edge in translated:
                    if not all_sumo_edges or all_sumo_edges[-1] != edge:
                        all_sumo_edges.append(edge)

                if leg_edges:
                    previous_physical_edge = leg_edges[-1]

                # The first len(stops) physical legs correspond to customer
                # stops; the final leg is the depot return.
                if leg_index < len(stop_evaluations):
                    stop = stop_evaluations[leg_index]
                    if not leg_edges:
                        raise ValueError(
                            "Customer stop resolved without physical edges: "
                            f"vehicle={logical_route.vehicle_id!r}, "
                            f"customer={stop.customer_id!r}"
                        )

                    request = self._requests.get(stop.customer_id)
                    if request is None:
                        raise ValueError(
                            "Evaluator returned an unknown customer: "
                            f"{stop.customer_id!r}"
                        )

                    requested_until = (
                        max(
                            request.earliest_service_start_s,
                            request.release_s,
                        )
                        + request.service_duration_s
                    )
                    stop_until = max(
                        previous_stop_until,
                        requested_until,
                    )
                    previous_stop_until = stop_until
                    stop_specs.append(
                        {
                            "edge": self._edge_map.get(
                                leg_edges[-1],
                                leg_edges[-1],
                            ),
                            "duration": f"{request.service_duration_s:.1f}",
                            "tripId": request.request_id,
                            "until": f"{stop_until:.2f}",
                        }
                    )

            if not all_sumo_edges:
                if logical_route.customer_ids:
                    raise ValueError(
                        "Logical vehicle route has customers but resolved "
                        f"to zero physical edges: {logical_route.vehicle_id!r}"
                    )
                continue

            vehicle_el = ET.SubElement(
                parent,
                "vehicle",
                attrib={
                    "id": logical_route.vehicle_id,
                    "type": "delivery_van",
                    "depart": "0.0",
                    "color": "1,0,0",
                },
            )

            ET.SubElement(
                vehicle_el,
                "route",
                attrib={
                    "edges": " ".join(
                        all_sumo_edges
                    ),
                },
            )

            for spec in stop_specs:
                ET.SubElement(
                    vehicle_el,
                    "stop",
                    attrib={
                        "edge": spec["edge"],
                        "endPos": "-1",
                        "duration": spec["duration"],
                        "parking": "true",
                        "tripId": spec["tripId"],
                        "until": spec["until"],
                    },
                )

    def _path_without_u_turns(
        self,
        start_node: str,
        end_node: str,
        *,
        previous_edge_id: str | None = None,
    ) -> list[str]:
        """Find a shortest free-flow path with no immediate U-turns.

        SUMO's Step-6 network deliberately contains no U-turn connections,
        while the generic routing graph may select an immediate reverse edge.
        This adapter therefore resolves only the affected physical leg into
        a SUMO-compatible directed path instead of weakening the network
        connection rules.
        """
        if start_node == end_node:
            return []

        edges = {e.edge_id: e for e in self.scenario.edges if e.open_by_default}
        outgoing: dict[str, list] = {}
        for edge in edges.values():
            outgoing.setdefault(edge.from_node, []).append(edge)
        for values in outgoing.values():
            values.sort(key=lambda e: str(e.edge_id))

        initial_prev_from = None
        if previous_edge_id is not None and previous_edge_id in edges:
            initial_prev_from = edges[previous_edge_id].from_node

        # State includes the previous edge's origin node, which is exactly
        # what is needed to forbid an immediate A->B->A reversal.
        start_state = (start_node, initial_prev_from)
        distances = {start_state: 0.0}
        previous: dict[tuple[str, str | None], tuple[tuple[str, str | None], str]] = {}
        queue = [(0.0, str(start_node), "", start_node, initial_prev_from)]
        goal_state = None

        while queue:
            cost, _, _, node, prev_from = heapq.heappop(queue)
            state = (node, prev_from)
            if cost > distances.get(state, inf):
                continue
            if node == end_node:
                goal_state = state
                break

            for edge in outgoing.get(node, ()):
                if prev_from is not None and edge.to_node == prev_from:
                    continue

                next_state = (edge.to_node, edge.from_node)
                next_cost = cost + edge.free_flow_time_s
                if next_cost < distances.get(next_state, inf):
                    distances[next_state] = next_cost
                    previous[next_state] = (state, edge.edge_id)
                    heapq.heappush(
                        queue,
                        (next_cost, str(edge.to_node), str(edge.edge_id), edge.to_node, edge.from_node),
                    )

        if goal_state is None:
            raise ValueError(
                f"No SUMO-compatible no-U-turn path exists from "
                f"{start_node!r} to {end_node!r}"
            )

        result: list[str] = []
        state = goal_state
        while state != start_state:
            prior, edge_id = previous[state]
            result.append(edge_id)
            state = prior
        result.reverse()
        return result

    @staticmethod
    def _contains_u_turn(
        edge_ids,
        edge_lookup,
        *,
        previous_edge_id=None,
    ) -> bool:
        previous = (
            edge_lookup.get(previous_edge_id)
            if previous_edge_id is not None
            else None
        )
        for edge_id in edge_ids:
            edge = edge_lookup[edge_id]
            if previous is not None and edge.to_node == previous.from_node:
                return True
            previous = edge
        return False

    def _translate_edges(
        self,
        edge_ids,
    ) -> list[str]:
        result = []
        for edge_id in edge_ids:
            sumo_edge = self._edge_map.get(
                edge_id,
                edge_id,
            )
            if not result or result[-1] != sumo_edge:
                result.append(sumo_edge)
        return result

    def _add_vtypes(
        self,
        root: ET.Element,
    ) -> None:
        ET.SubElement(
            root,
            "vType",
            attrib={
                "id": "delivery_van",
                "accel": "1.5",
                "decel": "4.5",
                "length": "5.0",
                "minGap": "2.5",
                "tau": "2.0",
                "maxSpeed": "20.0",
                "sigma": "0.0",
                "color": "1,0.5,0",
            },
        )
        ET.SubElement(
            root,
            "vType",
            attrib={
                "id": "background_car",
                "accel": "1.2",
                "decel": "4.5",
                "length": "4.5",
                "minGap": "2.5",
                "tau": "2.0",
                "maxSpeed": "13.89",
                "sigma": "0.0",
                "color": "0.6,0.6,0.6",
            },
        )

    def _add_contract_delivery_vehicle(
        self,
        parent: ET.Element,
        veh_route: PhysicalVehicleRoute,
    ) -> None:
        """Emit a physical contract VehicleRoute unchanged."""

        if not veh_route.legs:
            logger.debug(
                "Vehicle %r has no legs — skipping",
                veh_route.vehicle_id,
            )
            return

        all_sumo_edges: list[str] = []
        stop_specs: list[dict] = []
        previous_stop_until = 0.0

        for leg in veh_route.legs:
            leg_sumo_edges = self._translate_edges(
                leg.physical_edge_ids
            )

            for edge in leg_sumo_edges:
                if (
                    not all_sumo_edges
                    or all_sumo_edges[-1] != edge
                ):
                    all_sumo_edges.append(edge)

            if not leg_sumo_edges:
                continue

            stop_edge = leg_sumo_edges[-1]

            for request_id, request in (
                self._requests_at_stop(
                    leg.to_stop_id,
                    veh_route,
                )
            ):
                requested_until = (
                    max(
                        request.earliest_service_start_s,
                        request.release_s,
                    )
                    + request.service_duration_s
                )
                stop_until = max(
                    previous_stop_until,
                    requested_until,
                )
                previous_stop_until = stop_until
                stop_specs.append(
                    {
                        "edge": stop_edge,
                        "duration": (
                            f"{request.service_duration_s:.1f}"
                        ),
                        "tripId": request_id,
                        "until": f"{stop_until:.2f}",
                    }
                )

        if not all_sumo_edges:
            logger.warning(
                "Vehicle %r: no SUMO edges after translation",
                veh_route.vehicle_id,
            )
            return

        vehicle_el = ET.SubElement(
            parent,
            "vehicle",
            attrib={
                "id": veh_route.vehicle_id,
                "type": "delivery_van",
                "depart": (
                    f"{max(0.0, float(veh_route.expected_departure_s)):.1f}"
                ),
                "departLane": "best",
                "departSpeed": "max",
                "color": "1,0,0",
            },
        )

        ET.SubElement(
            vehicle_el,
            "route",
            attrib={
                "edges": " ".join(
                    all_sumo_edges
                ),
            },
        )

        for spec in stop_specs:
            ET.SubElement(
                vehicle_el,
                "stop",
                attrib={
                    "edge": spec["edge"],
                    "endPos": "-1",
                    "duration": spec["duration"],
                    "parking": "true",
                    "tripId": spec["tripId"],
                    "until": spec["until"],
                },
            )

    def _requests_at_stop(
        self,
        stop_node_id: str,
        veh_route: PhysicalVehicleRoute,
    ) -> list[tuple[str, Request]]:
        candidates = self._node_to_request.get(
            stop_node_id,
            [],
        )

        order = tuple(
            veh_route.customer_order
        )
        order_index = {
            rid: i
            for i, rid in enumerate(order)
        }
        order_set = set(order)

        result = [
            (
                req.request_id,
                req,
            )
            for req in candidates
            if req.request_id in order_set
        ]

        result.sort(
            key=lambda item: order_index.get(
                item[0],
                999999,
            )
        )
        return result

    def _add_background_trips(self, parent: ET.Element) -> None:
        """Emit deterministic background trips on safe road-edge pairs."""
        edge_ids = list(dict.fromkeys(self._edge_map.values()))
        if len(edge_ids) < 2:
            logger.warning("Cannot generate background traffic: fewer than two mapped edges")
            return
        affected = set(self._background_target_edges)
        safe_edges = [edge for edge in edge_ids if edge not in affected]
        preferred = [
            (edge_ids[0], edge_ids[min(4, len(edge_ids) - 1)]),
            (edge_ids[1], edge_ids[min(5, len(edge_ids) - 1)]),
        ]
        corridors = [
            pair for pair in preferred
            if pair[0] != pair[1] and pair[0] not in affected and pair[1] not in affected
        ]
        if not corridors and len(safe_edges) >= 2:
            corridors = [(safe_edges[0], safe_edges[1])]
            if len(safe_edges) >= 4:
                corridors.append((safe_edges[2], safe_edges[3]))
        if not corridors:
            logger.warning("Cannot generate background traffic: no safe edge pair")
            return
        if self._background_duration_s <= self._background_start_s:
            departures = [self._background_start_s]
        else:
            departures = []
            departure = self._background_start_s
            while departure < self._background_duration_s:
                departures.append(departure)
                departure += self._background_interval_s
        trip_index = 0
        for departure in departures:
            for from_edge, to_edge in corridors:
                if any(
                    start <= departure < end and (from_edge in affected or to_edge in affected)
                    for start, end in self._background_blocked_intervals
                ):
                    continue
                ET.SubElement(parent, "trip", attrib={
                    "id": f"bg_{trip_index}",
                    "type": "background_car",
                    "from": from_edge,
                    "to": to_edge,
                    "depart": f"{departure:.1f}",
                    "departLane": "best",
                    "departSpeed": "max",
                })
                trip_index += 1


def _write_xml(
    root: ET.Element,
    path: Path,
) -> None:
    tree = ET.ElementTree(root)
    ET.indent(
        tree,
        space="  ",
    )
    with open(
        path,
        "wb",
    ) as fh:
        tree.write(
            fh,
            xml_declaration=True,
            encoding="utf-8",
        )
