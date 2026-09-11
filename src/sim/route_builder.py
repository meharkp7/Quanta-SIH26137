"""
route_builder.py — Sub-task 3 of Step 6
=========================================
Converts a RoutePlan contract into SUMO route/trip XML files (.rou.xml).

Design
------
Each delivery vehicle is emitted as a SUMO <vehicle> with:
  - An embedded <route> that lists the full physical edge sequence.
  - One <stop> per served customer, placed on the **last edge of the
    StopLeg that arrives at that customer's access node**.

Using the leg's `to_stop_id` → last edge relationship is the correct
generalizable approach because:
  * Each StopLeg already encodes exactly which edges connect two stops.
  * The last edge of leg L is guaranteed to lead into the destination
    node of leg L — no graph scan is needed.
  * This scales naturally to any network size and any route structure.

Background filler trips are also emitted using direct <trip> elements
so that SUMO routes them automatically.

Owner: P1  |  Step 6, Days 1-3
"""

from __future__ import annotations

import logging
from pathlib import Path
from xml.etree import ElementTree as ET

from src.contracts.routing import RoutePlan, StopLeg, VehicleRoute
from src.contracts.scenario import Request, Scenario

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RouteBuilder:
    """
    Builds SUMO route/vehicle XML from a RoutePlan + Scenario.

    Usage::

        builder = RouteBuilder(scenario, route_plan, sumo_mapping)
        rou_path = builder.build(output_dir=Path("artifacts/sumo/step6"))

    ``sumo_mapping`` is the dict from ``sumo_mapping.json`` (produced by
    SumoExporter).  Only ``edge_mapping`` is used here.
    """

    def __init__(
        self,
        scenario: Scenario,
        route_plan: RoutePlan,
        sumo_mapping: dict,
    ) -> None:
        self.scenario = scenario
        self.route_plan = route_plan
        self._edge_map: dict[str, str] = sumo_mapping.get("edge_mapping", {})

        # Quick look-ups
        self._requests: dict[str, Request] = {
            r.request_id: r for r in scenario.requests
        }
        # Map stop_id (node_id or request_id) → request for the leg lookup
        # A "stop" in the leg is identified by its to_stop_id; for customer
        # stops, to_stop_id is the access_node_id.
        self._node_to_request: dict[str, list[Request]] = {}
        for req in scenario.requests:
            self._node_to_request.setdefault(req.access_node_id, []).append(req)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(self, output_dir: Path) -> Path:
        """Write vehicles.rou.xml and return its path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        root = ET.Element("routes")
        self._add_vtypes(root)

        for veh_route in self.route_plan.vehicle_routes:
            self._add_delivery_vehicle(root, veh_route)

        self._add_background_trips(root)

        path = output_dir / "vehicles.rou.xml"
        _write_xml(root, path)
        return path

    # ------------------------------------------------------------------
    # vType definitions
    # ------------------------------------------------------------------

    def _add_vtypes(self, root: ET.Element) -> None:
        ET.SubElement(root, "vType", attrib={
            "id": "delivery_van",
            "accel": "2.0",
            "decel": "4.5",
            "length": "5.0",
            "maxSpeed": "20.0",
            "sigma": "0.0",           # deterministic
            "color": "1,0.5,0",
        })
        ET.SubElement(root, "vType", attrib={
            "id": "background_car",
            "accel": "2.6",
            "decel": "4.5",
            "length": "4.5",
            "maxSpeed": "13.89",
            "sigma": "0.5",
            "color": "0.6,0.6,0.6",
        })

    # ------------------------------------------------------------------
    # Per-vehicle route builder
    # ------------------------------------------------------------------

    def _add_delivery_vehicle(
        self,
        parent: ET.Element,
        veh_route: VehicleRoute,
    ) -> None:
        """
        Emit one SUMO <vehicle> with an embedded <route> and one <stop>
        per served customer.

        Stop placement strategy
        -----------------------
        For each leg in `veh_route.legs`:
          - Translate its `physical_edge_ids` to SUMO edge IDs.
          - If the leg's `to_stop_id` matches a customer's `access_node_id`,
            place a <stop> on the LAST edge of that leg.

        This is correct because:
          * physical_edge_ids[last] always leads into to_stop_id by definition.
          * SUMO requires stops to be in route order; building them leg-by-leg
            naturally preserves this order.
          * The approach is independent of network topology — no graph scan.
        """
        vehicle_id = veh_route.vehicle_id

        if not veh_route.legs:
            logger.debug("Vehicle %r has no legs — skipping", vehicle_id)
            return

        # Build ordered stop list from legs + deduplicated full edge list
        # We process legs in order so that stops are also in order.
        all_sumo_edges: list[str] = []
        stop_specs: list[dict] = []   # {edge, duration, tripId}

        # Map request_id -> request for the customer_order lookup
        # Build a customer-order index so we can match legs to requests.
        order_index: dict[str, int] = {
            rid: i for i, rid in enumerate(veh_route.customer_order)
        }

        for leg in veh_route.legs:
            # Translate edges
            leg_sumo_edges = [self._edge_map.get(e, e) for e in leg.physical_edge_ids]

            # Deduplicate consecutive edges across legs
            for sumo_eid in leg_sumo_edges:
                if not all_sumo_edges or all_sumo_edges[-1] != sumo_eid:
                    all_sumo_edges.append(sumo_eid)

            # Check if this leg ends at a customer access node
            if not leg_sumo_edges:
                continue

            stop_edge = leg_sumo_edges[-1]   # last edge of leg → leads to to_stop_id

            # Find requests served at this leg's destination node
            # (to_stop_id is a node ID for customer stops, depot ID for depot)
            served_here = self._requests_at_stop(leg.to_stop_id, veh_route)
            for request_id, req in served_here:
                stop_specs.append({
                    "edge": stop_edge,
                    "duration": f"{req.service_duration_s:.1f}",
                    "tripId": request_id,
                    "until": str(max(req.earliest_service_start_s, req.release_s) + req.service_duration_s),
                })

        if not all_sumo_edges:
            logger.warning("Vehicle %r: no SUMO edges after translation", vehicle_id)
            return

        vehicle_el = ET.SubElement(parent, "vehicle", attrib={
            "id": vehicle_id,
            "type": "delivery_van",
            "depart": f"{veh_route.expected_departure_s:.1f}",
            "color": "1,0,0",
        })

        ET.SubElement(vehicle_el, "route", attrib={
            "edges": " ".join(all_sumo_edges),
        })

        for spec in stop_specs:
            ET.SubElement(vehicle_el, "stop", attrib={
                "edge": spec["edge"],
                "endPos": "-1",       # SUMO: stop at end of the edge lane
                "duration": spec["duration"],
                "parking": "true",
                "tripId": spec["tripId"],
                "until": spec["until"],
            })

    def _requests_at_stop(
        self,
        stop_node_id: str,
        veh_route: VehicleRoute,
    ) -> list[tuple[str, Request]]:
        """
        Return (request_id, Request) pairs for customers whose access_node_id
        equals stop_node_id AND who appear in this vehicle's customer_order.

        Preserving customer_order order ensures stops are emitted in the order
        the vehicle serves them.
        """
        candidates = self._node_to_request.get(stop_node_id, [])
        order_set = set(veh_route.customer_order)
        result = [
            (req.request_id, req)
            for req in candidates
            if req.request_id in order_set
        ]
        # Sort by position in customer_order for determinism
        order_index = {rid: i for i, rid in enumerate(veh_route.customer_order)}
        result.sort(key=lambda x: order_index.get(x[0], 9999))
        return result

    # ------------------------------------------------------------------
    # Background trips
    # ------------------------------------------------------------------

    def _add_background_trips(self, parent: ET.Element) -> None:
        """
        Emit lightweight background car trips to create realistic traffic
        context.  Only edges that exist in the edge_mapping are used.

        For portability across different scenarios, background trips are
        generated from the scenario's own edges rather than hard-coded IDs.
        We pick a subset of edges that form a valid corridor.
        """
        # Use the first and last edge from the scenario as a simple corridor.
        # This is scenario-agnostic and will always reference valid edges.
        edge_ids = list(self._edge_map.values())
        if len(edge_ids) < 2:
            return

        background_trips = [
            ("bg_0", edge_ids[0],  edge_ids[min(4, len(edge_ids)-1)],   5.0),
            ("bg_1", edge_ids[1],  edge_ids[min(5, len(edge_ids)-1)],  12.0),
        ]
        for trip_id, from_edge, to_edge, depart in background_trips:
            if from_edge == to_edge:
                continue
            ET.SubElement(parent, "trip", attrib={
                "id": trip_id,
                "type": "background_car",
                "from": from_edge,
                "to": to_edge,
                "depart": f"{depart:.1f}",
            })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(path, "wb") as fh:
        tree.write(fh, xml_declaration=True, encoding="utf-8")
