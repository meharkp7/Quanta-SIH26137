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
        *,
        background_duration_s: float = 0.0,
        background_interval_s: float = 60.0,
        background_start_s: float = 0.0,
        background_target_edges: list[str] | None = None,
        background_blocked_intervals: list[tuple[float, float]] | None = None,
    ) -> None:
        self.scenario = scenario
        self.route_plan = route_plan
        self._edge_map: dict[str, str] = sumo_mapping.get("edge_mapping", {})

        if background_duration_s < 0:
            raise ValueError("background_duration_s must be non-negative")

        if background_interval_s <= 0:
            raise ValueError("background_interval_s must be positive")

        if background_start_s < 0:
            raise ValueError("background_start_s must be non-negative")

        self._background_duration_s = float(background_duration_s)
        self._background_interval_s = float(background_interval_s)
        self._background_start_s = float(background_start_s)
        self._background_target_edges = list(background_target_edges or [])
        self._background_blocked_intervals = list(
            background_blocked_intervals or []
        )

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
        Emit background traffic only when explicit target edges have been
        supplied by the caller.

        Ordinary RouteBuilder/SUMO runs do not need background traffic.
        Causal SUMO episodes opt in by providing background_target_edges.
        """
        if not self._background_target_edges:
            return

        target_set = set(self._background_target_edges)

        # Current Step-3 fixture causal corridor.
        #
        # E01 -> E16 -> E62
        #
        # Both E16 and E62 are on the valid event-prone path. Only create
        # this route when the required physical edges are present.
        available = set(self._edge_map.values())

        event_route = ["E01", "E12", "E23", "E34", "E45", "E51", "E16", "E62"]

        if not all(edge in available for edge in event_route):
            logger.warning(
                "Cannot generate causal background route: "
                "required event corridor is not present in SUMO mapping"
            )
            return

        if not target_set.intersection({"E16", "E62"}):
            return

        route_id = "bg_event_route"

        ET.SubElement(
            parent,
            "route",
            attrib={
                "id": route_id,
                "edges": " ".join(event_route),
            },
        )

        if self._background_duration_s <= self._background_start_s:
            candidate_departures = [self._background_start_s]
        else:
            candidate_departures = []
            departure = self._background_start_s

            while departure < self._background_duration_s:
                candidate_departures.append(departure)
                departure += self._background_interval_s

        # Do not inject vehicles onto an event-affected route while one of
        # its edges is explicitly closed.
        #
        # Vehicles may depart before the closure and continue into it.
        # Vehicles may also resume after the edge is reopened.
        departures = [
            departure
            for departure in candidate_departures
            if not any(
                start <= departure < end
                for start, end in self._background_blocked_intervals
            )
        ]

        for index, departure in enumerate(departures):
            ET.SubElement(
                parent,
                "vehicle",
                attrib={
                    "id": f"bg_{index:05d}",
                    "type": "background_car",
                    "route": route_id,
                    "depart": f"{departure:.1f}",
                },
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    with open(path, "wb") as fh:
        tree.write(fh, xml_declaration=True, encoding="utf-8")
