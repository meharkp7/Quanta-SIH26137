"""
route_builder.py — Sub-task 3 of Step 6
=========================================
Converts a RoutePlan contract into SUMO route/trip XML files (.rou.xml).

Design
------
Each delivery vehicle is emitted as a SUMO <vehicle> with:
  - An embedded <route> that lists the full physical edge sequence.
  - One <stop> per served customer, placed on the last edge of the
    StopLeg that arrives at that customer's access node.

Background filler trips are emitted as SUMO <trip> elements.

Background traffic supports two modes:

1. Normal / Step-6 mode:
   - Uses the original deterministic fixture-compatible corridors.
   - Produces background vehicles such as bg_00000, bg_00001, ...

2. Causal / Step-11 mode:
   - Caller supplies background_target_edges.
   - Additional traffic is routed through affected edges.
   - Caller may supply background_blocked_intervals so departures are not
     introduced during scheduled closure intervals.

The default constructor behavior remains compatible with Step 6.
Passing background_target_edges=[] explicitly disables background traffic.

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

    Parameters
    ----------
    scenario:
        Scenario containing vehicles, requests and network edges.

    route_plan:
        RoutePlan whose delivery vehicles should be emitted.

    sumo_mapping:
        Mapping produced by SumoExporter.

    background_duration_s:
        End of the background traffic generation window.

        Default is 0.0 for backwards compatibility with Step 6.
        Causal Step-11 generation supplies the episode duration.

    background_interval_s:
        Time between background departures.

    background_start_s:
        First background departure time.

    background_target_edges:
        Optional explicit set of edges that background traffic should
        traverse.

        None:
            Use the normal Step-6 background corridors.

        []:
            Explicitly disable background traffic.

        non-empty list:
            Use the normal corridor plus deterministic corridors to
            the supplied target edges.

    background_blocked_intervals:
        Optional closure intervals represented as
        (start_time_s, end_time_s).

        Background vehicles are not spawned when their departure time falls
        inside one of these intervals.
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

        # IMPORTANT:
        # None means "use the normal Step-6 background behavior".
        # [] means "caller explicitly disabled background traffic".
        self._background_target_edges_explicit = (
            background_target_edges is not None
        )
        self._background_target_edges = list(
            background_target_edges or []
        )

        self._background_blocked_intervals = [
            (
                float(start),
                float(end),
            )
            for start, end in (
                background_blocked_intervals or []
            )
        ]

        # Validate blocked intervals.
        for start, end in self._background_blocked_intervals:
            if start < 0:
                raise ValueError(
                    "background blocked interval start must be non-negative"
                )
            if end < start:
                raise ValueError(
                    "background blocked interval end must be >= start"
                )

        # Quick look-ups.
        self._requests: dict[str, Request] = {
            r.request_id: r
            for r in scenario.requests
        }

        # Map access node -> requests.
        self._node_to_request: dict[
            str,
            list[Request],
        ] = {}

        for req in scenario.requests:
            self._node_to_request.setdefault(
                req.access_node_id,
                [],
            ).append(req)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(self, output_dir: Path) -> Path:
        """Write vehicles.rou.xml and return its path."""

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        root = ET.Element("routes")

        self._add_vtypes(root)

        for veh_route in self.route_plan.vehicle_routes:
            self._add_delivery_vehicle(
                root,
                veh_route,
            )

        self._add_background_trips(root)

        path = output_dir / "vehicles.rou.xml"

        _write_xml(
            root,
            path,
        )

        return path

    # ------------------------------------------------------------------
    # vType definitions
    # ------------------------------------------------------------------

    def _add_vtypes(
        self,
        root: ET.Element,
    ) -> None:
        ET.SubElement(
            root,
            "vType",
            attrib={
                "id": "delivery_van",
                "accel": "2.0",
                "decel": "4.5",
                "length": "5.0",
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
                "accel": "2.6",
                "decel": "4.5",
                "length": "4.5",
                "maxSpeed": "13.89",
                "sigma": "0.5",
                "color": "0.6,0.6,0.6",
            },
        )

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

        Stop placement:
            The stop is placed on the last physical edge of the leg
            arriving at the customer's access node.
        """

        vehicle_id = veh_route.vehicle_id

        if not veh_route.legs:
            logger.debug(
                "Vehicle %r has no legs — skipping",
                vehicle_id,
            )
            return

        all_sumo_edges: list[str] = []
        stop_specs: list[dict] = []

        for leg in veh_route.legs:
            leg_sumo_edges = [
                self._edge_map.get(
                    edge_id,
                    edge_id,
                )
                for edge_id in leg.physical_edge_ids
            ]

            # Deduplicate consecutive edges across legs.
            for sumo_edge_id in leg_sumo_edges:
                if (
                    not all_sumo_edges
                    or all_sumo_edges[-1] != sumo_edge_id
                ):
                    all_sumo_edges.append(
                        sumo_edge_id
                    )

            if not leg_sumo_edges:
                continue

            # Last edge leads to leg.to_stop_id.
            stop_edge = leg_sumo_edges[-1]

            served_here = self._requests_at_stop(
                leg.to_stop_id,
                veh_route,
            )

            for request_id, req in served_here:
                stop_specs.append(
                    {
                        "edge": stop_edge,
                        "duration": (
                            f"{req.service_duration_s:.1f}"
                        ),
                        "tripId": request_id,
                        "until": str(
                            max(
                                req.earliest_service_start_s,
                                req.release_s,
                            )
                            + req.service_duration_s
                        ),
                    }
                )

        if not all_sumo_edges:
            logger.warning(
                "Vehicle %r: no SUMO edges after translation",
                vehicle_id,
            )
            return

        vehicle_el = ET.SubElement(
            parent,
            "vehicle",
            attrib={
                "id": vehicle_id,
                "type": "delivery_van",
                "depart": (
                    f"{veh_route.expected_departure_s:.1f}"
                ),
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

    # ------------------------------------------------------------------
    # Stop/request mapping
    # ------------------------------------------------------------------

    def _requests_at_stop(
        self,
        stop_node_id: str,
        veh_route: VehicleRoute,
    ) -> list[tuple[str, Request]]:
        """
        Return (request_id, Request) pairs for customers whose
        access_node_id equals stop_node_id and who appear in this
        vehicle's customer_order.

        The original customer_order is preserved.
        """

        candidates = self._node_to_request.get(
            stop_node_id,
            [],
        )

        order_set = set(
            veh_route.customer_order
        )

        result = [
            (
                req.request_id,
                req,
            )
            for req in candidates
            if req.request_id in order_set
        ]

        order_index = {
            request_id: index
            for index, request_id
            in enumerate(
                veh_route.customer_order
            )
        }

        result.sort(
            key=lambda item: order_index.get(
                item[0],
                9999,
            )
        )

        return result

    # ------------------------------------------------------------------
    # Background trips
    # ------------------------------------------------------------------

    def _add_background_trips(
        self,
        parent: ET.Element,
    ) -> None:
        """
        Emit deterministic background traffic.

        Normal Step-6 behavior:
            Two deterministic corridors are generated from the scenario's
            mapped edges.

        Causal Step-11 behavior:
            Additional corridors are generated from the first mapped edge
            to caller-supplied affected edges.

        Explicitly passing background_target_edges=[] disables all
        background traffic.

        Background departures falling inside a closure interval are skipped.
        """

        # --------------------------------------------------------------
        # Explicit opt-out.
        # --------------------------------------------------------------

        if (
            self._background_target_edges_explicit
            and not self._background_target_edges
        ):
            return

        edge_ids = list(
            self._edge_map.values()
        )

        if len(edge_ids) < 2:
            logger.warning(
                "Cannot generate background traffic: "
                "fewer than two mapped edges"
            )
            return

        # --------------------------------------------------------------
        # Build valid corridors.
        # --------------------------------------------------------------

        corridors: list[
            tuple[str, str]
        ] = []

        def add_corridor(
            from_edge: str,
            to_edge: str,
        ) -> None:
            if (
                from_edge == to_edge
                or not from_edge
                or not to_edge
            ):
                return

            pair = (
                from_edge,
                to_edge,
            )

            if pair not in corridors:
                corridors.append(pair)

        # --------------------------------------------------------------
        # Normal Step-6 corridors.
        #
        # These reproduce the original deterministic behavior:
        #
        #   edge[0] -> edge[min(4)]
        #   edge[1] -> edge[min(5)]
        #
        # This is intentionally retained because Step 6's acceptance
        # tests expect background vehicles to be present and complete.
        # --------------------------------------------------------------

        normal_to_0 = edge_ids[
            min(
                4,
                len(edge_ids) - 1,
            )
        ]

        normal_to_1 = edge_ids[
            min(
                5,
                len(edge_ids) - 1,
            )
        ]

        add_corridor(
            edge_ids[0],
            normal_to_0,
        )

        if len(edge_ids) >= 2:
            add_corridor(
                edge_ids[1],
                normal_to_1,
            )

        # --------------------------------------------------------------
        # Causal event-targeted corridors.
        # --------------------------------------------------------------

        if self._background_target_edges:
            origin = edge_ids[0]
            valid_edges = set(edge_ids)

            for target in self._background_target_edges:
                if (
                    target in valid_edges
                    and target != origin
                ):
                    add_corridor(
                        origin,
                        target,
                    )

        if not corridors:
            logger.warning(
                "Cannot generate background traffic: "
                "no valid corridors"
            )
            return

        # --------------------------------------------------------------
        # Generate departure times.
        # --------------------------------------------------------------

        if (
            self._background_duration_s
            <= self._background_start_s
        ):
            departures = [
                self._background_start_s
            ]
        else:
            departures = []

            departure = (
                self._background_start_s
            )

            while (
                departure
                < self._background_duration_s
            ):
                departures.append(
                    departure
                )

                departure += (
                    self._background_interval_s
                )

        # --------------------------------------------------------------
        # Emit trips.
        # --------------------------------------------------------------

        trip_index = 0

        for departure in departures:

            if self._departure_is_blocked(
                departure
            ):
                continue

            for from_edge, to_edge in corridors:

                trip_id = f"bg_{trip_index}"

                ET.SubElement(
                    parent,
                    "trip",
                    attrib={
                        "id": trip_id,
                        "type": "background_car",
                        "from": from_edge,
                        "to": to_edge,
                        "depart": (
                            f"{departure:.1f}"
                        ),
                    },
                )

                trip_index += 1

    # ------------------------------------------------------------------
    # Closure filtering
    # ------------------------------------------------------------------

    def _departure_is_blocked(
        self,
        departure_s: float,
    ) -> bool:
        """
        Return True when a background departure falls inside a blocked
        interval.

        Semantics:
            start <= departure < end

        This matches the Step-11 requirement that traffic should not be
        newly introduced onto a scheduled closure during the closure window.
        """

        for start_s, end_s in (
            self._background_blocked_intervals
        ):
            if (
                start_s
                <= departure_s
                < end_s
            ):
                return True

        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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