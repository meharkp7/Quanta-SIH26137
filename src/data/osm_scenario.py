"""Production-grade integration of an OSM-derived network into a Quanta Scenario.

The integration boundary is intentionally explicit:

* OSM provides the directed road graph.
* The existing Quanta Scenario remains the source of demand/fleet state.
* Request/vehicle locations are mapped to OSM nodes explicitly.
* No synthetic coordinate is silently interpreted as real geography.
* The resulting Scenario is reconstructed through Pydantic validation.

This module does not acquire data from the network. Use ``osm_ingestion`` to
load a local OSM/GraphML extract first.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

try:
    import numpy as np
    from scipy.spatial import cKDTree
except ImportError:  # optional acceleration
    np = None
    cKDTree = None

from src.contracts.scenario import RoadNode, Scenario
from src.contracts.core_types import NodeKind, RoadNodeId
from .osm_ingestion import OSMNetwork


class OSMScenarioError(ValueError):
    """Raised when an OSM network cannot safely become a Quanta scenario."""


@dataclass(frozen=True, slots=True)
class OSMSnapConfig:
    """Safety policy for projected-coordinate snapping."""

    max_snap_distance_m: float = 500.0
    require_projected_coordinates: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_snap_distance_m) or self.max_snap_distance_m <= 0:
            raise OSMScenarioError("max_snap_distance_m must be finite and positive")


@dataclass(frozen=True, slots=True)
class OSMSnapResult:
    node_id: RoadNodeId
    distance_m: float


@dataclass(frozen=True, slots=True)
class OSMScenarioBuildResult:
    scenario: Scenario
    request_snap_distances_m: tuple[float, ...]
    vehicle_snap_distances_m: tuple[float, ...]
    network_fingerprint: str


def _xy(value: Any, label: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            raise OSMScenarioError(f"{label} must contain x and y")
        x, y = value["x"], value["y"]
    else:
        try:
            if len(value) != 2:
                raise ValueError
            x, y = value
        except (TypeError, ValueError) as exc:
            raise OSMScenarioError(f"{label} must be an (x, y) pair") from exc
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError) as exc:
        raise OSMScenarioError(f"{label} contains non-numeric coordinates") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise OSMScenarioError(f"{label} contains non-finite coordinates")
    return x, y


class _NearestNodeIndex:
    """Deterministic nearest-node index; KD-tree for scale, exact fallback for minimal installs."""

    def __init__(self, nodes: tuple[RoadNode, ...]) -> None:
        if not nodes:
            raise OSMScenarioError("OSM network contains no nodes")
        self.nodes = nodes
        self.coords = [(float(n.x_m), float(n.y_m)) for n in nodes]
        self.tree = (
            cKDTree(np.asarray(self.coords, dtype=float))
            if cKDTree is not None and np is not None
            else None
        )

    def nearest(self, coordinate: Any, max_distance_m: float) -> OSMSnapResult:
        x, y = _xy(coordinate, "coordinate")
        if self.tree is not None:
            distance, idx = self.tree.query([x, y], k=1)
            idx, distance = int(idx), float(distance)
        else:
            idx, distance = min(
                enumerate(self.coords),
                key=lambda item: (
                    (item[1][0] - x) ** 2 + (item[1][1] - y) ** 2,
                    str(self.nodes[item[0]].node_id),
                ),
            )
            distance = math.hypot(self.coords[idx][0] - x, self.coords[idx][1] - y)
        if distance > max_distance_m:
            raise OSMScenarioError(
                f"Nearest OSM node is {distance:.3f} m away, exceeding "
                f"max_snap_distance_m={max_distance_m:.3f}"
            )
        return OSMSnapResult(self.nodes[idx].node_id, distance)


class OSMScenarioBuilder:
    """Attach an already-ingested OSM network to an existing Quanta Scenario."""

    def __init__(
        self,
        network: OSMNetwork,
        *,
        snap_config: OSMSnapConfig | None = None,
    ) -> None:
        self.network = network
        self.snap_config = snap_config or OSMSnapConfig()
        if self.snap_config.require_projected_coordinates and not network.projected:
            raise OSMScenarioError(
                "OSM network must be projected into a metric CRS before scenario integration"
            )
        self._nodes = tuple(network.nodes)
        self._node_ids = {n.node_id for n in self._nodes}
        if len(self._node_ids) != len(self._nodes):
            raise OSMScenarioError("OSM network contains duplicate node IDs")
        self._index = _NearestNodeIndex(self._nodes)

    def snap(self, coordinate: Any) -> OSMSnapResult:
        return self._index.nearest(
            coordinate,
            self.snap_config.max_snap_distance_m,
        )

    def build(
        self,
        scenario: Scenario,
        *,
        request_coordinates: Mapping[str, Any] | None = None,
        vehicle_start_nodes: Mapping[str, RoadNodeId] | None = None,
        vehicle_depot_nodes: Mapping[str, RoadNodeId] | None = None,
        scenario_id: str | None = None,
    ) -> OSMScenarioBuildResult:
        """Return a fully validated Scenario using the OSM road graph.

        ``request_coordinates`` is keyed by ``Request.request_id`` and must
        use the same projected metric CRS as the OSM network. Existing OSM
        node IDs may be used only if they already belong to this network.

        Vehicles have no coordinate fields in the current contract, so
        production callers must explicitly provide OSM start/depot mappings
        when their existing node IDs are not OSM IDs.
        """
        request_coordinates = request_coordinates or {}
        vehicle_start_nodes = vehicle_start_nodes or {}
        vehicle_depot_nodes = vehicle_depot_nodes or {}

        request_snaps: list[float] = []
        new_requests = []

        access_ids: set[RoadNodeId] = set()
        for request in scenario.requests:
            rid = str(request.request_id)
            access_id = request.access_node_id
            if access_id in self._node_ids:
                new_request = request
            else:
                coordinate = request_coordinates.get(rid)
                if coordinate is None:
                    raise OSMScenarioError(
                        f"Request {rid!r} does not reference an OSM node and has no "
                        "explicit projected coordinate mapping"
                    )
                snap = self.snap(coordinate)
                request_snaps.append(snap.distance_m)
                new_request = request.model_copy(
                    update={
                        "access_node_id": snap.node_id,
                        "access_distance_m": snap.distance_m,
                    }
                )
            access_ids.add(new_request.access_node_id)
            new_requests.append(new_request)

        new_fleet = []
        depot_ids: set[RoadNodeId] = set()
        for vehicle in scenario.fleet:
            vid = str(vehicle.vehicle_id)
            start_id = vehicle_start_nodes.get(vid, vehicle.start_node_id)
            depot_id = vehicle_depot_nodes.get(vid, vehicle.depot_node_id)

            if start_id not in self._node_ids:
                raise OSMScenarioError(
                    f"Vehicle {vid!r} start node {start_id!r} is not in the OSM network; "
                    "provide vehicle_start_nodes"
                )
            if depot_id not in self._node_ids:
                raise OSMScenarioError(
                    f"Vehicle {vid!r} depot node {depot_id!r} is not in the OSM network; "
                    "provide vehicle_depot_nodes"
                )

            new_fleet.append(
                vehicle.model_copy(
                    update={"start_node_id": start_id, "depot_node_id": depot_id}
                )
            )
            depot_ids.add(depot_id)

        marked_nodes = []
        for node in self._nodes:
            kind = node.kind
            if node.node_id in depot_ids:
                kind = NodeKind.DEPOT
            elif node.node_id in access_ids:
                kind = NodeKind.CUSTOMER_ACCESS
            marked_nodes.append(node.model_copy(update={"kind": kind}))

        # Validate references before constructing the final Scenario.
        node_ids = {n.node_id for n in marked_nodes}
        for edge in self.network.edges:
            if edge.from_node not in node_ids or edge.to_node not in node_ids:
                raise OSMScenarioError(
                    f"OSM edge {edge.edge_id!r} references a missing node"
                )

        fingerprint = self.network_fingerprint()
        new_metadata = dict(scenario.field_provenance)
        new_metadata.update({
            "osm_network_fingerprint": fingerprint,
            "osm_source_crs": self.network.source_crs or "unknown",
            "osm_projected": str(self.network.projected),
            "osm_request_snap_count": str(len(request_snaps)),
            "osm_max_request_snap_distance_m": str(max(request_snaps, default=0.0)),
            "osm_integration": "directed OSM road graph attached to existing Quanta demand/fleet",
        })

        payload = scenario.model_dump(mode="python")
        payload.update({
            "scenario_id": scenario_id or f"{scenario.scenario_id}:osm:{fingerprint[:12]}",
            "source_name": f"{scenario.source_name}:osm",
            "source_checksum": fingerprint,
            "graph_version": f"{scenario.scenario_id}:osm:{fingerprint[:16]}",
            "nodes": tuple(marked_nodes),
            "edges": tuple(self.network.edges),
            "requests": tuple(new_requests),
            "fleet": tuple(new_fleet),
            "field_provenance": new_metadata,
            "parent_instance_id": scenario.parent_instance_id or str(scenario.scenario_id),
        })

        # model_validate is deliberate: model_copy(update=...) does not
        # validate the complete updated model, while Scenario's cross-field
        # validators are an important production safety boundary.
        rebuilt = Scenario.model_validate(payload)

        return OSMScenarioBuildResult(
            scenario=rebuilt,
            request_snap_distances_m=tuple(request_snaps),
            vehicle_snap_distances_m=(),
            network_fingerprint=fingerprint,
        )

    def network_fingerprint(self) -> str:
        nodes = sorted(
            (
                str(n.node_id),
                round(float(n.x_m), 9),
                round(float(n.y_m), 9),
            )
            for n in self.network.nodes
        )
        edges = sorted(
            (
                str(e.edge_id),
                str(e.from_node),
                str(e.to_node),
                round(float(e.length_m), 6),
                round(float(e.speed_limit_mps), 6),
                int(e.lane_count),
                round(float(e.capacity_veh_per_hour), 6),
                str(e.road_class),
                bool(e.open_by_default),
            )
            for e in self.network.edges
        )
        blob = json.dumps(
            {"nodes": nodes, "edges": edges},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
