"""
Step 16.3 — Multi-scale causal traffic representation.

Builds traffic representations at four deterministic spatial scales:

    edge -> parent road -> node neighborhood -> zone

Only values visible in the supplied GraphState are aggregated. Missing
observations are excluded from numeric traffic aggregates and represented
through explicit coverage / missingness statistics.

This module is model-independent. It does not perform GNN message passing,
temporal modelling, PPO, QPSO, or simulator interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .graph_state import (
    EDGE_FEATURE_NAMES,
    GraphState,
)

# Dynamic traffic columns in GraphState.edge_features.
_SPEED = EDGE_FEATURE_NAMES.index("speed_ratio")
_TRAVEL = EDGE_FEATURE_NAMES.index("travel_time_ratio")
_OCC = EDGE_FEATURE_NAMES.index("occupancy")
_HALT = EDGE_FEATURE_NAMES.index("halting_count_normalized")
_AGE = EDGE_FEATURE_NAMES.index("observation_age_normalized")
_MISSING = EDGE_FEATURE_NAMES.index("missing_observation")
_CLOSED = EDGE_FEATURE_NAMES.index("known_closed")
_AVAILABLE = EDGE_FEATURE_NAMES.index("traffic_observation_available")

TRAFFIC_FEATURE_NAMES: tuple[str, ...] = (
    "mean_speed_ratio",
    "mean_travel_time_ratio",
    "mean_occupancy",
    "mean_halting_count_normalized",
    "mean_observation_age_normalized",
    "closed_fraction",
    "missing_fraction",
    "traffic_coverage",
    "edge_count_normalized",
)

SCALE_FEATURE_NAMES: tuple[str, ...] = TRAFFIC_FEATURE_NAMES


@dataclass(frozen=True)
class MultiScaleTrafficState:
    """Immutable multi-scale spatial traffic representation."""

    parent_road_ids: tuple[str, ...]
    zone_ids: tuple[str, ...]

    parent_road_features: np.ndarray
    node_neighborhood_features: np.ndarray
    zone_features: np.ndarray

    parent_road_edge_index: np.ndarray
    zone_edge_index: np.ndarray

    node_ids: tuple[str, ...]
    node_id_to_index: Mapping[str, int]
    parent_road_id_to_index: Mapping[str, int]
    zone_id_to_index: Mapping[str, int]

    def __post_init__(self) -> None:
        node_count = len(self.node_ids)
        road_count = len(self.parent_road_ids)
        zone_count = len(self.zone_ids)

        expected = len(SCALE_FEATURE_NAMES)
        if self.parent_road_features.shape != (road_count, expected):
            raise ValueError(
                f"parent_road_features shape {self.parent_road_features.shape}; "
                f"expected {(road_count, expected)}"
            )
        if self.node_neighborhood_features.shape != (node_count, expected):
            raise ValueError(
                f"node_neighborhood_features shape "
                f"{self.node_neighborhood_features.shape}; "
                f"expected {(node_count, expected)}"
            )
        if self.zone_features.shape != (zone_count, expected):
            raise ValueError(
                f"zone_features shape {self.zone_features.shape}; "
                f"expected {(zone_count, expected)}"
            )
        if self.parent_road_edge_index.shape != (2, road_count):
            raise ValueError(
                f"parent_road_edge_index shape {self.parent_road_edge_index.shape}; "
                f"expected {(2, road_count)}"
            )
        if self.zone_edge_index.shape != (2, zone_count):
            raise ValueError(
                f"zone_edge_index shape {self.zone_edge_index.shape}; "
                f"expected {(2, zone_count)}"
            )

        for name, array in (
            ("parent_road_features", self.parent_road_features),
            ("node_neighborhood_features", self.node_neighborhood_features),
            ("zone_features", self.zone_features),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or infinite values")

        if node_count and np.any(self.parent_road_edge_index < 0):
            raise ValueError("parent_road_edge_index contains negative indices")
        if node_count and np.any(self.parent_road_edge_index >= node_count):
            raise ValueError("parent_road_edge_index references unknown nodes")

        if node_count and np.any(self.zone_edge_index < 0):
            raise ValueError("zone_edge_index contains negative indices")
        if node_count and np.any(self.zone_edge_index >= node_count):
            raise ValueError("zone_edge_index references unknown nodes")

        if dict(self.node_id_to_index) != {
            value: i for i, value in enumerate(self.node_ids)
        }:
            raise ValueError("node_id_to_index is inconsistent")

        if dict(self.parent_road_id_to_index) != {
            value: i for i, value in enumerate(self.parent_road_ids)
        }:
            raise ValueError("parent_road_id_to_index is inconsistent")

        if dict(self.zone_id_to_index) != {
            value: i for i, value in enumerate(self.zone_ids)
        }:
            raise ValueError("zone_id_to_index is inconsistent")

        self.parent_road_features.setflags(write=False)
        self.node_neighborhood_features.setflags(write=False)
        self.zone_features.setflags(write=False)
        self.parent_road_edge_index.setflags(write=False)
        self.zone_edge_index.setflags(write=False)

    @property
    def parent_road_count(self) -> int:
        return len(self.parent_road_ids)

    @property
    def zone_count(self) -> int:
        return len(self.zone_ids)

    @property
    def feature_count(self) -> int:
        return len(SCALE_FEATURE_NAMES)


class MultiScaleTrafficBuilder:
    """Construct causal multi-scale traffic features from Scenario + GraphState."""

    def __init__(self, scenario) -> None:
        self.scenario = scenario
        self._nodes = {
            str(node.node_id): node for node in scenario.nodes
        }
        self._edges = {
            str(edge.edge_id): edge for edge in scenario.edges
        }

        self._node_ids = tuple(sorted(self._nodes))
        self._edge_ids = tuple(sorted(self._edges))

        self._parent_road_ids = tuple(
            sorted({str(edge.parent_road_id) for edge in scenario.edges})
        )
        self._zone_ids = tuple(
            sorted({str(node.zone_id) for node in scenario.nodes})
        )

        self._parent_road_index = {
            value: i for i, value in enumerate(self._parent_road_ids)
        }
        self._zone_index = {
            value: i for i, value in enumerate(self._zone_ids)
        }
        self._node_index = {
            value: i for i, value in enumerate(self._node_ids)
        }

        self._edges_by_parent = {key: [] for key in self._parent_road_ids}
        for edge in self._edges.values():
            self._edges_by_parent[str(edge.parent_road_id)].append(str(edge.edge_id))

        self._edges_by_zone = {key: [] for key in self._zone_ids}
        for edge in self._edges.values():
            # A directed edge belongs to the zone of its source node. If an
            # edge crosses zones it is still represented exactly once, while
            # node-neighborhood features retain both endpoints.
            zone = str(self._nodes[str(edge.from_node)].zone_id)
            self._edges_by_zone.setdefault(zone, []).append(str(edge.edge_id))

        self._incident_edges = {key: [] for key in self._node_ids}
        for edge in self._edges.values():
            self._incident_edges[str(edge.from_node)].append(str(edge.edge_id))
            self._incident_edges[str(edge.to_node)].append(str(edge.edge_id))

        max_parent = max((len(v) for v in self._edges_by_parent.values()), default=1)
        max_zone = max((len(v) for v in self._edges_by_zone.values()), default=1)
        max_node = max((len(v) for v in self._incident_edges.values()), default=1)
        self._max_group_size = float(max(max_parent, max_zone, max_node, 1))

    def build(self, graph_state: GraphState) -> MultiScaleTrafficState:
        if graph_state.scenario_id != str(self.scenario.scenario_id):
            raise ValueError("GraphState scenario_id does not match Scenario")
        if graph_state.graph_version != str(self.scenario.graph_version):
            raise ValueError("GraphState graph_version does not match Scenario")
        if graph_state.node_ids != self._node_ids:
            raise ValueError("GraphState node ordering does not match Scenario")
        if graph_state.edge_ids != self._edge_ids:
            raise ValueError("GraphState edge ordering does not match Scenario")

        parent_features = np.zeros(
            (len(self._parent_road_ids), len(SCALE_FEATURE_NAMES)),
            dtype=np.float32,
        )
        node_features = np.zeros(
            (len(self._node_ids), len(SCALE_FEATURE_NAMES)),
            dtype=np.float32,
        )
        zone_features = np.zeros(
            (len(self._zone_ids), len(SCALE_FEATURE_NAMES)),
            dtype=np.float32,
        )

        for i, road_id in enumerate(self._parent_road_ids):
            indices = [graph_state.edge_id_to_index[e] for e in self._edges_by_parent[road_id]]
            parent_features[i] = self._aggregate(graph_state.edge_features[indices])

        for i, node_id in enumerate(self._node_ids):
            indices = [graph_state.edge_id_to_index[e] for e in self._incident_edges[node_id]]
            node_features[i] = self._aggregate(graph_state.edge_features[indices])

        for i, zone_id in enumerate(self._zone_ids):
            indices = [graph_state.edge_id_to_index[e] for e in self._edges_by_zone.get(zone_id, [])]
            zone_features[i] = self._aggregate(graph_state.edge_features[indices])

        # Parent-road representatives use the first physical edge in each
        # road group. This preserves a deterministic road->graph anchor.
        parent_edge_index = np.zeros((2, len(self._parent_road_ids)), dtype=np.int64)
        for i, road_id in enumerate(self._parent_road_ids):
            edge = self._edges[self._edges_by_parent[road_id][0]]
            parent_edge_index[0, i] = self._node_index[str(edge.from_node)]
            parent_edge_index[1, i] = self._node_index[str(edge.to_node)]

        # Zone representatives use one deterministic edge whose source node
        # belongs to that zone. Empty zones are impossible for Scenario's
        # zone set because every zone comes from a node.
        zone_edge_index = np.zeros((2, len(self._zone_ids)), dtype=np.int64)
        for i, zone_id in enumerate(self._zone_ids):
            candidates = self._edges_by_zone.get(zone_id, [])
            if candidates:
                edge = self._edges[candidates[0]]
                zone_edge_index[0, i] = self._node_index[str(edge.from_node)]
                zone_edge_index[1, i] = self._node_index[str(edge.to_node)]
            else:
                nodes = [n for n in self._node_ids if str(self._nodes[n].zone_id) == zone_id]
                anchor = self._node_index[nodes[0]]
                zone_edge_index[:, i] = anchor

        return MultiScaleTrafficState(
            parent_road_ids=self._parent_road_ids,
            zone_ids=self._zone_ids,
            parent_road_features=parent_features,
            node_neighborhood_features=node_features,
            zone_features=zone_features,
            parent_road_edge_index=parent_edge_index,
            zone_edge_index=zone_edge_index,
            node_ids=self._node_ids,
            node_id_to_index=dict(self._node_index),
            parent_road_id_to_index=dict(self._parent_road_index),
            zone_id_to_index=dict(self._zone_index),
        )

    @staticmethod
    def _aggregate(values: np.ndarray) -> np.ndarray:
        result = np.zeros(len(SCALE_FEATURE_NAMES), dtype=np.float32)
        if len(values) == 0:
            return result

        available = values[:, _AVAILABLE] > 0.5
        total = float(len(values))
        observed = values[available]

        # Numeric traffic aggregates only use observed edges.
        if len(observed):
            result[0] = float(np.mean(observed[:, _SPEED]))
            result[1] = float(np.mean(observed[:, _TRAVEL]))
            result[2] = float(np.mean(observed[:, _OCC]))
            result[3] = float(np.mean(observed[:, _HALT]))
            result[4] = float(np.mean(observed[:, _AGE]))

        result[5] = float(np.mean(values[:, _CLOSED]))
        result[6] = float(np.mean(values[:, _MISSING]))
        result[7] = float(np.mean(available))
        result[8] = min(1.0, total / 1.0)
        return result
