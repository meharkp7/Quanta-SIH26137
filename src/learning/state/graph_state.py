"""
Step 16.2 — Causal graph-state representation.

This module converts the immutable Scenario road graph and the currently
visible Observation into a stable tensor representation suitable for later
GNN / temporal / PPO components.

Architectural boundary
----------------------
Scenario
    provides:
        - physical directed road topology
        - static road attributes
        - node coordinates/types
        - lane/capacity/speed information

Observation
    provides:
        - currently observed traffic values
        - missingness
        - currently known closures

This module MUST NOT consume:
    - EnvironmentTruth
    - HiddenEvent
    - future labels
    - future observations
    - simulator truth

The resulting GraphState is model-independent. It does not perform GNN
message passing or learning.

Stable ordering
---------------
Nodes are ordered deterministically by node_id.
Edges are ordered deterministically by edge_id.

Integer indices are positional indices only. They carry no semantic meaning.

Tensor conventions
------------------
node_features:
    [N_nodes, F_node]

edge_features:
    [N_edges, F_edge]

edge_index:
    [2, N_edges]
    row 0 = source node index
    row 1 = destination node index

edge_observation_mask:
    [N_edges, 4]

    Columns:
        0 speed observed
        1 travel time observed
        2 occupancy observed
        3 halting count observed

Missing observations are NEVER silently treated as observed zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from src.contracts.observation import (
    EdgeObservation,
    Observation,
)
from src.contracts.scenario import (
    RoadEdge,
    Scenario,
)


# ============================================================================
# Feature contracts
# ============================================================================

NODE_FEATURE_NAMES: tuple[str, ...] = (
    "x_normalized",
    "y_normalized",
    "kind_junction",
    "kind_depot",
    "kind_customer_access",
    "kind_other",
    "signalized",
    "in_degree_normalized",
    "out_degree_normalized",
)


EDGE_FEATURE_NAMES: tuple[str, ...] = (
    # ------------------------------------------------------------------
    # Static road attributes.
    # ------------------------------------------------------------------
    "length_normalized",
    "speed_limit_normalized",
    "lane_count_normalized",
    "capacity_normalized",
    "free_flow_time_normalized",

    # ------------------------------------------------------------------
    # Road-class one-hot representation.
    # ------------------------------------------------------------------
    "road_class_local",
    "road_class_collector",
    "road_class_arterial",
    "road_class_access_connector",

    # ------------------------------------------------------------------
    # Current observed traffic state.
    # ------------------------------------------------------------------
    "speed_ratio",
    "travel_time_ratio",
    "occupancy",
    "halting_count_normalized",
    "observation_age_normalized",

    # ------------------------------------------------------------------
    # Explicit observation / operational state.
    # ------------------------------------------------------------------
    "missing_observation",
    "known_closed",
    "open_by_default",
    "traffic_observation_available",
)


OBSERVATION_MASK_NAMES: tuple[str, ...] = (
    "speed_observed",
    "travel_time_observed",
    "occupancy_observed",
    "halting_count_observed",
)


# ============================================================================
# GraphState
# ============================================================================


@dataclass(frozen=True)
class GraphState:
    """
    Immutable numerical representation of one currently visible road graph.

    The arrays are made read-only so that downstream model code cannot
    accidentally mutate the state from which a decision was generated.
    """

    scenario_id: str
    episode_id: str
    graph_version: str
    observation_time_s: float

    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]

    node_features: np.ndarray
    edge_features: np.ndarray

    edge_index: np.ndarray

    edge_observation_mask: np.ndarray

    node_id_to_index: Mapping[str, int]
    edge_id_to_index: Mapping[str, int]

    def __post_init__(self) -> None:
        # ------------------------------------------------------------------
        # Metadata.
        # ------------------------------------------------------------------

        if not isfinite(
            float(self.observation_time_s)
        ):
            raise ValueError(
                "observation_time_s must be finite"
            )

        if self.observation_time_s < 0:
            raise ValueError(
                "observation_time_s must be non-negative"
            )

        node_count = len(self.node_ids)
        edge_count = len(self.edge_ids)

        # ------------------------------------------------------------------
        # Shape validation.
        # ------------------------------------------------------------------

        expected_node_shape = (
            node_count,
            len(NODE_FEATURE_NAMES),
        )

        if self.node_features.shape != expected_node_shape:
            raise ValueError(
                "node_features has inconsistent shape: "
                f"{self.node_features.shape}; expected "
                f"{expected_node_shape}"
            )

        expected_edge_shape = (
            edge_count,
            len(EDGE_FEATURE_NAMES),
        )

        if self.edge_features.shape != expected_edge_shape:
            raise ValueError(
                "edge_features has inconsistent shape: "
                f"{self.edge_features.shape}; expected "
                f"{expected_edge_shape}"
            )

        expected_edge_index_shape = (
            2,
            edge_count,
        )

        if self.edge_index.shape != expected_edge_index_shape:
            raise ValueError(
                "edge_index must have shape "
                f"{expected_edge_index_shape}; got "
                f"{self.edge_index.shape}"
            )

        expected_mask_shape = (
            edge_count,
            len(OBSERVATION_MASK_NAMES),
        )

        if (
            self.edge_observation_mask.shape
            != expected_mask_shape
        ):
            raise ValueError(
                "edge_observation_mask has inconsistent shape: "
                f"{self.edge_observation_mask.shape}; expected "
                f"{expected_mask_shape}"
            )

        # ------------------------------------------------------------------
        # Numerical safety.
        # ------------------------------------------------------------------

        if not np.all(
            np.isfinite(self.node_features)
        ):
            raise ValueError(
                "node_features contains NaN or infinite values"
            )

        if not np.all(
            np.isfinite(self.edge_features)
        ):
            raise ValueError(
                "edge_features contains NaN or infinite values"
            )

        # ------------------------------------------------------------------
        # Topology validity.
        # ------------------------------------------------------------------

        if edge_count > 0:
            if np.any(
                self.edge_index < 0
            ):
                raise ValueError(
                    "edge_index contains negative node indices"
                )

            if np.any(
                self.edge_index >= node_count
            ):
                raise ValueError(
                    "edge_index references a node outside node_ids"
                )

        # ------------------------------------------------------------------
        # Lookup-map consistency.
        # ------------------------------------------------------------------

        expected_node_lookup = {
            node_id: index
            for index, node_id
            in enumerate(self.node_ids)
        }

        expected_edge_lookup = {
            edge_id: index
            for index, edge_id
            in enumerate(self.edge_ids)
        }

        if dict(
            self.node_id_to_index
        ) != expected_node_lookup:
            raise ValueError(
                "node_id_to_index is inconsistent with node_ids"
            )

        if dict(
            self.edge_id_to_index
        ) != expected_edge_lookup:
            raise ValueError(
                "edge_id_to_index is inconsistent with edge_ids"
            )

        # ------------------------------------------------------------------
        # Immutability.
        # ------------------------------------------------------------------

        self.node_features.setflags(
            write=False
        )

        self.edge_features.setflags(
            write=False
        )

        self.edge_index.setflags(
            write=False
        )

        self.edge_observation_mask.setflags(
            write=False
        )

    # ------------------------------------------------------------------
    # Convenience properties.
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def edge_count(self) -> int:
        return len(self.edge_ids)

    @property
    def node_feature_count(self) -> int:
        return self.node_features.shape[1]

    @property
    def edge_feature_count(self) -> int:
        return self.edge_features.shape[1]

    # ------------------------------------------------------------------
    # Lookup helpers.
    # ------------------------------------------------------------------

    def edge_features_for(
        self,
        edge_id: str,
    ) -> np.ndarray:
        try:
            index = self.edge_id_to_index[
                edge_id
            ]
        except KeyError as exc:
            raise KeyError(
                f"unknown edge_id={edge_id!r}"
            ) from exc

        return self.edge_features[index]

    def node_features_for(
        self,
        node_id: str,
    ) -> np.ndarray:
        try:
            index = self.node_id_to_index[
                node_id
            ]
        except KeyError as exc:
            raise KeyError(
                f"unknown node_id={node_id!r}"
            ) from exc

        return self.node_features[index]

    def edge_index_for(
        self,
        edge_id: str,
    ) -> int:
        try:
            return self.edge_id_to_index[
                edge_id
            ]
        except KeyError as exc:
            raise KeyError(
                f"unknown edge_id={edge_id!r}"
            ) from exc


# ============================================================================
# GraphStateBuilder
# ============================================================================


class GraphStateBuilder:
    """
    Build a causal GraphState from Scenario + current Observation.

    Static normalization references are derived exclusively from the
    Scenario. Dynamic values come exclusively from the supplied Observation.
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        age_normalization_s: float = 300.0,
        halting_normalization: float = 20.0,
    ) -> None:

        if age_normalization_s <= 0:
            raise ValueError(
                "age_normalization_s must be positive"
            )

        if halting_normalization <= 0:
            raise ValueError(
                "halting_normalization must be positive"
            )

        self.scenario = scenario

        self.age_normalization_s = float(
            age_normalization_s
        )

        self.halting_normalization = float(
            halting_normalization
        )

        # --------------------------------------------------------------
        # Stable topology ordering.
        # --------------------------------------------------------------

        self._nodes = tuple(
            sorted(
                scenario.nodes,
                key=lambda node: str(
                    node.node_id
                ),
            )
        )

        self._edges = tuple(
            sorted(
                scenario.edges,
                key=lambda edge: str(
                    edge.edge_id
                ),
            )
        )

        self.node_id_to_index = {
            str(node.node_id): index
            for index, node
            in enumerate(self._nodes)
        }

        self.edge_id_to_index = {
            str(edge.edge_id): index
            for index, edge
            in enumerate(self._edges)
        }

        self._validate_topology()

        # --------------------------------------------------------------
        # Static normalization references.
        # --------------------------------------------------------------

        self._max_length_m = self._positive_max(
            [
                float(edge.length_m)
                for edge in self._edges
            ],
            fallback=1.0,
        )

        self._max_speed_limit_mps = self._positive_max(
            [
                float(edge.speed_limit_mps)
                for edge in self._edges
            ],
            fallback=1.0,
        )

        self._max_lane_count = self._positive_max(
            [
                float(edge.lane_count)
                for edge in self._edges
            ],
            fallback=1.0,
        )

        self._max_capacity = self._positive_max(
            [
                float(
                    edge.capacity_veh_per_hour
                )
                for edge in self._edges
            ],
            fallback=1.0,
        )

        self._max_free_flow_time_s = self._positive_max(
            [
                float(
                    edge.free_flow_time_s
                )
                for edge in self._edges
            ],
            fallback=1.0,
        )

        # --------------------------------------------------------------
        # Coordinate normalization.
        # --------------------------------------------------------------

        xs = [
            float(node.x_m)
            for node in self._nodes
        ]

        ys = [
            float(node.y_m)
            for node in self._nodes
        ]

        self._min_x = (
            min(xs)
            if xs
            else 0.0
        )

        self._max_x = (
            max(xs)
            if xs
            else 1.0
        )

        self._min_y = (
            min(ys)
            if ys
            else 0.0
        )

        self._max_y = (
            max(ys)
            if ys
            else 1.0
        )

        self._x_span = max(
            self._max_x - self._min_x,
            1e-9,
        )

        self._y_span = max(
            self._max_y - self._min_y,
            1e-9,
        )

        # --------------------------------------------------------------
        # Degree normalization.
        # --------------------------------------------------------------

        self._max_in_degree = max(
            (
                self._in_degree(
                    str(node.node_id)
                )
                for node in self._nodes
            ),
            default=1,
        )

        self._max_out_degree = max(
            (
                self._out_degree(
                    str(node.node_id)
                )
                for node in self._nodes
            ),
            default=1,
        )

    # ==================================================================
    # Public API
    # ==================================================================

    def build(
        self,
        observation: Observation,
    ) -> GraphState:
        """
        Build the causal graph representation for one observation.
        """

        self._validate_observation(
            observation
        )

        observations_by_edge = {
            str(item.edge_id): item
            for item
            in observation.edge_observations
        }

        node_features = (
            self._build_node_features()
        )

        edge_features = np.zeros(
            (
                len(self._edges),
                len(EDGE_FEATURE_NAMES),
            ),
            dtype=np.float32,
        )

        observation_mask = np.zeros(
            (
                len(self._edges),
                len(
                    OBSERVATION_MASK_NAMES
                ),
            ),
            dtype=np.float32,
        )

        edge_index = np.zeros(
            (
                2,
                len(self._edges),
            ),
            dtype=np.int64,
        )

        for position, edge in enumerate(
            self._edges
        ):
            edge_id = str(
                edge.edge_id
            )

            edge_index[
                0,
                position,
            ] = self.node_id_to_index[
                str(edge.from_node)
            ]

            edge_index[
                1,
                position,
            ] = self.node_id_to_index[
                str(edge.to_node)
            ]

            edge_observation = (
                observations_by_edge.get(
                    edge_id
                )
            )

            edge_features[
                position
            ] = self._build_edge_features(
                edge,
                edge_observation,
            )

            observation_mask[
                position
            ] = self._build_observation_mask(
                edge_observation
            )

        return GraphState(
            scenario_id=str(
                observation.scenario_id
            ),
            episode_id=str(
                observation.episode_id
            ),
            graph_version=str(
                observation.graph_version
            ),
            observation_time_s=float(
                observation.observation_time_s
            ),
            node_ids=tuple(
                str(node.node_id)
                for node in self._nodes
            ),
            edge_ids=tuple(
                str(edge.edge_id)
                for edge in self._edges
            ),
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index,
            edge_observation_mask=(
                observation_mask
            ),
            node_id_to_index=dict(
                self.node_id_to_index
            ),
            edge_id_to_index=dict(
                self.edge_id_to_index
            ),
        )

    # ==================================================================
    # Node features
    # ==================================================================

    def _build_node_features(
        self,
    ) -> np.ndarray:

        features = np.zeros(
            (
                len(self._nodes),
                len(NODE_FEATURE_NAMES),
            ),
            dtype=np.float32,
        )

        for index, node in enumerate(
            self._nodes
        ):
            x = (
                float(node.x_m)
                - self._min_x
            ) / self._x_span

            y = (
                float(node.y_m)
                - self._min_y
            ) / self._y_span

            features[
                index,
                NODE_FEATURE_NAMES.index(
                    "x_normalized"
                ),
            ] = np.clip(
                x,
                0.0,
                1.0,
            )

            features[
                index,
                NODE_FEATURE_NAMES.index(
                    "y_normalized"
                ),
            ] = np.clip(
                y,
                0.0,
                1.0,
            )

            kind = self._enum_value(
                node.kind
            )

            kind_index = {
                "junction": (
                    NODE_FEATURE_NAMES.index(
                        "kind_junction"
                    )
                ),
                "depot": (
                    NODE_FEATURE_NAMES.index(
                        "kind_depot"
                    )
                ),
                "customer_access": (
                    NODE_FEATURE_NAMES.index(
                        "kind_customer_access"
                    )
                ),
                "other": (
                    NODE_FEATURE_NAMES.index(
                        "kind_other"
                    )
                ),
            }.get(kind)

            if kind_index is not None:
                features[
                    index,
                    kind_index,
                ] = 1.0

            features[
                index,
                NODE_FEATURE_NAMES.index(
                    "signalized"
                ),
            ] = float(
                bool(node.signalized)
            )

            features[
                index,
                NODE_FEATURE_NAMES.index(
                    "in_degree_normalized"
                ),
            ] = np.clip(
                self._in_degree(
                    str(node.node_id)
                )
                / self._max_in_degree,
                0.0,
                1.0,
            )

            features[
                index,
                NODE_FEATURE_NAMES.index(
                    "out_degree_normalized"
                ),
            ] = np.clip(
                self._out_degree(
                    str(node.node_id)
                )
                / self._max_out_degree,
                0.0,
                1.0,
            )

        return features

    # ==================================================================
    # Edge features
    # ==================================================================

    def _build_edge_features(
        self,
        edge: RoadEdge,
        observation: EdgeObservation | None,
    ) -> np.ndarray:

        features = np.zeros(
            len(EDGE_FEATURE_NAMES),
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Static features.
        # --------------------------------------------------------------

        features[
            EDGE_FEATURE_NAMES.index(
                "length_normalized"
            )
        ] = self._safe_ratio(
            float(edge.length_m),
            self._max_length_m,
        )

        features[
            EDGE_FEATURE_NAMES.index(
                "speed_limit_normalized"
            )
        ] = self._safe_ratio(
            float(edge.speed_limit_mps),
            self._max_speed_limit_mps,
        )

        features[
            EDGE_FEATURE_NAMES.index(
                "lane_count_normalized"
            )
        ] = self._safe_ratio(
            float(edge.lane_count),
            self._max_lane_count,
        )

        features[
            EDGE_FEATURE_NAMES.index(
                "capacity_normalized"
            )
        ] = self._safe_ratio(
            float(
                edge.capacity_veh_per_hour
            ),
            self._max_capacity,
        )

        features[
            EDGE_FEATURE_NAMES.index(
                "free_flow_time_normalized"
            )
        ] = self._safe_ratio(
            float(
                edge.free_flow_time_s
            ),
            self._max_free_flow_time_s,
        )

        # --------------------------------------------------------------
        # Road class.
        # --------------------------------------------------------------

        road_class = self._enum_value(
            edge.road_class
        )

        road_class_features = {
            "local": "road_class_local",
            "collector": "road_class_collector",
            "arterial": "road_class_arterial",
            "access_connector": (
                "road_class_access_connector"
            ),
        }

        class_feature_name = (
            road_class_features.get(
                road_class
            )
        )

        if class_feature_name is not None:
            features[
                EDGE_FEATURE_NAMES.index(
                    class_feature_name
                )
            ] = 1.0

        # --------------------------------------------------------------
        # No observation means:
        #
        #   missing = 1
        #   traffic_observation_available = 0
        #
        # This is exactly what prevents missingness from being mistaken
        # for an observed zero.
        # --------------------------------------------------------------

        if observation is None:
            features[
                EDGE_FEATURE_NAMES.index(
                    "missing_observation"
                )
            ] = 1.0

            features[
                EDGE_FEATURE_NAMES.index(
                    "known_closed"
                )
            ] = 0.0

            features[
                EDGE_FEATURE_NAMES.index(
                    "open_by_default"
                )
            ] = float(
                bool(
                    edge.open_by_default
                )
            )

            features[
                EDGE_FEATURE_NAMES.index(
                    "traffic_observation_available"
                )
            ] = 0.0

            return features

        missing = bool(
            observation.missing
        )

        known_closed = bool(
            observation.known_closed
        )

        features[
            EDGE_FEATURE_NAMES.index(
                "missing_observation"
            )
        ] = float(missing)

        features[
            EDGE_FEATURE_NAMES.index(
                "known_closed"
            )
        ] = float(known_closed)

        features[
            EDGE_FEATURE_NAMES.index(
                "open_by_default"
            )
        ] = float(
            bool(
                edge.open_by_default
            )
        )

        features[
            EDGE_FEATURE_NAMES.index(
                "traffic_observation_available"
            )
        ] = float(
            not missing
        )

        # --------------------------------------------------------------
        # Speed.
        # --------------------------------------------------------------

        if (
            observation.observed_speed_mps
            is not None
            and not missing
        ):
            features[
                EDGE_FEATURE_NAMES.index(
                    "speed_ratio"
                )
            ] = np.clip(
                self._safe_ratio(
                    float(
                        observation.observed_speed_mps
                    ),
                    float(
                        edge.speed_limit_mps
                    ),
                ),
                0.0,
                2.0,
            )

        # --------------------------------------------------------------
        # Travel time.
        # --------------------------------------------------------------

        if (
            observation.observed_travel_time_s
            is not None
            and not missing
        ):
            features[
                EDGE_FEATURE_NAMES.index(
                    "travel_time_ratio"
                )
            ] = np.clip(
                self._safe_ratio(
                    float(
                        observation.observed_travel_time_s
                    ),
                    float(
                        edge.free_flow_time_s
                    ),
                ),
                0.0,
                10.0,
            )

        # --------------------------------------------------------------
        # Occupancy.
        # --------------------------------------------------------------

        if (
            observation.occupancy
            is not None
        ):
            features[
                EDGE_FEATURE_NAMES.index(
                    "occupancy"
                )
            ] = np.clip(
                float(
                    observation.occupancy
                ),
                0.0,
                1.0,
            )

        # --------------------------------------------------------------
        # Halting count.
        # --------------------------------------------------------------

        if (
            observation.halting_count
            is not None
        ):
            features[
                EDGE_FEATURE_NAMES.index(
                    "halting_count_normalized"
                )
            ] = np.clip(
                float(
                    observation.halting_count
                )
                / self.halting_normalization,
                0.0,
                1.0,
            )

        # --------------------------------------------------------------
        # Observation age.
        # --------------------------------------------------------------

        features[
            EDGE_FEATURE_NAMES.index(
                "observation_age_normalized"
            )
        ] = np.clip(
            float(
                observation.observation_age_s
            )
            / self.age_normalization_s,
            0.0,
            10.0,
        )

        return features

    # ==================================================================
    # Observation mask
    # ==================================================================

    @staticmethod
    def _build_observation_mask(
        observation: EdgeObservation | None,
    ) -> np.ndarray:

        mask = np.zeros(
            len(
                OBSERVATION_MASK_NAMES
            ),
            dtype=np.float32,
        )

        if observation is None:
            return mask

        mask[
            OBSERVATION_MASK_NAMES.index(
                "speed_observed"
            )
        ] = float(
            observation.observed_speed_mps
            is not None
        )

        mask[
            OBSERVATION_MASK_NAMES.index(
                "travel_time_observed"
            )
        ] = float(
            observation.observed_travel_time_s
            is not None
        )

        mask[
            OBSERVATION_MASK_NAMES.index(
                "occupancy_observed"
            )
        ] = float(
            observation.occupancy
            is not None
        )

        mask[
            OBSERVATION_MASK_NAMES.index(
                "halting_count_observed"
            )
        ] = float(
            observation.halting_count
            is not None
        )

        return mask

    # ==================================================================
    # Validation
    # ==================================================================

    def _validate_topology(
        self,
    ) -> None:

        known_node_ids = set(
            self.node_id_to_index
        )

        for edge in self._edges:
            if (
                str(edge.from_node)
                not in known_node_ids
            ):
                raise ValueError(
                    f"edge {edge.edge_id!r} references "
                    f"unknown from_node={edge.from_node!r}"
                )

            if (
                str(edge.to_node)
                not in known_node_ids
            ):
                raise ValueError(
                    f"edge {edge.edge_id!r} references "
                    f"unknown to_node={edge.to_node!r}"
                )

    def _validate_observation(
        self,
        observation: Observation,
    ) -> None:

        if (
            str(observation.scenario_id)
            != str(
                self.scenario.scenario_id
            )
        ):
            raise ValueError(
                "observation scenario_id does not match "
                "GraphStateBuilder scenario"
            )

        if (
            str(observation.graph_version)
            != str(
                self.scenario.graph_version
            )
        ):
            raise ValueError(
                "observation graph_version does not match "
                "Scenario graph_version"
            )

        if not isfinite(
            float(
                observation.observation_time_s
            )
        ):
            raise ValueError(
                "observation_time_s must be finite"
            )

        known_edge_ids = set(
            self.edge_id_to_index
        )

        seen_edge_ids: set[str] = set()

        for item in (
            observation.edge_observations
        ):
            edge_id = str(
                item.edge_id
            )

            if edge_id not in known_edge_ids:
                raise ValueError(
                    f"observation references unknown "
                    f"edge_id={edge_id!r}"
                )

            if edge_id in seen_edge_ids:
                raise ValueError(
                    f"duplicate edge observation "
                    f"for edge_id={edge_id!r}"
                )

            seen_edge_ids.add(
                edge_id
            )

    # ==================================================================
    # Graph utilities
    # ==================================================================

    def _in_degree(
        self,
        node_id: str,
    ) -> int:
        return sum(
            str(edge.to_node)
            == node_id
            for edge in self._edges
        )

    def _out_degree(
        self,
        node_id: str,
    ) -> int:
        return sum(
            str(edge.from_node)
            == node_id
            for edge in self._edges
        )

    # ==================================================================
    # Enum / numeric utilities
    # ==================================================================

    @staticmethod
    def _enum_value(
        value: object,
    ) -> str:
        """
        Return an enum's value when available, otherwise its string form.

        This keeps the builder compatible with both Pydantic enum-backed
        contracts and plain string-like values.
        """

        raw = getattr(
            value,
            "value",
            value,
        )

        return str(
            raw
        ).lower()

    @staticmethod
    def _positive_max(
        values: Sequence[float],
        *,
        fallback: float,
    ) -> float:

        positive = [
            value
            for value in values
            if isfinite(value)
            and value > 0
        ]

        if not positive:
            return fallback

        return max(
            positive
        )

    @staticmethod
    def _safe_ratio(
        numerator: float,
        denominator: float,
    ) -> float:

        if not isfinite(
            numerator
        ):
            return 0.0

        if not isfinite(
            denominator
        ):
            return 0.0

        if denominator <= 0:
            return 0.0

        value = (
            numerator
            / denominator
        )

        if not isfinite(
            value
        ):
            return 0.0

        return value