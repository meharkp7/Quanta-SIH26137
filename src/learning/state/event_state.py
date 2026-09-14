"""
Step 16.5 — Causal event / incident representation.

Converts currently visible Observation events into a deterministic,
model-independent representation that can later be consumed by GNN,
temporal, and PPO components.

No future event is inferred.  In particular, an event is represented only
after its reveal time is visible in Observation.visible_events.  A future
effect_start_s is retained as announced schedule information; it is never
treated as an already-active incident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.contracts.observation import Observation
from src.contracts.scenario import Scenario
from .graph_state import GraphState


EVENT_FEATURE_NAMES: tuple[str, ...] = (
    "type_incident",
    "type_closure",
    "type_scheduled_closure",
    "type_other",
    "time_since_reveal_normalized",
    "effect_delay_normalized",
    "effect_started",
    "affected_edge_fraction",
    "affected_zone_fraction",
    "affected_parent_road_count_normalized",
)

_EVENT_MASK_NAMES: tuple[str, ...] = (
    "event_visible",
    "has_future_effect_start",
)


@dataclass(frozen=True)
class EventState:
    """Immutable causal event representation for one observation."""

    event_ids: tuple[str, ...]
    event_features: np.ndarray
    event_feature_names: tuple[str, ...]
    event_mask: np.ndarray
    event_edge_index: np.ndarray
    affected_edge_mask: np.ndarray
    affected_parent_road_ids: tuple[tuple[str, ...], ...]
    event_id_to_index: Mapping[str, int]

    def __post_init__(self) -> None:
        event_count = len(self.event_ids)
        feature_count = len(self.event_feature_names)

        if self.event_features.shape != (event_count, feature_count):
            raise ValueError(
                f"event_features shape {self.event_features.shape}; "
                f"expected {(event_count, feature_count)}"
            )

        if self.event_mask.shape != (event_count, len(_EVENT_MASK_NAMES)):
            raise ValueError(
                f"event_mask shape {self.event_mask.shape}; "
                f"expected {(event_count, len(_EVENT_MASK_NAMES))}"
            )

        if self.event_edge_index.shape[0] != 2:
            raise ValueError("event_edge_index must have shape [2, E]")
        if self.event_edge_index.shape[1] != event_count:
            raise ValueError("event_edge_index must contain one column per event")

        if self.affected_edge_mask.ndim != 2 or self.affected_edge_mask.shape[0] != event_count:
            raise ValueError(
                "affected_edge_mask must have shape [event_count, edge_count]"
            )

        if len(self.affected_parent_road_ids) != event_count:
            raise ValueError("affected_parent_road_ids must align with event_ids")

        if not np.all(np.isfinite(self.event_features)):
            raise ValueError("event_features contains NaN or infinite values")
        if not np.all(np.isfinite(self.event_mask)):
            raise ValueError("event_mask contains NaN or infinite values")

        expected_lookup = {event_id: i for i, event_id in enumerate(self.event_ids)}
        if dict(self.event_id_to_index) != expected_lookup:
            raise ValueError("event_id_to_index is inconsistent with event_ids")

        if self.affected_edge_mask.shape[1] and (
            np.any(self.affected_edge_mask < 0)
            or np.any(self.affected_edge_mask > 1)
        ):
            raise ValueError("affected_edge_mask must be binary")

        self.event_features.setflags(write=False)
        self.event_mask.setflags(write=False)
        self.event_edge_index.setflags(write=False)
        self.affected_edge_mask.setflags(write=False)

    @property
    def event_count(self) -> int:
        return len(self.event_ids)

    @property
    def feature_count(self) -> int:
        return len(self.event_feature_names)


class EventStateBuilder:
    """Build event features using only Scenario + currently visible Observation."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        time_normalization_s: float = 300.0,
        max_parent_road_count: float | None = None,
    ) -> None:
        if time_normalization_s <= 0:
            raise ValueError("time_normalization_s must be positive")
        if max_parent_road_count is not None and max_parent_road_count <= 0:
            raise ValueError("max_parent_road_count must be positive")

        self.scenario = scenario
        self.time_normalization_s = float(time_normalization_s)

        self._edges = tuple(
            sorted(scenario.edges, key=lambda e: str(e.edge_id))
        )
        self._edge_ids = tuple(str(e.edge_id) for e in self._edges)
        self._edge_index = {edge_id: i for i, edge_id in enumerate(self._edge_ids)}

        self._nodes = {
            str(node.node_id): node for node in scenario.nodes
        }
        self._parent_roads = {
            str(edge.parent_road_id) for edge in self._edges
        }
        self._max_parent_road_count = float(
            max_parent_road_count
            if max_parent_road_count is not None
            else max(len(self._parent_roads), 1)
        )

    def build(self, observation: Observation, graph_state: GraphState) -> EventState:
        self._validate(observation, graph_state)

        events = tuple(
            sorted(
                observation.visible_events,
                key=lambda event: (float(event.revealed_at_s), str(event.event_id)),
            )
        )

        count = len(events)
        features = np.zeros(
            (count, len(EVENT_FEATURE_NAMES)), dtype=np.float32
        )
        mask = np.zeros(
            (count, len(_EVENT_MASK_NAMES)), dtype=np.float32
        )
        affected_edges = np.zeros(
            (count, len(self._edge_ids)), dtype=np.float32
        )
        event_edge_index = np.zeros((2, count), dtype=np.int64)
        affected_roads: list[tuple[str, ...]] = []

        for i, event in enumerate(events):
            event_id = str(event.event_id)
            parent_roads = tuple(sorted(set(str(x) for x in event.affected_parent_road_ids)))
            affected_roads.append(parent_roads)

            edge_ids = [
                edge_id
                for edge_id in self._edge_ids
                if str(self._edges[self._edge_index[edge_id]].parent_road_id) in set(parent_roads)
            ]
            edge_indices = [self._edge_index[eid] for eid in edge_ids]
            if edge_indices:
                affected_edges[i, edge_indices] = 1.0

            event_edge_index[0, i] = (
                edge_indices[0] if edge_indices else 0
            )
            event_edge_index[1, i] = (
                edge_indices[-1] if edge_indices else 0
            )

            type_value = str(getattr(event.event_type, "value", event.event_type)).lower()
            type_column = {
                "incident": "type_incident",
                "closure": "type_closure",
                "scheduled_closure": "type_scheduled_closure",
            }.get(type_value, "type_other")
            features[i, EVENT_FEATURE_NAMES.index(type_column)] = 1.0

            now = float(observation.observation_time_s)
            reveal = float(event.revealed_at_s)
            features[i, EVENT_FEATURE_NAMES.index("time_since_reveal_normalized")] = np.clip(
                max(0.0, now - reveal) / self.time_normalization_s, 0.0, 10.0
            )

            has_future_effect = event.effect_start_s is not None and float(event.effect_start_s) > now
            mask[i, 0] = 1.0
            mask[i, 1] = float(has_future_effect)

            if event.effect_start_s is not None:
                effect_start = float(event.effect_start_s)
                features[i, EVENT_FEATURE_NAMES.index("effect_delay_normalized")] = np.clip(
                    max(0.0, effect_start - now) / self.time_normalization_s, 0.0, 10.0
                )
                features[i, EVENT_FEATURE_NAMES.index("effect_started")] = float(effect_start <= now)

            features[i, EVENT_FEATURE_NAMES.index("affected_edge_fraction")] = (
                len(edge_indices) / max(len(self._edge_ids), 1)
            )

            affected_zones = {
                str(self._nodes[str(edge.from_node)].zone_id)
                for edge in self._edges
                if str(edge.parent_road_id) in set(parent_roads)
            }
            total_zones = max(
                len({str(node.zone_id) for node in self._nodes.values()}), 1
            )
            features[i, EVENT_FEATURE_NAMES.index("affected_zone_fraction")] = (
                len(affected_zones) / total_zones
            )

            features[i, EVENT_FEATURE_NAMES.index("affected_parent_road_count_normalized")] = np.clip(
                len(parent_roads) / self._max_parent_road_count, 0.0, 1.0
            )

        return EventState(
            event_ids=tuple(str(e.event_id) for e in events),
            event_features=features,
            event_feature_names=EVENT_FEATURE_NAMES,
            event_mask=mask,
            event_edge_index=event_edge_index,
            affected_edge_mask=affected_edges,
            affected_parent_road_ids=tuple(affected_roads),
            event_id_to_index={
                str(event.event_id): i for i, event in enumerate(events)
            },
        )

    def _validate(self, observation: Observation, graph_state: GraphState) -> None:
        if str(observation.scenario_id) != str(self.scenario.scenario_id):
            raise ValueError("observation scenario_id does not match Scenario")
        if str(observation.graph_version) != str(self.scenario.graph_version):
            raise ValueError("observation graph_version does not match Scenario")
        if graph_state.scenario_id != str(self.scenario.scenario_id):
            raise ValueError("GraphState scenario_id does not match Scenario")
        if graph_state.graph_version != str(self.scenario.graph_version):
            raise ValueError("GraphState graph_version does not match Scenario")
        if graph_state.edge_ids != self._edge_ids:
            raise ValueError("GraphState edge ordering does not match Scenario")
        event_ids = [str(e.event_id) for e in observation.visible_events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate visible event_id")
        for event in observation.visible_events:
            if float(event.revealed_at_s) > float(observation.observation_time_s):
                raise ValueError(
                    f"event {event.event_id!r} is not causally visible at observation time"
                )
