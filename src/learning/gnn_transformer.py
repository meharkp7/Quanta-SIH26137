"""Step 13 causal edge-aware GNN--Transformer forecaster.

The forecaster operates on Step-12 tensors ``[B, L, E, F]``.

Road connectivity is represented sparsely.  A message is sent from source
road ``u`` to target road ``v`` iff ``u.to_node == v.from_node``.  This keeps
the Step-13 directed edge-aware GNN semantics while avoiding dense ``E x E``
adjacency and ``E x E x D`` message tensors, which are infeasible on real
road networks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.contracts.scenario import Scenario
from src.learning.loader import ForecastBatch


@dataclass(frozen=True)
class GraphBatch:
    """Directed road graph aligned to the batch edge order.

    ``adjacency`` is retained as a backwards-compatible constructor field for
    the existing Step-13 tests and small callers. Production graph batches
    created by :func:`build_graph_batch` leave it as ``None`` and use the
    sparse ``edge_index``/``edge_weight`` representation instead.

    ``edge_index`` has shape ``[2, N]``:
      row 0 = source edge position
      row 1 = target edge position.
    """

    adjacency: Tensor | None
    static_features: Tensor
    edge_index: Tensor | None = None
    edge_weight: Tensor | None = None
    edge_counts: tuple[int, ...] = ()


def _road_class_value(road_class: object) -> float:
    return {
        "local": 0.0,
        "collector": 0.5,
        "arterial": 1.0,
    }.get(str(road_class).split(".")[-1].lower(), 0.0)


def build_graph_batch(
    batch: ForecastBatch,
    scenarios: Mapping[str, Scenario],
    *,
    device=None,
) -> GraphBatch:
    """Build the sparse directed road graph in batch edge order.

    The graph is built once per unique scenario represented by the batch.
    This is O(E + C), where C is the number of actual legal edge-to-edge
    connections, rather than O(E^2).
    """
    b, _, e, _ = batch.features.shape
    static = np.zeros((b, e, 6), dtype=np.float32)

    # A single sparse edge list can only be shared when all samples in the
    # batch have the same topology. The training loop therefore batches
    # windows by scenario. We still support mixed scenarios by constructing
    # a block-diagonal sparse graph with per-sample offsets.
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    edge_counts: list[int] = []

    offset = 0

    for bi, scenario_id in enumerate(batch.scenario_ids):
        scenario = scenarios[scenario_id]
        edges = {str(edge.edge_id): edge for edge in scenario.edges}
        nodes = {str(node.node_id): node for node in scenario.nodes}

        raw_ids = list(batch.edge_ids_by_sample[bi])
        valid_positions: dict[str, int] = {}

        for position, raw_id in enumerate(raw_ids):
            edge_id = str(raw_id)
            if not edge_id:
                continue

            if edge_id not in edges:
                raise KeyError(
                    f"Scenario {scenario_id!r} does not contain edge "
                    f"{edge_id!r} referenced by the forecast batch."
                )

            valid_positions[edge_id] = position
            edge = edges[edge_id]
            head = nodes[str(edge.from_node)]
            tail = nodes[str(edge.to_node)]

            static[bi, position] = (
                float(edge.length_m) / 100.0,
                float(edge.speed_limit_mps) / 10.0,
                float(edge.lane_count),
                _road_class_value(edge.road_class),
                (float(tail.x_m) - float(head.x_m)) / 100.0,
                (float(tail.y_m) - float(head.y_m)) / 100.0,
            )

        outgoing: dict[str, list[str]] = {}
        for edge_id in valid_positions:
            edge = edges[edge_id]
            outgoing.setdefault(str(edge.from_node), []).append(edge_id)

        sources: list[int] = []
        targets: list[int] = []

        for target_id, target_position in valid_positions.items():
            target = edges[target_id]
            # source.to_node == target.from_node.
            # Indexing outgoing by target.from_node gives exactly those
            # source edges without scanning every edge pair.
            for source_id in outgoing.get(str(target.from_node), ()):
                if source_id == target_id:
                    continue
                sources.append(offset + valid_positions[source_id])
                targets.append(offset + target_position)

            # Explicit self information / isolated-edge support.
            sources.append(offset + target_position)
            targets.append(offset + target_position)

        if sources:
            source_arr = np.asarray(sources, dtype=np.int64)
            target_arr = np.asarray(targets, dtype=np.int64)

            # Row-normalization by target edge.
            degree = np.bincount(
                target_arr - offset,
                minlength=e,
            ).astype(np.float32)

            weights = 1.0 / np.maximum(
                degree[target_arr - offset],
                1.0,
            )

            source_parts.append(source_arr)
            target_parts.append(target_arr)
            weight_parts.append(weights.astype(np.float32))

        edge_counts.append(len(valid_positions))
        offset += e

    if source_parts:
        source = np.concatenate(source_parts)
        target = np.concatenate(target_parts)
        weight = np.concatenate(weight_parts)
    else:
        source = np.empty(0, dtype=np.int64)
        target = np.empty(0, dtype=np.int64)
        weight = np.empty(0, dtype=np.float32)

    edge_index = torch.as_tensor(
        np.stack((source, target), axis=0),
        dtype=torch.long,
        device=device,
    )

    return GraphBatch(
        adjacency=None,
        static_features=torch.as_tensor(
            static,
            dtype=torch.float32,
            device=device,
        ),
        edge_index=edge_index,
        edge_weight=torch.as_tensor(
            weight,
            dtype=torch.float32,
            device=device,
        ),
        edge_counts=tuple(edge_counts),
    )


class EdgeAwareGraphLayer(nn.Module):
    """Directed edge-to-edge message passing using only real neighbours."""

    def __init__(
        self,
        width: int,
        static_width: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(width + static_width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.update = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        x: Tensor,
        graph: GraphBatch,
    ) -> Tensor:
        """Apply sparse directed message passing.

        ``x`` is flattened as ``[B*E, D]`` while graph indices are already
        block-offset for each batch sample.
        """
        total_edges = x.shape[0]

        # Production path: sparse graph supplied by build_graph_batch().
        source_index = graph.edge_index
        edge_weight = graph.edge_weight

        # Backwards-compatible path for the existing Step-13 tests and small
        # external callers that still construct GraphBatch(adjacency, static).
        # Conversion is intentionally local and only applies to the legacy
        # dense input; production Delhi batches never create this tensor.
        if source_index is None or edge_weight is None:
            if graph.adjacency is None:
                raise ValueError(
                    "GraphBatch must provide either sparse edge_index/"
                    "edge_weight or a legacy adjacency tensor."
                )

            adjacency = graph.adjacency
            if adjacency.ndim != 3:
                raise ValueError(
                    "Legacy GraphBatch adjacency must have shape [B,E,E]."
                )

            b, e, _ = adjacency.shape

            # adjacency[b, target, source].
            # Flatten first so the batch dimension can be converted into the
            # block offset used by the flattened [B*E, D] representation.
            flat = torch.nonzero(
                adjacency.reshape(-1) > 0,
                as_tuple=False,
            ).flatten()

            block = flat // (e * e)
            local = flat % (e * e)
            target = local // e
            source = local % e

            source_index = torch.stack(
                (
                    block * e + source,
                    block * e + target,
                ),
                dim=0,
            )

            edge_weight = adjacency.reshape(-1).index_select(
                0,
                flat,
            ).float()

            degree = torch.zeros(
                b * e,
                dtype=edge_weight.dtype,
                device=adjacency.device,
            )
            degree.index_add_(
                0,
                source_index[1],
                edge_weight,
            )
            edge_weight = edge_weight / degree.index_select(
                0,
                source_index[1],
            ).clamp_min(1.0)

        source_index = source_index.to(device=x.device)
        edge_weight = edge_weight.to(device=x.device)

        if source_index.numel() == 0:
            return self.norm(x)

        source_x = x.index_select(0, source_index[0])

        # static_features is [B,E,S]. Flatten it using the same block order
        # as the graph offsets.
        static = graph.static_features.reshape(
            total_edges,
            graph.static_features.shape[-1],
        )
        target_static = static.index_select(0, source_index[1])

        messages = self.message(
            torch.cat((source_x, target_static), dim=-1)
        )

        weighted = messages * edge_weight.unsqueeze(-1)

        aggregate = torch.zeros_like(x)
        aggregate.index_add_(0, source_index[1], weighted)

        return self.norm(
            x
            + self.update(
                torch.cat((x, aggregate), dim=-1)
            )
        )


class CausalGNNTransformer(nn.Module):
    """Two directed graph layers followed by a per-road temporal Transformer."""

    def __init__(
        self,
        feature_count: int = 6,
        horizons: int = 3,
        width: int = 32,
        static_width: int = 6,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        max_history: int = 12,
    ):
        super().__init__()

        if width % heads:
            raise ValueError("width must be divisible by heads")
        if max_history <= 0:
            raise ValueError("max_history must be positive")

        self.config = dict(
            feature_count=feature_count,
            horizons=horizons,
            width=width,
            static_width=static_width,
            heads=heads,
            layers=layers,
            dropout=dropout,
            max_history=max_history,
        )

        self.input_projection = nn.Linear(
            feature_count + static_width,
            width,
        )

        self.graph_layers = nn.ModuleList(
            [
                EdgeAwareGraphLayer(width, static_width, dropout)
                for _ in range(2)
            ]
        )

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )

        self.position = nn.Parameter(
            torch.zeros(1, max_history, width)
        )
        nn.init.normal_(self.position, std=0.02)

        self.speed_head = nn.Linear(width, horizons)
        self.traversal_head = nn.Linear(width, horizons)
        self.accessibility_head = nn.Linear(width, horizons)

    def forward(
        self,
        features: Tensor,
        graph: GraphBatch,
        edge_padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        b, l, e, _ = features.shape

        if l > self.position.shape[1]:
            raise ValueError(
                f"History length {l} exceeds configured max_history "
                f"{self.position.shape[1]}."
            )

        static = graph.static_features.unsqueeze(1).expand(
            -1, l, -1, -1
        )

        h = self.input_projection(
            torch.cat((features, static), dim=-1)
        )

        spatial = []
        for t in range(l):
            step = h[:, t].reshape(b * e, -1)
            for layer in self.graph_layers:
                step = layer(step, graph)
            spatial.append(step.reshape(b, e, -1))

        h = torch.stack(spatial, dim=1)

        # One temporal sequence per road.
        h = h.permute(0, 2, 1, 3).reshape(
            b * e,
            l,
            -1,
        )

        h = self.temporal(h + self.position[:, :l])

        last = h[:, -1].reshape(b, e, -1)

        if edge_padding_mask is not None:
            last = last.masked_fill(
                edge_padding_mask.unsqueeze(-1),
                0.0,
            )

        return {
            "speed_ratio": self.speed_head(last),
            "traversal_time_s": F.softplus(
                self.traversal_head(last)
            ),
            "accessibility_logits": self.accessibility_head(last),
            "road_embeddings": last,
        }


def masked_huber(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    delta: float = 1.0,
) -> Tensor:
    valid = (
        mask.bool()
        & torch.isfinite(target)
        & torch.isfinite(prediction)
    )
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.huber_loss(
        prediction[valid],
        target[valid],
        delta=delta,
    )


def forecast_loss(
    outputs,
    batch: ForecastBatch,
    *,
    device=None,
) -> dict[str, Tensor]:
    speed_target = torch.as_tensor(
        batch.speed_targets,
        dtype=torch.float32,
        device=device,
    )
    speed_mask = torch.as_tensor(
        batch.speed_target_mask,
        dtype=torch.bool,
        device=device,
    )
    traversal_target = torch.as_tensor(
        batch.traversal_targets,
        dtype=torch.float32,
        device=device,
    )
    traversal_mask = torch.as_tensor(
        batch.traversal_target_mask,
        dtype=torch.bool,
        device=device,
    )

    speed = masked_huber(
        outputs["speed_ratio"],
        speed_target,
        speed_mask,
    )
    traversal = masked_huber(
        outputs["traversal_time_s"],
        traversal_target,
        traversal_mask,
    )

    return {
        "speed": speed,
        "traversal": traversal,
        "total": speed + traversal,
    }
