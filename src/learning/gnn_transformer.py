"""Step 13 causal edge-aware GNN--Transformer forecaster.

Inputs are the Step 12 tensors ``[B, L, E, F]``.  Roads remain separate all
the way to the forecast heads; only the temporal axis is flattened for the
Transformer.  The graph is directed: a message travels from edge ``u`` to
edge ``v`` only when the legal road endpoint of ``u`` reaches the start of
``v``.
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
    adjacency: Tensor  # [B, E, E], row=target edge, col=source edge
    static_features: Tensor  # [B, E, S]


def build_graph_batch(batch: ForecastBatch, scenarios: Mapping[str, Scenario], *, device=None) -> GraphBatch:
    """Build directed road adjacency/static features in batch edge order."""
    b, _, e, _ = batch.features.shape
    adjacency = np.zeros((b, e, e), dtype=np.float32)
    static = np.zeros((b, e, 6), dtype=np.float32)
    for bi, scenario_id in enumerate(batch.scenario_ids):
        scenario = scenarios[scenario_id]
        edges = {edge.edge_id: edge for edge in scenario.edges}
        nodes = {node.node_id: node for node in scenario.nodes}
        ids = [str(x) for x in batch.edge_ids_by_sample[bi] if str(x)]
        for i, edge_id in enumerate(ids):
            edge = edges[edge_id]
            tail = nodes[edge.to_node]
            head = nodes[edge.from_node]
            road_class = {"local": 0.0, "collector": 0.5, "arterial": 1.0}.get(
                str(edge.road_class).split(".")[-1].lower(), 0.0
            )
            static[bi, i] = (
                float(edge.length_m) / 100.0,
                float(edge.speed_limit_mps) / 10.0,
                float(edge.lane_count),
                road_class,
                (float(tail.x_m) - float(head.x_m)) / 100.0,
                (float(tail.y_m) - float(head.y_m)) / 100.0,
            )
        for target_i, target_id in enumerate(ids):
            target = edges[target_id]
            for source_i, source_id in enumerate(ids):
                source = edges[source_id]
                if source.to_node == target.from_node and source_id != target_id:
                    adjacency[bi, target_i, source_i] = 1.0
        # Self information is handled by the layer's residual path. Add a
        # self-loop for stable normalization and isolated roads.
        adjacency[bi, np.arange(len(ids)), np.arange(len(ids))] = 1.0
    degree = adjacency.sum(axis=-1, keepdims=True)
    adjacency = adjacency / np.maximum(degree, 1.0)
    return GraphBatch(
        adjacency=torch.as_tensor(adjacency, dtype=torch.float32, device=device),
        static_features=torch.as_tensor(static, dtype=torch.float32, device=device),
    )


class EdgeAwareGraphLayer(nn.Module):
    """Directed message passing with target-road static edge features."""
    def __init__(self, width: int, static_width: int, dropout: float = 0.1):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(width + static_width, width), nn.GELU(), nn.Linear(width, width)
        )
        self.update = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor, adjacency: Tensor, static_features: Tensor) -> Tensor:
        # x [B,E,D], adjacency [B,target,source]
        source = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        target_static = static_features.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        messages = self.message(torch.cat((source, target_static), dim=-1))
        aggregate = torch.einsum("bij,bijd->bid", adjacency, messages)
        return self.norm(x + self.update(torch.cat((x, aggregate), dim=-1)))


class CausalGNNTransformer(nn.Module):
    """Two directed graph layers followed by a per-road temporal Transformer."""
    def __init__(self, feature_count: int = 6, horizons: int = 3, width: int = 32,
                 static_width: int = 6, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, max_history: int = 12):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.config = dict(feature_count=feature_count, horizons=horizons, width=width,
                           static_width=static_width, heads=heads, layers=layers,
                           dropout=dropout, max_history=max_history)
        self.input_projection = nn.Linear(feature_count + static_width, width)
        self.graph_layers = nn.ModuleList(
            [EdgeAwareGraphLayer(width, static_width, dropout) for _ in range(2)]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.position = nn.Parameter(torch.zeros(1, max_history, width))
        nn.init.normal_(self.position, std=0.02)
        self.speed_head = nn.Linear(width, horizons)
        self.traversal_head = nn.Linear(width, horizons)
        self.accessibility_head = nn.Linear(width, horizons)

    def forward(self, features: Tensor, graph: GraphBatch, edge_padding_mask: Tensor | None = None) -> dict[str, Tensor]:
        b, l, e, _ = features.shape
        static = graph.static_features.unsqueeze(1).expand(-1, l, -1, -1)
        h = self.input_projection(torch.cat((features, static), dim=-1))
        spatial = []
        for t in range(l):
            step = h[:, t]
            for layer in self.graph_layers:
                step = layer(step, graph.adjacency, graph.static_features)
            spatial.append(step)
        h = torch.stack(spatial, dim=1)  # [B,L,E,D]
        h = h.permute(0, 2, 1, 3).reshape(b * e, l, -1)
        h = self.temporal(h + self.position[:, :l])
        last = h[:, -1].reshape(b, e, -1)
        if edge_padding_mask is not None:
            last = last.masked_fill(edge_padding_mask.unsqueeze(-1), 0.0)
        return {
            "speed_ratio": self.speed_head(last),
            "traversal_time_s": F.softplus(self.traversal_head(last)),
            "accessibility_logits": self.accessibility_head(last),
            "road_embeddings": last,
        }


def masked_huber(prediction: Tensor, target: Tensor, mask: Tensor, delta: float = 1.0) -> Tensor:
    """Finite Huber loss over mature labels only."""
    valid = mask.bool() & torch.isfinite(target) & torch.isfinite(prediction)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return F.huber_loss(prediction[valid], target[valid], delta=delta)


def forecast_loss(outputs: Mapping[str, Tensor], batch: ForecastBatch, *, device=None) -> dict[str, Tensor]:
    speed_target = torch.as_tensor(batch.speed_targets, dtype=torch.float32, device=device)
    speed_mask = torch.as_tensor(batch.speed_target_mask, dtype=torch.bool, device=device)
    traversal_target = torch.as_tensor(batch.traversal_targets, dtype=torch.float32, device=device)
    traversal_mask = torch.as_tensor(batch.traversal_target_mask, dtype=torch.bool, device=device)
    speed = masked_huber(outputs["speed_ratio"], speed_target, speed_mask)
    traversal = masked_huber(outputs["traversal_time_s"], traversal_target, traversal_mask)
    return {"speed": speed, "traversal": traversal, "total": speed + traversal}
