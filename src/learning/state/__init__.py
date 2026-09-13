"""
Step-16 state representations.

This package contains causal, model-independent representations of the
traffic-routing environment.

The representations are intentionally separate from:
    - GNN architectures
    - Transformer architectures
    - PPO policies
    - QPSO
    - SUMO

Those components consume these state objects later.
"""

from .graph_state import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    GraphState,
    GraphStateBuilder,
)

from .state_fusion import (
    FUSED_GLOBAL_FEATURE_NAMES,
    FusedState,
    StateFusionBuilder,
)

__all__ = [
    "EDGE_FEATURE_NAMES",
    "NODE_FEATURE_NAMES",
    "GraphState",
    "GraphStateBuilder",
    "FUSED_GLOBAL_FEATURE_NAMES",
    "FusedState",
    "StateFusionBuilder",
]