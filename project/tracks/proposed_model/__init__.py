"""
Proposed Model package - Graph-Aware Mixture-of-Experts.
"""
from .model import GraphAwareMoE, create_graph_aware_moe, DEFAULT_CONFIG

__all__ = [
    "GraphAwareMoE",
    "create_graph_aware_moe",
    "DEFAULT_CONFIG",
]