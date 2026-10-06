"""
Fusion package for Graph-Aware Mixture-of-Experts.
"""
from .fusion import (
    AdaptiveFusion,
    HierarchicalFusion,
    DynamicFusion,
    ModalityDropout,
)

__all__ = [
    "AdaptiveFusion",
    "HierarchicalFusion",
    "DynamicFusion",
    "ModalityDropout",
]