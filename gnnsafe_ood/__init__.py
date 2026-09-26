"""
GNNSafe OOD pipeline ported from GraphOOD-GNNSafe.

Energy-based OOD detection for graph neural networks.
Supports backbones: gcn, dssgnn (DSS-GNN).
"""

from .dataset import load_dataset
from .gnnsafe import GNNSafe
from .baselines import GraphEBM, MSP

__all__ = ["load_dataset", "GNNSafe", "MSP", "GraphEBM"]
