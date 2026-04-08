"""
Intermediate Representation (IR) for computation graphs.

This module provides the core data structures for representing computation graphs
that can be analyzed with bound propagation methods.
"""

from .graph import Graph
from .metadata import DeviceType, TensorMetadata
from .node import Node, NodeType
from .operations import OperationType

__all__ = [
    "TensorMetadata",
    "DeviceType",
    "Node",
    "NodeType",
    "Graph",
    "OperationType",
]
