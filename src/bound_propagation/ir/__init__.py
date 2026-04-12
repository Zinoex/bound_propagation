"""
Intermediate Representation (IR) for computation graphs.

This module provides the core data structures for representing computation graphs
that can be analyzed with bound propagation methods.
"""

from .graph import Graph
from .metadata import TensorMetadata
from .node import AbstractValueType, Node, NodeType
from .operations import OperationType

__all__ = [
    "TensorMetadata",
    "Node",
    "NodeType",
    "AbstractValueType",
    "Graph",
    "OperationType",
]
