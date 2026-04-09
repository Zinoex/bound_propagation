"""
Backward-mode Linear Bound Propagation (Backward CROWN) strategies.

Backward LBP propagates linear bounds backwards through the network in reverse
topological order, computing how the output depends on each intermediate node.

This module now uses the new backward propagation algorithm (Algorithm 2 from auto_LiRPA).
Use BackwardLBPPropagator instead of the regular BoundPropagator for backward mode.
"""

from __future__ import annotations

from ...ir import OperationType
from ..registry import get_global_registry
from .add_backward import BackwardAddStrategy
from .matmul_backward import BackwardMatmulStrategy
from .relu_backward import BackwardReluStrategy

__all__ = [
    "BackwardAddStrategy",
    "BackwardMatmulStrategy",
    "BackwardReluStrategy",
]


def _register_backward_strategies():
    """Register backward propagation strategies with the global registry."""
    registry = get_global_registry()

    # Arithmetic operations
    registry.register(OperationType.ADD, "backward", BackwardAddStrategy())
    
    # Activation functions
    registry.register(OperationType.RELU, "backward", BackwardReluStrategy())

    # Linear operations
    registry.register(OperationType.MATMUL, "backward", BackwardMatmulStrategy())


# Auto-register strategies on module import
_register_backward_strategies()
