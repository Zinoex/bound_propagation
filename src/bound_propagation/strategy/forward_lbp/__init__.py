"""
Forward-mode ForwardLBP (Convex Relaxation Of Neural Networks) strategies.

ForwardLBP propagates linear (affine) bounds through the network, computing
tighter relaxations than IBP for non-linear operations.

This module implements forward-mode bound propagation using linear relaxations.
Each operation has its own strategy that computes affine bounds.
"""

from __future__ import annotations

from ...ir import OperationType
from ..registry import get_global_registry
from .add import ForwardLBPAddStrategy
from .div import ForwardLBPDivStrategy
from .exp import ForwardLBPExpStrategy
from .flatten import ForwardLBPFlattenStrategy
from .linear import ForwardLBPLinearStrategy
from .log import ForwardLBPLogStrategy
from .matmul import ForwardLBPMatmulStrategy
from .mul import ForwardLBPMulStrategy
from .relu import ForwardLBPReluStrategy
from .reshape import ForwardLBPReshapeStrategy
from .sigmoid import ForwardLBPSigmoidStrategy
from .sub import ForwardLBPSubStrategy
from .tanh import ForwardLBPTanhStrategy

__all__ = [
    "ForwardLBPAddStrategy",
    "ForwardLBPSubStrategy",
    "ForwardLBPMulStrategy",
    "ForwardLBPDivStrategy",
    "ForwardLBPReluStrategy",
    "ForwardLBPSigmoidStrategy",
    "ForwardLBPTanhStrategy",
    "ForwardLBPExpStrategy",
    "ForwardLBPLogStrategy",
    "ForwardLBPLinearStrategy",
    "ForwardLBPMatmulStrategy",
    "ForwardLBPReshapeStrategy",
    "ForwardLBPFlattenStrategy",
]


def _register_forward_lbp_strategies():
    """Register all ForwardLBP strategies with the global registry."""
    registry = get_global_registry()

    # Arithmetic operations
    registry.register(OperationType.ADD, "forward", ForwardLBPAddStrategy())
    registry.register(OperationType.SUB, "forward", ForwardLBPSubStrategy())
    registry.register(OperationType.MUL, "forward", ForwardLBPMulStrategy())
    registry.register(OperationType.DIV, "forward", ForwardLBPDivStrategy())

    # Activation functions
    registry.register(OperationType.RELU, "forward", ForwardLBPReluStrategy())
    registry.register(OperationType.SIGMOID, "forward", ForwardLBPSigmoidStrategy())
    registry.register(OperationType.TANH, "forward", ForwardLBPTanhStrategy())
    registry.register(OperationType.EXP, "forward", ForwardLBPExpStrategy())
    registry.register(OperationType.LOG, "forward", ForwardLBPLogStrategy())

    # Linear operations
    registry.register(OperationType.LINEAR, "forward", ForwardLBPLinearStrategy())
    registry.register(OperationType.MATMUL, "forward", ForwardLBPMatmulStrategy())

    # Reshape operations
    registry.register(OperationType.RESHAPE, "forward", ForwardLBPReshapeStrategy())
    registry.register(OperationType.FLATTEN, "forward", ForwardLBPFlattenStrategy())


# Auto-register strategies on module import
_register_forward_lbp_strategies()
