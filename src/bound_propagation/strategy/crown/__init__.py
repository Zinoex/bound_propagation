"""
Forward-mode CROWN (Convex Relaxation Of Neural Networks) strategies.

CROWN propagates linear (affine) bounds through the network, computing
tighter relaxations than IBP for non-linear operations.

This module implements forward-mode bound propagation using linear relaxations.
Each operation has its own strategy that computes affine bounds.
"""

from __future__ import annotations

from ...ir import OperationType
from ..registry import get_global_registry
from .add import CROWNAddStrategy
from .div import CROWNDivStrategy
from .exp import CROWNExpStrategy
from .flatten import CROWNFlattenStrategy
from .linear import CROWNLinearStrategy
from .log import CROWNLogStrategy
from .matmul import CROWNMatmulStrategy
from .mul import CROWNMulStrategy
from .relu import CROWNReluStrategy
from .reshape import CROWNReshapeStrategy
from .sigmoid import CROWNSigmoidStrategy
from .sub import CROWNSubStrategy
from .tanh import CROWNTanhStrategy

__all__ = [
    "CROWNAddStrategy",
    "CROWNSubStrategy",
    "CROWNMulStrategy",
    "CROWNDivStrategy",
    "CROWNReluStrategy",
    "CROWNSigmoidStrategy",
    "CROWNTanhStrategy",
    "CROWNExpStrategy",
    "CROWNLogStrategy",
    "CROWNLinearStrategy",
    "CROWNMatmulStrategy",
    "CROWNReshapeStrategy",
    "CROWNFlattenStrategy",
]


def _register_crown_strategies():
    """Register all CROWN strategies with the global registry."""
    registry = get_global_registry()

    # Arithmetic operations
    registry.register(OperationType.ADD, "crown", CROWNAddStrategy())
    registry.register(OperationType.SUB, "crown", CROWNSubStrategy())
    registry.register(OperationType.MUL, "crown", CROWNMulStrategy())
    registry.register(OperationType.DIV, "crown", CROWNDivStrategy())

    # Activation functions
    registry.register(OperationType.RELU, "crown", CROWNReluStrategy())
    registry.register(OperationType.SIGMOID, "crown", CROWNSigmoidStrategy())
    registry.register(OperationType.TANH, "crown", CROWNTanhStrategy())
    registry.register(OperationType.EXP, "crown", CROWNExpStrategy())
    registry.register(OperationType.LOG, "crown", CROWNLogStrategy())

    # Linear operations
    registry.register(OperationType.LINEAR, "crown", CROWNLinearStrategy())
    registry.register(OperationType.MATMUL, "crown", CROWNMatmulStrategy())

    # Reshape operations
    registry.register(OperationType.RESHAPE, "crown", CROWNReshapeStrategy())
    registry.register(OperationType.FLATTEN, "crown", CROWNFlattenStrategy())


# Auto-register strategies on module import
_register_crown_strategies()
