"""
Forward-mode ForwardCrown (Convex Relaxation Of Neural Networks) strategies.

ForwardCrown propagates linear (affine) bounds through the network, computing
tighter relaxations than IBP for non-linear operations.

This module implements forward-mode bound propagation using linear relaxations.
Each operation has its own strategy that computes affine bounds.
"""

from __future__ import annotations

from ...ir import OperationType
from ..registry import get_global_registry
from .add import ForwardCrownAddStrategy
from .div import ForwardCrownDivStrategy
from .exp import ForwardCrownExpStrategy
from .flatten import ForwardCrownFlattenStrategy
from .linear import ForwardCrownLinearStrategy
from .log import ForwardCrownLogStrategy
from .matmul import ForwardCrownMatmulStrategy
from .mul import ForwardCrownMulStrategy
from .relu import ForwardCrownReluStrategy
from .reshape import ForwardCrownReshapeStrategy
from .sigmoid import ForwardCrownSigmoidStrategy
from .sub import ForwardCrownSubStrategy
from .tanh import ForwardCrownTanhStrategy

__all__ = [
    "ForwardCrownAddStrategy",
    "ForwardCrownSubStrategy",
    "ForwardCrownMulStrategy",
    "ForwardCrownDivStrategy",
    "ForwardCrownReluStrategy",
    "ForwardCrownSigmoidStrategy",
    "ForwardCrownTanhStrategy",
    "ForwardCrownExpStrategy",
    "ForwardCrownLogStrategy",
    "ForwardCrownLinearStrategy",
    "ForwardCrownMatmulStrategy",
    "ForwardCrownReshapeStrategy",
    "ForwardCrownFlattenStrategy",
]


def _register_forward_crown_strategies():
    """Register all ForwardCrown strategies with the global registry."""
    registry = get_global_registry()

    # Arithmetic operations
    registry.register(OperationType.ADD, "forward", ForwardCrownAddStrategy())
    registry.register(OperationType.SUB, "forward", ForwardCrownSubStrategy())
    registry.register(OperationType.MUL, "forward", ForwardCrownMulStrategy())
    registry.register(OperationType.DIV, "forward", ForwardCrownDivStrategy())

    # Activation functions
    registry.register(OperationType.RELU, "forward", ForwardCrownReluStrategy())
    registry.register(OperationType.SIGMOID, "forward", ForwardCrownSigmoidStrategy())
    registry.register(OperationType.TANH, "forward", ForwardCrownTanhStrategy())
    registry.register(OperationType.EXP, "forward", ForwardCrownExpStrategy())
    registry.register(OperationType.LOG, "forward", ForwardCrownLogStrategy())

    # Linear operations
    registry.register(OperationType.LINEAR, "forward", ForwardCrownLinearStrategy())
    registry.register(OperationType.MATMUL, "forward", ForwardCrownMatmulStrategy())

    # Reshape operations
    registry.register(OperationType.RESHAPE, "forward", ForwardCrownReshapeStrategy())
    registry.register(OperationType.FLATTEN, "forward", ForwardCrownFlattenStrategy())


# Auto-register strategies on module import
_register_forward_crown_strategies()
