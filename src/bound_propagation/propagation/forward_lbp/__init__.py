from __future__ import annotations

from ...ir import OperationType
from .add import ForwardLBPAddStrategy
from .div import ForwardLBPDivStrategy
from .exp import ForwardLBPExpStrategy
from .flatten import ForwardLBPFlattenStrategy
from .linear import ForwardLBPLinearStrategy
from .log import ForwardLBPLogStrategy
from .matmul import ForwardLBPMatmulStrategy
from .mul import ForwardLBPMulStrategy
from .registry import ForwardLBPStrategyRegistry
from .relu import ForwardLBPReluStrategy
from .reshape import ForwardLBPReshapeStrategy
from .sigmoid import ForwardLBPSigmoidStrategy
from .sub import ForwardLBPSubStrategy
from .tanh import ForwardLBPTanhStrategy


def _register_forward_lbp_strategies():
    """Register all ForwardLBP strategies with the default forward LBP registry."""
    # Arithmetic operations
    ForwardLBPStrategyRegistry.register_default(OperationType.ADD, ForwardLBPAddStrategy())
    ForwardLBPStrategyRegistry.register_default(OperationType.SUB, ForwardLBPSubStrategy())
    ForwardLBPStrategyRegistry.register_default(OperationType.MUL, ForwardLBPMulStrategy())
    ForwardLBPStrategyRegistry.register_default(OperationType.DIV, ForwardLBPDivStrategy())

    # Activation functions
    ForwardLBPStrategyRegistry.register_default(OperationType.RELU, ForwardLBPReluStrategy())
    ForwardLBPStrategyRegistry.register_default(OperationType.SIGMOID, ForwardLBPSigmoidStrategy())
    ForwardLBPStrategyRegistry.register_default(OperationType.TANH, ForwardLBPTanhStrategy())
    ForwardLBPStrategyRegistry.register_default(OperationType.EXP, ForwardLBPExpStrategy())
    ForwardLBPStrategyRegistry.register_default(OperationType.LOG, ForwardLBPLogStrategy())

    # Linear operations
    ForwardLBPStrategyRegistry.register_default(OperationType.LINEAR, ForwardLBPLinearStrategy())
    ForwardLBPStrategyRegistry.register_default(OperationType.MATMUL, ForwardLBPMatmulStrategy())

    # Reshape operations
    ForwardLBPStrategyRegistry.register_default(OperationType.RESHAPE, ForwardLBPReshapeStrategy())
    ForwardLBPStrategyRegistry.register_default(OperationType.FLATTEN, ForwardLBPFlattenStrategy())


# Auto-register strategies on module import
_register_forward_lbp_strategies()
