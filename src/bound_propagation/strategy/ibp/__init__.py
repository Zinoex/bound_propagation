"""
Interval Bound Propagation (IBP) strategies.

IBP computes output bounds using interval arithmetic - the simplest and
most efficient bounding method. It propagates IntervalBounds through
the network using conservative interval operations.

Each operation has its own strategy class that is automatically registered
with the global registry.
"""

from ...ir import OperationType
from ..registry import get_global_registry

# Import all strategy classes
from .add import IBPAddStrategy
from .div import IBPDivStrategy
from .exp import IBPExpStrategy
from .flatten import IBPFlattenStrategy
from .linear import IBPLinearStrategy
from .log import IBPLogStrategy
from .matmul import IBPMatmulStrategy
from .mul import IBPMulStrategy
from .relu import IBPReluStrategy
from .reshape import IBPReshapeStrategy
from .sigmoid import IBPSigmoidStrategy
from .sub import IBPSubStrategy
from .tanh import IBPTanhStrategy

__all__ = [
    "IBPAddStrategy",
    "IBPSubStrategy",
    "IBPMulStrategy",
    "IBPDivStrategy",
    "IBPReluStrategy",
    "IBPSigmoidStrategy",
    "IBPTanhStrategy",
    "IBPExpStrategy",
    "IBPLogStrategy",
    "IBPLinearStrategy",
    "IBPMatmulStrategy",
    "IBPReshapeStrategy",
    "IBPFlattenStrategy",
]


# Auto-register all IBP strategies with the global registry
def _register_ibp_strategies() -> None:
    """Register all IBP strategies with the global registry."""
    registry = get_global_registry()

    # Arithmetic operations
    registry.register(OperationType.ADD, "ibp", IBPAddStrategy())
    registry.register(OperationType.SUB, "ibp", IBPSubStrategy())
    registry.register(OperationType.MUL, "ibp", IBPMulStrategy())
    registry.register(OperationType.DIV, "ibp", IBPDivStrategy())

    # Activation functions
    registry.register(OperationType.RELU, "ibp", IBPReluStrategy())
    registry.register(OperationType.SIGMOID, "ibp", IBPSigmoidStrategy())
    registry.register(OperationType.TANH, "ibp", IBPTanhStrategy())
    registry.register(OperationType.EXP, "ibp", IBPExpStrategy())
    registry.register(OperationType.LOG, "ibp", IBPLogStrategy())

    # Linear operations
    registry.register(OperationType.LINEAR, "ibp", IBPLinearStrategy())
    registry.register(OperationType.MATMUL, "ibp", IBPMatmulStrategy())

    # Reshaping operations
    registry.register(OperationType.RESHAPE, "ibp", IBPReshapeStrategy())
    registry.register(OperationType.FLATTEN, "ibp", IBPFlattenStrategy())


# Register strategies on module import
_register_ibp_strategies()
