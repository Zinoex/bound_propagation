from ...ir import OperationType
from .add import IBPAddStrategy
from .div import IBPDivStrategy
from .exp import IBPExpStrategy
from .flatten import IBPFlattenStrategy
from .linear import IBPLinearStrategy
from .log import IBPLogStrategy
from .matmul import IBPMatmulStrategy
from .mul import IBPMulStrategy
from .registry import ForwardIBPStrategyRegistry
from .relu import IBPReluStrategy
from .reshape import IBPReshapeStrategy
from .sigmoid import IBPSigmoidStrategy
from .sub import IBPSubStrategy
from .tanh import IBPTanhStrategy


def _register_ibp_strategies() -> None:
    """Register all IBP strategies with the default IBP strategy registry."""

    # Arithmetic operations
    ForwardIBPStrategyRegistry.register_default(OperationType.ADD, IBPAddStrategy())
    ForwardIBPStrategyRegistry.register_default(OperationType.SUB, IBPSubStrategy())
    ForwardIBPStrategyRegistry.register_default(OperationType.MUL, IBPMulStrategy())
    ForwardIBPStrategyRegistry.register_default(OperationType.DIV, IBPDivStrategy())

    # Activation functions
    ForwardIBPStrategyRegistry.register_default(OperationType.RELU, IBPReluStrategy())
    ForwardIBPStrategyRegistry.register_default(OperationType.SIGMOID, IBPSigmoidStrategy())
    ForwardIBPStrategyRegistry.register_default(OperationType.TANH, IBPTanhStrategy())
    ForwardIBPStrategyRegistry.register_default(OperationType.EXP, IBPExpStrategy())
    ForwardIBPStrategyRegistry.register_default(OperationType.LOG, IBPLogStrategy())

    # Linear operations
    ForwardIBPStrategyRegistry.register_default(OperationType.LINEAR, IBPLinearStrategy())
    ForwardIBPStrategyRegistry.register_default(OperationType.MATMUL, IBPMatmulStrategy())

    # Reshaping operations
    ForwardIBPStrategyRegistry.register_default(OperationType.RESHAPE, IBPReshapeStrategy())
    ForwardIBPStrategyRegistry.register_default(OperationType.FLATTEN, IBPFlattenStrategy())


# Register strategies on module import
_register_ibp_strategies()
