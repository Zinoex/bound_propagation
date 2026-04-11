from ...ir import AbstractValueType, OperationType
from .add import IBPAdd, IBPAddWithConstant
from .base import ForwardIBPStrategy
from .div import IBPConstantDiv, IBPDiv, IBPDivConstant
from .exp import IBPExp
from .flatten import IBPFlatten
from .linear import IBPLinear
from .log import IBPLog
from .mul import IBPMul, IBPMulWithConstant
from .registry import ForwardIBPStrategyRegistry
from .relu import IBPRelu
from .reshape import IBPReshape
from .sigmoid import IBPSigmoid
from .sub import IBPSub
from .tanh import IBPTanh

__all__ = [
    "ForwardIBPStrategy",
    "ForwardIBPStrategyRegistry",
]


def _register_ibp_strategies() -> None:
    """Register all IBP strategies with the default IBP strategy registry."""

    # Arithmetic operations
    ForwardIBPStrategyRegistry.register_default(
        OperationType.ADD,
        IBPAdd(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.ADD,
        IBPAddWithConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.ADD,
        IBPAddWithConstant(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.SUB,
        IBPSub(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.SUB,
        IBPSub(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MUL,
        IBPMul(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MUL,
        IBPMulWithConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MUL,
        IBPMulWithConstant(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.DIV,
        IBPDiv(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.DIV,
        IBPDivConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.DIV,
        IBPConstantDiv(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )

    # Activation functions
    ForwardIBPStrategyRegistry.register_default(OperationType.RELU, IBPRelu(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.SIGMOID, IBPSigmoid(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.TANH, IBPTanh(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.EXP, IBPExp(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.LOG, IBPLog(), signature=(AbstractValueType.ABSTRACT,))

    # Linear operations
    ForwardIBPStrategyRegistry.register_default(OperationType.LINEAR, IBPLinear(), signature=(AbstractValueType.ABSTRACT,))

    # Reshaping operations
    ForwardIBPStrategyRegistry.register_default(OperationType.RESHAPE, IBPReshape(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.FLATTEN, IBPFlatten(), signature=(AbstractValueType.ABSTRACT,))


# Register strategies on module import
_register_ibp_strategies()
