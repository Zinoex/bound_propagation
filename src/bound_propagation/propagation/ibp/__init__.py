from ...ir import AbstractValueType, OperationType
from .abs import IBPAbs
from .add import IBPAdd, IBPAddWithConstant
from .base import ForwardIBPStrategy
from .cat import IBPCat
from .clamp import IBPClamp
from .cos import IBPCos
from .div import IBPConstantDiv, IBPDiv, IBPDivConstant
from .exp import IBPExp
from .flatten import IBPFlatten
from .getitem import IBPGetItem
from .linear import IBPLinear
from .log import IBPLog
from .matmul import IBPConstantMatmul, IBPMatmul, IBPMatmulConstant
from .max import IBPMax
from .maximum import IBPMaximum, IBPMaximumWithConstant
from .mean import IBPMean
from .min import IBPMin
from .minimum import IBPMinimum, IBPMinimumWithConstant
from .mul import IBPMul, IBPMulWithConstant
from .neg import IBPNeg
from .reciprocal import IBPReciprocal
from .registry import ForwardIBPStrategyRegistry
from .relu import IBPRelu
from .reshape import IBPReshape
from .select import IBPSelect
from .sigmoid import IBPSigmoid
from .sin import IBPSin
from .sqrt import IBPSqrt
from .squeeze import IBPSqueeze
from .stack import IBPStack
from .sub import IBPSub
from .sum import IBPSum
from .tanh import IBPTanh
from .transpose import IBPTranspose
from .unsqueeze import IBPUnsqueeze
from .view import IBPView

# TODO: Figure out how to support cbrt

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
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MAXIMUM, IBPMaximum(), signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT)
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MAXIMUM,
        IBPMaximumWithConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MAXIMUM,
        IBPMaximumWithConstant(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MINIMUM, IBPMinimum(), signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT)
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MINIMUM,
        IBPMinimumWithConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MINIMUM,
        IBPMinimumWithConstant(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )

    # Element-wise functions
    ForwardIBPStrategyRegistry.register_default(OperationType.RELU, IBPRelu(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(
        OperationType.SIGMOID, IBPSigmoid(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(OperationType.TANH, IBPTanh(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.EXP, IBPExp(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.LOG, IBPLog(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.SQRT, IBPSqrt(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(
        OperationType.RECIPROCAL, IBPReciprocal(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(OperationType.NEG, IBPNeg(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.ABS, IBPAbs(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(
        OperationType.CLAMP, IBPClamp(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(OperationType.COS, IBPCos(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.SIN, IBPSin(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.TAN, IBPTanh(), signature=(AbstractValueType.ABSTRACT,))

    # Linear operations
    ForwardIBPStrategyRegistry.register_default(
        OperationType.LINEAR, IBPLinear(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MATMUL,
        IBPMatmul(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MATMUL,
        IBPMatmulConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.MATMUL,
        IBPConstantMatmul(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )

    # Reductions
    ForwardIBPStrategyRegistry.register_default(OperationType.MEAN, IBPMean(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.SUM, IBPSum(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.MAX, IBPMax(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(OperationType.MIN, IBPMin(), signature=(AbstractValueType.ABSTRACT,))

    # Reshaping operations
    ForwardIBPStrategyRegistry.register_default(
        OperationType.RESHAPE, IBPReshape(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.FLATTEN, IBPFlatten(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(OperationType.CONCAT, IBPCat(), signature=(AbstractValueType.ABSTRACT,))
    ForwardIBPStrategyRegistry.register_default(
        OperationType.GETITEM, IBPGetItem(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.STACK, IBPStack(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.SELECT, IBPSelect(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.UNSQUEEZE, IBPUnsqueeze(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.SQUEEZE, IBPSqueeze(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.TRANSPOSE, IBPTranspose(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(
        OperationType.PERMUTE, IBPTranspose(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardIBPStrategyRegistry.register_default(OperationType.VIEW, IBPView(), signature=(AbstractValueType.ABSTRACT,))


# Register strategies on module import
_register_ibp_strategies()
