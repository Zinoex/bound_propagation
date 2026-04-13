from ...ir import AbstractValueType, OperationType
from .abs import ForwardLBPAbs
from .add import ForwardLBPAdd, ForwardLBPAddWithConstant
from .base import ForwardLBPStrategy
from .cat import ForwardLBPConcat
from .clamp import ForwardLBPClamp
from .cos import ForwardLBPCos
from .div import ForwardLBPConstantDiv, ForwardLBPDiv, ForwardLBPDivConstant
from .exp import ForwardLBPExp
from .flatten import ForwardLBPFlatten
from .getitem import ForwardLBPGetItem
from .linear import ForwardLBPLinear
from .log import ForwardLBPLog
from .matmul import ForwardLBPConstantMatmul, ForwardLBPMatmul, ForwardLBPMatmulConstant
from .max import ForwardLBPMax
from .maximum import ForwardLBPMaximum, ForwardLBPMaximumWithConstant
from .mean import ForwardLBPMean
from .min import ForwardLBPMin
from .minimum import ForwardLBPMinimum, ForwardLBPMinimumWithConstant
from .mul import ForwardLBPMul, ForwardLBPMulWithConstant
from .neg import ForwardLBPNeg
from .reciprocal import ForwardLBPReciprocal
from .registry import ForwardLBPStrategyRegistry
from .relu import ForwardLBPRelu
from .reshape import ForwardLBPReshape
from .select import ForwardLBPSelect
from .sigmoid import ForwardLBPSigmoid
from .sin import ForwardLBPSin
from .sqrt import ForwardLBPSqrt
from .squeeze import ForwardLBPSqueeze
from .stack import ForwardLBPStack
from .sub import ForwardLBPSub, ForwardLBPSubConstantLeft, ForwardLBPSubConstantRight
from .sum import ForwardLBPSum
from .tan import ForwardLBPTan
from .tanh import ForwardLBPTanh
from .transpose import ForwardLBPTranspose
from .unsqueeze import ForwardLBPUnsqueeze
from .view import ForwardLBPView

__all__ = ["ForwardLBPStrategy", "ForwardLBPStrategyRegistry"]


def _register_strategies():
    """Register all ForwardLBP strategies with signatures."""

    # Arithmetic operations
    ForwardLBPStrategyRegistry.register_default(
        OperationType.ADD,
        ForwardLBPAdd(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.ADD,
        ForwardLBPAddWithConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.ADD,
        ForwardLBPAddWithConstant(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )

    ForwardLBPStrategyRegistry.register_default(
        OperationType.SUB,
        ForwardLBPSub(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SUB,
        ForwardLBPSubConstantRight(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SUB,
        ForwardLBPSubConstantLeft(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )

    ForwardLBPStrategyRegistry.register_default(
        OperationType.MUL,
        ForwardLBPMul(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MUL,
        ForwardLBPMulWithConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MUL,
        ForwardLBPMulWithConstant(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )

    ForwardLBPStrategyRegistry.register_default(
        OperationType.DIV,
        ForwardLBPDiv(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.DIV,
        ForwardLBPDivConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.DIV,
        ForwardLBPConstantDiv(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )

    ForwardLBPStrategyRegistry.register_default(
        OperationType.MAXIMUM,
        ForwardLBPMaximum(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MAXIMUM,
        ForwardLBPMaximumWithConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MAXIMUM,
        ForwardLBPMaximumWithConstant(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )

    ForwardLBPStrategyRegistry.register_default(
        OperationType.MINIMUM,
        ForwardLBPMinimum(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MINIMUM,
        ForwardLBPMinimumWithConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MINIMUM,
        ForwardLBPMinimumWithConstant(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )

    # Element-wise functions
    ForwardLBPStrategyRegistry.register_default(
        OperationType.RELU, ForwardLBPRelu(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SIGMOID, ForwardLBPSigmoid(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.TANH, ForwardLBPTanh(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.EXP, ForwardLBPExp(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.LOG, ForwardLBPLog(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SQRT, ForwardLBPSqrt(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.RECIPROCAL, ForwardLBPReciprocal(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.NEG, ForwardLBPNeg(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.ABS, ForwardLBPAbs(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.CLAMP, ForwardLBPClamp(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.COS, ForwardLBPCos(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SIN, ForwardLBPSin(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.TAN, ForwardLBPTan(), signature=(AbstractValueType.ABSTRACT,)
    )

    # Linear operations
    ForwardLBPStrategyRegistry.register_default(
        OperationType.LINEAR, ForwardLBPLinear(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MATMUL,
        ForwardLBPMatmul(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.ABSTRACT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MATMUL,
        ForwardLBPMatmulConstant(),
        signature=(AbstractValueType.ABSTRACT, AbstractValueType.CONSTANT),
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MATMUL,
        ForwardLBPConstantMatmul(),
        signature=(AbstractValueType.CONSTANT, AbstractValueType.ABSTRACT),
    )

    # Reductions
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MEAN, ForwardLBPMean(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SUM, ForwardLBPSum(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MAX, ForwardLBPMax(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MIN, ForwardLBPMin(), signature=(AbstractValueType.ABSTRACT,)
    )

    # Reshaping operations
    ForwardLBPStrategyRegistry.register_default(
        OperationType.RESHAPE, ForwardLBPReshape(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.FLATTEN, ForwardLBPFlatten(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.CONCAT, ForwardLBPConcat(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.GETITEM, ForwardLBPGetItem(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.STACK, ForwardLBPStack(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SELECT, ForwardLBPSelect(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.UNSQUEEZE, ForwardLBPUnsqueeze(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SQUEEZE, ForwardLBPSqueeze(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.TRANSPOSE, ForwardLBPTranspose(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.PERMUTE, ForwardLBPTranspose(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.VIEW, ForwardLBPView(), signature=(AbstractValueType.ABSTRACT,)
    )


_register_strategies()
