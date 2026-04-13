from ...ir import AbstractValueType, OperationType
from .abs import ForwardLBPAbsStrategy
from .add import ForwardLBPAddStrategy, ForwardLBPAddWithConstant
from .base import ForwardLBPStrategy
from .cat import ForwardLBPConcatStrategy
from .clamp import ForwardLBPClampStrategy
from .cos import ForwardLBPCosStrategy
from .div import ForwardLBPConstantDiv, ForwardLBPDivConstant, ForwardLBPDivStrategy
from .exp import ForwardLBPExpStrategy
from .flatten import ForwardLBPFlattenStrategy
from .getitem import ForwardLBPGetItemStrategy
from .linear import ForwardLBPLinearStrategy
from .log import ForwardLBPLogStrategy
from .matmul import ForwardLBPConstantMatmul, ForwardLBPMatmulConstant, ForwardLBPMatmulStrategy
from .max import ForwardLBPMaxStrategy
from .maximum import ForwardLBPMaximumStrategy, ForwardLBPMaximumWithConstant
from .mean import ForwardLBPMeanStrategy
from .min import ForwardLBPMinStrategy
from .minimum import ForwardLBPMinimumStrategy, ForwardLBPMinimumWithConstant
from .mul import ForwardLBPMulStrategy, ForwardLBPMulWithConstant
from .neg import ForwardLBPNegStrategy
from .reciprocal import ForwardLBPReciprocalStrategy
from .registry import ForwardLBPStrategyRegistry
from .relu import ForwardLBPReluStrategy
from .reshape import ForwardLBPReshapeStrategy
from .select import ForwardLBPSelectStrategy
from .sigmoid import ForwardLBPSigmoidStrategy
from .sin import ForwardLBPSinStrategy
from .sqrt import ForwardLBPSqrtStrategy
from .squeeze import ForwardLBPSqueezeStrategy
from .stack import ForwardLBPStackStrategy
from .sub import ForwardLBPSubConstantLeft, ForwardLBPSubConstantRight, ForwardLBPSubStrategy
from .sum import ForwardLBPSumStrategy
from .tan import ForwardLBPTanStrategy
from .tanh import ForwardLBPTanhStrategy
from .transpose import ForwardLBPTransposeStrategy
from .unsqueeze import ForwardLBPUnsqueezeStrategy
from .view import ForwardLBPViewStrategy

__all__ = ["ForwardLBPStrategy", "ForwardLBPStrategyRegistry"]


def _register_strategies():
    """Register all ForwardLBP strategies with signatures."""

    # Arithmetic operations
    ForwardLBPStrategyRegistry.register_default(
        OperationType.ADD,
        ForwardLBPAddStrategy(),
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
        ForwardLBPSubStrategy(),
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
        ForwardLBPMulStrategy(),
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
        ForwardLBPDivStrategy(),
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
        ForwardLBPMaximumStrategy(),
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
        ForwardLBPMinimumStrategy(),
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
        OperationType.RELU, ForwardLBPReluStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SIGMOID, ForwardLBPSigmoidStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.TANH, ForwardLBPTanhStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.EXP, ForwardLBPExpStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.LOG, ForwardLBPLogStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SQRT, ForwardLBPSqrtStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.RECIPROCAL, ForwardLBPReciprocalStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.NEG, ForwardLBPNegStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.ABS, ForwardLBPAbsStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.CLAMP, ForwardLBPClampStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.COS, ForwardLBPCosStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SIN, ForwardLBPSinStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.TAN, ForwardLBPTanStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )

    # Linear operations
    ForwardLBPStrategyRegistry.register_default(
        OperationType.LINEAR, ForwardLBPLinearStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MATMUL,
        ForwardLBPMatmulStrategy(),
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
        OperationType.MEAN, ForwardLBPMeanStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SUM, ForwardLBPSumStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MAX, ForwardLBPMaxStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.MIN, ForwardLBPMinStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )

    # Reshaping operations
    ForwardLBPStrategyRegistry.register_default(
        OperationType.RESHAPE, ForwardLBPReshapeStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.FLATTEN, ForwardLBPFlattenStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.CONCAT, ForwardLBPConcatStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.GETITEM, ForwardLBPGetItemStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.STACK, ForwardLBPStackStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SELECT, ForwardLBPSelectStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.UNSQUEEZE, ForwardLBPUnsqueezeStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.SQUEEZE, ForwardLBPSqueezeStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.TRANSPOSE, ForwardLBPTransposeStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.PERMUTE, ForwardLBPTransposeStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )
    ForwardLBPStrategyRegistry.register_default(
        OperationType.VIEW, ForwardLBPViewStrategy(), signature=(AbstractValueType.ABSTRACT,)
    )


_register_strategies()
