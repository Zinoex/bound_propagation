import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import TargetRegistry
from .abs import ForwardLBPAbs
from .add import ForwardLBPAdd
from .base import ForwardLBPStrategy
from .cat import ForwardLBPConcat
from .clamp import ForwardLBPClamp
from .cos import ForwardLBPCos
from .div import ForwardLBPDiv
from .exp import ForwardLBPExp
from .flatten import ForwardLBPFlatten
from .getitem import ForwardLBPGetItem
from .linear import ForwardLBPLinear
from .log import ForwardLBPLog
from .matmul import ForwardLBPMatmul
from .max import ForwardLBPMax
from .maximum import ForwardLBPMaximum
from .mean import ForwardLBPMean
from .min import ForwardLBPMin
from .minimum import ForwardLBPMinimum
from .mul import ForwardLBPMul
from .neg import ForwardLBPNeg
from .reciprocal import ForwardLBPReciprocal
from .relu import ForwardLBPRelu
from .reshape import ForwardLBPReshape
from .select import ForwardLBPSelect
from .sigmoid import ForwardLBPSigmoid
from .sin import ForwardLBPSin
from .sqrt import ForwardLBPSqrt
from .squeeze import ForwardLBPSqueeze
from .stack import ForwardLBPStack
from .sub import ForwardLBPSub
from .sum import ForwardLBPSum
from .tan import ForwardLBPTan
from .tanh import ForwardLBPTanh
from .transpose import ForwardLBPPermute, ForwardLBPTranspose
from .unsqueeze import ForwardLBPUnsqueeze
from .view import ForwardLBPView

__all__ = [
    "ForwardLBPStrategy",
    "create_default_forward_lbp_registry",
]


def create_default_forward_lbp_registry() -> TargetRegistry:
    """Create a :class:`TargetRegistry` pre-populated with all built-in Forward LBP strategies."""
    registry = TargetRegistry()

    # -- Arithmetic (binary, merged constant variants) ---------------------
    add = ForwardLBPAdd()
    registry.register_many([torch.add, operator.add], add)

    sub = ForwardLBPSub()
    registry.register_many([torch.sub, operator.sub], sub)

    mul = ForwardLBPMul()
    registry.register_many([torch.mul, operator.mul], mul)

    div = ForwardLBPDiv()
    registry.register_many([torch.div, operator.truediv], div)

    registry.register(torch.maximum, ForwardLBPMaximum())
    registry.register(torch.minimum, ForwardLBPMinimum())

    neg = ForwardLBPNeg()
    registry.register_many([torch.neg, operator.neg], neg)

    registry.register_many([torch.matmul, operator.matmul], ForwardLBPMatmul())

    # -- Element-wise activations ------------------------------------------
    relu = ForwardLBPRelu()
    registry.register_many([torch.relu, F.relu, nn.ReLU], relu)

    sigmoid = ForwardLBPSigmoid()
    registry.register_many([torch.sigmoid, F.sigmoid, nn.Sigmoid], sigmoid)

    tanh = ForwardLBPTanh()
    registry.register_many([torch.tanh, F.tanh, nn.Tanh], tanh)

    registry.register(torch.exp, ForwardLBPExp())
    registry.register(torch.log, ForwardLBPLog())
    registry.register(torch.sqrt, ForwardLBPSqrt())
    registry.register(torch.reciprocal, ForwardLBPReciprocal())
    registry.register(torch.abs, ForwardLBPAbs())
    registry.register(torch.clamp, ForwardLBPClamp())
    registry.register(torch.sin, ForwardLBPSin())
    registry.register(torch.cos, ForwardLBPCos())
    registry.register(torch.tan, ForwardLBPTan())

    # -- Linear / matmul ---------------------------------------------------
    linear = ForwardLBPLinear()
    registry.register_many([F.linear, nn.Linear], linear)

    # -- Reductions --------------------------------------------------------
    registry.register(torch.sum, ForwardLBPSum())
    registry.register(torch.mean, ForwardLBPMean())
    registry.register(torch.amax, ForwardLBPMax())
    registry.register(torch.amin, ForwardLBPMin())

    # -- Shape manipulation ------------------------------------------------
    registry.register(torch.reshape, ForwardLBPReshape())
    registry.register(torch.Tensor.reshape, ForwardLBPReshape())

    flatten = ForwardLBPFlatten()
    registry.register_many([torch.flatten, torch.Tensor.flatten, nn.Flatten], flatten)

    registry.register(torch.cat, ForwardLBPConcat())
    registry.register(torch.stack, ForwardLBPStack())
    registry.register(operator.getitem, ForwardLBPGetItem())

    registry.register(torch.Tensor.select, ForwardLBPSelect())
    registry.register(torch.unsqueeze, ForwardLBPUnsqueeze())
    registry.register(torch.Tensor.unsqueeze, ForwardLBPUnsqueeze())
    registry.register(torch.squeeze, ForwardLBPSqueeze())
    registry.register(torch.Tensor.squeeze, ForwardLBPSqueeze())
    registry.register(torch.Tensor.transpose, ForwardLBPTranspose())
    registry.register(torch.Tensor.permute, ForwardLBPPermute())
    registry.register(torch.Tensor.view, ForwardLBPView())

    return registry
