import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import TargetRegistry
from .abs import IBPAbs
from .add import IBPAdd
from .base import ForwardIBPStrategy
from .cat import IBPCat
from .clamp import IBPClamp
from .cos import IBPCos
from .div import IBPDiv
from .exp import IBPExp
from .flatten import IBPFlatten
from .getitem import IBPGetItem
from .linear import IBPLinear
from .log import IBPLog
from .matmul import IBPMatmul
from .max import IBPMax
from .maximum import IBPMaximum
from .mean import IBPMean
from .min import IBPMin
from .minimum import IBPMinimum
from .mul import IBPMul
from .neg import IBPNeg
from .permute import IBPPermute
from .pow import IBPPow
from .reciprocal import IBPReciprocal
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
from .tan import IBPTan
from .tanh import IBPTanh
from .transpose import IBPTranspose
from .unsqueeze import IBPUnsqueeze
from .view import IBPView

__all__ = [
    "ForwardIBPStrategy",
    "create_default_ibp_registry",
]


def create_default_ibp_registry() -> TargetRegistry:
    """Create a :class:`TargetRegistry` pre-populated with all built-in IBP strategies."""
    registry = TargetRegistry()

    # -- Arithmetic (binary, merged constant variants) ---------------------
    registry.register_many([torch.add, operator.add], IBPAdd())
    registry.register_many([torch.sub, operator.sub], IBPSub())
    registry.register_many([torch.mul, operator.mul], IBPMul())
    registry.register_many([torch.div, operator.truediv], IBPDiv())

    registry.register(torch.maximum, IBPMaximum())
    registry.register(torch.minimum, IBPMinimum())

    registry.register_many([torch.neg, operator.neg], IBPNeg())

    registry.register_many([torch.matmul, operator.matmul], IBPMatmul())

    # -- Element-wise activations ------------------------------------------
    registry.register_many([torch.relu, F.relu, nn.ReLU], IBPRelu())
    registry.register_many([torch.sigmoid, F.sigmoid, nn.Sigmoid], IBPSigmoid())
    registry.register_many([torch.tanh, F.tanh, nn.Tanh], IBPTanh())

    registry.register(torch.exp, IBPExp())
    registry.register(torch.log, IBPLog())
    registry.register(torch.sqrt, IBPSqrt())
    registry.register(torch.reciprocal, IBPReciprocal())
    registry.register(torch.abs, IBPAbs())
    registry.register(torch.clamp, IBPClamp())
    registry.register(torch.sin, IBPSin())
    registry.register(torch.cos, IBPCos())
    registry.register(torch.tan, IBPTan())

    pow_ = IBPPow()
    registry.register_many([torch.pow, operator.pow], pow_)

    # TODO: no native cbrt; need to add custom method to torch.fx and then register here
    # registry.register(torch.Tensor.cbrt, IBPCbrt())

    # -- Linear / matmul ---------------------------------------------------
    linear = IBPLinear()
    registry.register_many([F.linear, nn.Linear], linear)

    # -- Reductions --------------------------------------------------------
    registry.register(torch.sum, IBPSum())
    registry.register(torch.mean, IBPMean())
    registry.register(torch.amax, IBPMax())
    registry.register(torch.amin, IBPMin())

    # -- Shape manipulation ------------------------------------------------
    registry.register(torch.reshape, IBPReshape())
    registry.register(torch.Tensor.reshape, IBPReshape())

    flatten = IBPFlatten()
    registry.register_many([torch.flatten, torch.Tensor.flatten, nn.Flatten], flatten)

    registry.register(torch.cat, IBPCat())
    registry.register(torch.stack, IBPStack())
    registry.register(operator.getitem, IBPGetItem())

    registry.register(torch.Tensor.select, IBPSelect())
    registry.register(torch.unsqueeze, IBPUnsqueeze())
    registry.register(torch.Tensor.unsqueeze, IBPUnsqueeze())
    registry.register(torch.squeeze, IBPSqueeze())
    registry.register(torch.Tensor.squeeze, IBPSqueeze())
    registry.register(torch.Tensor.transpose, IBPTranspose())
    registry.register(torch.Tensor.permute, IBPPermute())
    registry.register(torch.Tensor.view, IBPView())

    return registry
