import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import TargetRegistry
from .base import ForwardLBPStrategy
from .elementwise import (
    ForwardLBPAbs,
    ForwardLBPClamp,
    ForwardLBPCos,
    ForwardLBPExp,
    ForwardLBPLog,
    ForwardLBPReciprocal,
    ForwardLBPRelu,
    ForwardLBPSigmoid,
    ForwardLBPSin,
    ForwardLBPSqrt,
    ForwardLBPTan,
    ForwardLBPTanh,
)
from .linear import ForwardLBPAdd, ForwardLBPLinear, ForwardLBPNeg, ForwardLBPSub
from .matmul import ForwardLBPMatmul
from .pairwise import ForwardLBPDiv, ForwardLBPMaximum, ForwardLBPMinimum, ForwardLBPMul
from .reduction import ForwardLBPMax, ForwardLBPMean, ForwardLBPMin, ForwardLBPSum
from .shape import (
    ForwardLBPConcat,
    ForwardLBPFlatten,
    ForwardLBPGetItem,
    ForwardLBPPermute,
    ForwardLBPReshape,
    ForwardLBPSelect,
    ForwardLBPSqueeze,
    ForwardLBPStack,
    ForwardLBPTranspose,
    ForwardLBPUnsqueeze,
    ForwardLBPView,
)

__all__ = [
    "ForwardLBPStrategy",
    "create_default_forward_lbp_registry",
]


def create_default_forward_lbp_registry() -> TargetRegistry[ForwardLBPStrategy]:
    """Create a :class:`TargetRegistry` pre-populated with all built-in Forward LBP strategies."""
    registry = TargetRegistry[ForwardLBPStrategy]()

    # -- Arithmetic (binary, merged constant variants) ---------------------
    registry.register_many([torch.add, operator.add], ForwardLBPAdd())
    registry.register_many([torch.sub, operator.sub], ForwardLBPSub())
    registry.register_many([torch.mul, operator.mul], ForwardLBPMul())
    registry.register_many([torch.div, operator.truediv], ForwardLBPDiv())

    registry.register(torch.maximum, ForwardLBPMaximum())
    registry.register(torch.minimum, ForwardLBPMinimum())

    registry.register_many([torch.neg, operator.neg], ForwardLBPNeg())

    registry.register_many([torch.matmul, operator.matmul], ForwardLBPMatmul())

    # -- Element-wise activations ------------------------------------------
    registry.register_many([torch.relu, F.relu, nn.ReLU], ForwardLBPRelu())
    registry.register_many([torch.sigmoid, F.sigmoid, nn.Sigmoid], ForwardLBPSigmoid())
    registry.register_many([torch.tanh, F.tanh, nn.Tanh], ForwardLBPTanh())

    registry.register_many([torch.exp, torch.Tensor.exp], ForwardLBPExp())
    registry.register_many([torch.log, torch.Tensor.log], ForwardLBPLog())
    registry.register_many([torch.sqrt, torch.Tensor.sqrt], ForwardLBPSqrt())
    registry.register_many([torch.reciprocal, torch.Tensor.reciprocal], ForwardLBPReciprocal())
    registry.register_many([torch.abs, torch.Tensor.abs], ForwardLBPAbs())
    registry.register_many([torch.clamp, torch.Tensor.clamp], ForwardLBPClamp())
    registry.register_many([torch.sin, torch.Tensor.sin], ForwardLBPSin())
    registry.register_many([torch.cos, torch.Tensor.cos], ForwardLBPCos())
    registry.register_many([torch.tan, torch.Tensor.tan], ForwardLBPTan())

    # -- Linear / matmul ---------------------------------------------------
    linear = ForwardLBPLinear()
    registry.register_many([F.linear, nn.Linear], linear)

    # -- Reductions --------------------------------------------------------
    registry.register_many([torch.sum, torch.Tensor.sum], ForwardLBPSum())
    registry.register_many([torch.mean, torch.Tensor.mean], ForwardLBPMean())
    registry.register_many([torch.amax, torch.Tensor.amax], ForwardLBPMax())
    registry.register_many([torch.amin, torch.Tensor.amin], ForwardLBPMin())

    # -- Shape manipulation ------------------------------------------------
    registry.register_many([torch.reshape, torch.Tensor.reshape], ForwardLBPReshape())

    flatten = ForwardLBPFlatten()
    registry.register_many([torch.flatten, torch.Tensor.flatten, nn.Flatten], flatten)

    registry.register(torch.cat, ForwardLBPConcat())
    registry.register(torch.stack, ForwardLBPStack())
    registry.register(operator.getitem, ForwardLBPGetItem())

    registry.register(torch.Tensor.select, ForwardLBPSelect())
    registry.register_many([torch.unsqueeze, torch.Tensor.unsqueeze], ForwardLBPUnsqueeze())
    registry.register_many([torch.squeeze, torch.Tensor.squeeze], ForwardLBPSqueeze())
    registry.register_many([torch.Tensor.transpose, torch.transpose], ForwardLBPTranspose())
    registry.register_many([torch.Tensor.permute, torch.permute], ForwardLBPPermute())
    registry.register(torch.Tensor.view, ForwardLBPView())

    return registry
