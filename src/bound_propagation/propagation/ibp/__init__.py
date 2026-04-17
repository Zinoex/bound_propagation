import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import TargetRegistry
from .base import ForwardIBPStrategy
from .elementwise import (
    IBPAbs,
    IBPClamp,
    IBPCos,
    IBPExp,
    IBPLog,
    IBPPow,
    IBPReciprocal,
    IBPRelu,
    IBPSigmoid,
    IBPSin,
    IBPSqrt,
    IBPTan,
    IBPTanh,
)
from .linear import IBPAdd, IBPLinear, IBPNeg, IBPSub
from .matmul import IBPMatmul
from .pairwise import IBPDiv, IBPMaximum, IBPMinimum, IBPMul
from .reduction import IBPMax, IBPMean, IBPMin, IBPSum
from .shape import (
    IBPCat,
    IBPFlatten,
    IBPGetItem,
    IBPPermute,
    IBPReshape,
    IBPSelect,
    IBPSqueeze,
    IBPStack,
    IBPTranspose,
    IBPUnsqueeze,
    IBPView,
)

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

    registry.register_many([torch.exp, torch.Tensor.exp], IBPExp())
    registry.register_many([torch.log, torch.Tensor.log], IBPLog())
    registry.register_many([torch.sqrt, torch.Tensor.sqrt], IBPSqrt())
    registry.register_many([torch.reciprocal, torch.Tensor.reciprocal], IBPReciprocal())
    registry.register_many([torch.abs, torch.Tensor.abs], IBPAbs())
    registry.register_many([torch.clamp, torch.Tensor.clamp], IBPClamp())
    registry.register_many([torch.sin, torch.Tensor.sin], IBPSin())
    registry.register_many([torch.cos, torch.Tensor.cos], IBPCos())
    registry.register_many([torch.tan, torch.Tensor.tan], IBPTan())

    registry.register_many([torch.pow, operator.pow], IBPPow())

    # TODO: no native cbrt; need to add custom method to torch.fx and then register here
    # registry.register(torch.Tensor.cbrt, IBPCbrt())

    # -- Linear / matmul ---------------------------------------------------
    registry.register_many([F.linear, nn.Linear], IBPLinear())

    # -- Reductions --------------------------------------------------------
    registry.register_many([torch.sum, torch.Tensor.sum], IBPSum())
    registry.register_many([torch.mean, torch.Tensor.mean], IBPMean())
    registry.register_many([torch.amax, torch.Tensor.amax], IBPMax())
    registry.register_many([torch.amin, torch.Tensor.amin], IBPMin())

    # -- Shape manipulation ------------------------------------------------
    registry.register_many([torch.reshape, torch.Tensor.reshape], IBPReshape())
    registry.register_many([torch.flatten, torch.Tensor.flatten, nn.Flatten], IBPFlatten())

    registry.register(torch.cat, IBPCat())
    registry.register(torch.stack, IBPStack())
    registry.register(operator.getitem, IBPGetItem())
    registry.register(torch.Tensor.view, IBPView())

    registry.register_many([torch.Tensor.select, torch.select], IBPSelect())
    registry.register_many([torch.unsqueeze, torch.Tensor.unsqueeze], IBPUnsqueeze())
    registry.register_many([torch.squeeze, torch.Tensor.squeeze], IBPSqueeze())
    registry.register_many([torch.transpose, torch.Tensor.transpose], IBPTranspose())
    registry.register_many([torch.permute, torch.Tensor.permute], IBPPermute())

    return registry
