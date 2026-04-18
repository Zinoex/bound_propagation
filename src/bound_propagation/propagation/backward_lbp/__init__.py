import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import TargetRegistry
from .base import BackwardLBPStrategy
from .elementwise import (
    BackwardLBPAbs,
    BackwardLBPClamp,
    BackwardLBPCos,
    BackwardLBPExp,
    BackwardLBPLog,
    BackwardLBPReciprocal,
    BackwardLBPRelu,
    BackwardLBPSigmoid,
    BackwardLBPSin,
    BackwardLBPSqrt,
    BackwardLBPTan,
    BackwardLBPTanh,
)
from .linear import (
    BackwardLBPAdd,
    BackwardLBPLinear,
    BackwardLBPMatmul,
    BackwardLBPNeg,
    BackwardLBPSub,
)
from .pairwise import (
    BackwardLBPDiv,
    BackwardLBPMaximum,
    BackwardLBPMinimum,
    BackwardLBPMul,
)
from .reduction import BackwardLBPMax, BackwardLBPMean, BackwardLBPMin, BackwardLBPSum
from .shape import (
    BackwardLBPConcat,
    BackwardLBPGetItem,
    BackwardLBPPermute,
    BackwardLBPReshape,
    BackwardLBPSelect,
    BackwardLBPSqueeze,
    BackwardLBPStack,
    BackwardLBPTranspose,
    BackwardLBPUnsqueeze,
)

__all__ = [
    "BackwardLBPStrategy",
    "create_default_backward_lbp_registry",
]


def create_default_backward_lbp_registry() -> TargetRegistry[BackwardLBPStrategy]:
    """Create a TargetRegistry pre-populated with built-in backward LBP strategies."""
    registry = TargetRegistry[BackwardLBPStrategy]()

    # -- Arithmetic -----------------------------------------------------------
    registry.register_many([torch.add, operator.add], BackwardLBPAdd())
    registry.register_many([torch.sub, operator.sub], BackwardLBPSub())
    registry.register_many([torch.mul, operator.mul], BackwardLBPMul())
    registry.register_many([torch.div, operator.truediv], BackwardLBPDiv())

    registry.register_many([torch.neg, operator.neg], BackwardLBPNeg())

    registry.register_many([torch.matmul, operator.matmul], BackwardLBPMatmul())

    registry.register(torch.maximum, BackwardLBPMaximum())
    registry.register(torch.minimum, BackwardLBPMinimum())

    # -- Element-wise activations ---------------------------------------------
    registry.register_many([torch.relu, F.relu, nn.ReLU], BackwardLBPRelu())
    registry.register_many([torch.sigmoid, F.sigmoid, nn.Sigmoid], BackwardLBPSigmoid())
    registry.register_many([torch.tanh, F.tanh, nn.Tanh], BackwardLBPTanh())

    registry.register_many([torch.exp, torch.Tensor.exp], BackwardLBPExp())
    registry.register_many([torch.log, torch.Tensor.log], BackwardLBPLog())
    registry.register_many([torch.sqrt, torch.Tensor.sqrt], BackwardLBPSqrt())
    registry.register_many([torch.reciprocal, torch.Tensor.reciprocal], BackwardLBPReciprocal())
    registry.register_many([torch.abs, torch.Tensor.abs], BackwardLBPAbs())
    registry.register_many([torch.clamp, torch.Tensor.clamp], BackwardLBPClamp())
    registry.register_many([torch.sin, torch.Tensor.sin], BackwardLBPSin())
    registry.register_many([torch.cos, torch.Tensor.cos], BackwardLBPCos())
    registry.register_many([torch.tan, torch.Tensor.tan], BackwardLBPTan())

    # -- Linear / affine ------------------------------------------------------
    registry.register_many([F.linear, nn.Linear], BackwardLBPLinear())

    # -- Reductions -----------------------------------------------------------
    registry.register_many([torch.sum, torch.Tensor.sum], BackwardLBPSum())
    registry.register_many([torch.mean, torch.Tensor.mean], BackwardLBPMean())
    registry.register_many([torch.amax, torch.Tensor.amax], BackwardLBPMax())
    registry.register_many([torch.amin, torch.Tensor.amin], BackwardLBPMin())

    # -- Shape manipulation ---------------------------------------------------
    registry.register_many([torch.reshape, torch.Tensor.reshape], BackwardLBPReshape())
    # registry.register_many([torch.flatten, torch.Tensor.flatten, nn.Flatten], BackwardLBPFlatten())
    # registry.register(torch.Tensor.view, BackwardLBPView())

    registry.register_many([torch.unsqueeze, torch.Tensor.unsqueeze], BackwardLBPUnsqueeze())
    registry.register_many([torch.squeeze, torch.Tensor.squeeze], BackwardLBPSqueeze())
    registry.register_many([torch.Tensor.transpose], BackwardLBPTranspose())
    registry.register_many([torch.Tensor.permute], BackwardLBPPermute())

    registry.register(operator.getitem, BackwardLBPGetItem())
    registry.register(torch.Tensor.select, BackwardLBPSelect())

    # -- Concatenation / Stacking ---------------------------------------------
    registry.register(torch.cat, BackwardLBPConcat())
    registry.register(torch.stack, BackwardLBPStack())

    return registry
