from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPLinear(ForwardIBPStrategy):
    """IBP strategy for linear: y = x @ W^T + b."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPLinear requires input to be IntervalBounds")

        if node.op == "call_module":
            module = ctx.get_module(node.target)
            weight = module.weight
            bias = getattr(module, "bias", None)
        else:
            # F.linear(input, weight, bias=None)
            weight = args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")

        weight_pos = torch.clamp(weight, min=0)
        weight_neg = torch.clamp(weight, max=0)

        lower = x_bounds.lower @ weight_pos.T + x_bounds.upper @ weight_neg.T
        upper = x_bounds.upper @ weight_pos.T + x_bounds.lower @ weight_neg.T

        if bias is not None:
            lower = lower + bias
            upper = upper + bias

        return IntervalBounds(lower, upper)
