from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPLinear(ForwardLBPStrategy):
    """Forward LBP strategy for nn.Linear / F.linear."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPLinear requires input to be LinearBounds")

        if node.op == "call_module":
            module = ctx.get_module(node.target)
            weight = module.weight
            bias = getattr(module, "bias", None)
        else:
            # F.linear(input, weight, bias=None)
            weight = args[1]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")

        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)

        # Lower bound: weight_pos @ lower_coeffs + weight_neg @ upper_coeffs
        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            linear_lower = weight_pos @ bounds.linear_lower + weight_neg @ bounds.linear_upper
        elif bounds.linear_lower is not None:
            linear_lower = weight_pos @ bounds.linear_lower + weight_neg @ bounds.linear_lower
        elif bounds.linear_upper is not None:
            linear_lower = weight_pos @ bounds.linear_upper + weight_neg @ bounds.linear_upper
        else:
            linear_lower = None

        bias_lower = weight_pos @ bounds.bias_lower + weight_neg @ bounds.bias_upper
        if bias is not None:
            bias_lower = bias_lower + bias

        # Upper bound: weight_pos @ upper_coeffs + weight_neg @ lower_coeffs
        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            linear_upper = weight_pos @ bounds.linear_upper + weight_neg @ bounds.linear_lower
        elif bounds.linear_upper is not None:
            linear_upper = weight_pos @ bounds.linear_upper + weight_neg @ bounds.linear_upper
        elif bounds.linear_lower is not None:
            linear_upper = weight_pos @ bounds.linear_lower + weight_neg @ bounds.linear_lower
        else:
            linear_upper = None

        bias_upper = weight_pos @ bounds.bias_upper + weight_neg @ bounds.bias_lower
        if bias is not None:
            bias_upper = bias_upper + bias

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
