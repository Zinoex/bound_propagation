from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPTranspose(ForwardLBPStrategy):
    """Forward LBP strategy for transpose and permute."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPTranspose requires input to be LinearBounds")

        # torch.transpose(input, dim0, dim1)
        dim0 = args[1] if len(args) > 1 else kwargs.get("dim0", 0)
        dim1 = args[2] if len(args) > 2 else kwargs.get("dim1", 1)

        linear_lower = bounds.linear_lower.transpose(dim0, dim1) if bounds.linear_lower is not None else None
        linear_upper = bounds.linear_upper.transpose(dim0, dim1) if bounds.linear_upper is not None else None
        bias_lower = bounds.bias_lower.transpose(dim0, dim1)
        bias_upper = bounds.bias_upper.transpose(dim0, dim1)

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )


class ForwardLBPPermute(ForwardLBPStrategy):
    """Forward LBP strategy for permute."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPPermute requires input to be LinearBounds")

        dims = args[1:]

        linear_lower = bounds.linear_lower.permute(*dims) if bounds.linear_lower is not None else None
        linear_upper = bounds.linear_upper.permute(*dims) if bounds.linear_upper is not None else None
        bias_lower = bounds.bias_lower.permute(*dims)
        bias_upper = bounds.bias_upper.permute(*dims)

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
