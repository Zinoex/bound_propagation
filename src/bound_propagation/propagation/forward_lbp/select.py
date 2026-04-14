from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPSelect(ForwardLBPStrategy):
    """Forward LBP strategy for select."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPSelect requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        index = args[2] if len(args) > 2 else kwargs.get("index", 0)

        output_ndim = bounds.bias_lower.ndim
        if dim < 0:
            dim += output_ndim
        if dim < 0 or dim >= output_ndim:
            raise ValueError(f"select dim must be in [0, {output_ndim - 1}], got {dim}")

        linear_lower = bounds.linear_lower.select(dim, index) if bounds.linear_lower is not None else None
        linear_upper = bounds.linear_upper.select(dim, index) if bounds.linear_upper is not None else None
        bias_lower = bounds.bias_lower.select(dim, index)
        bias_upper = bounds.bias_upper.select(dim, index)

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
