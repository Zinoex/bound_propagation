from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPUnsqueeze(ForwardLBPStrategy):
    """Forward LBP strategy for unsqueeze."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPUnsqueeze requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        linear_lower = bounds.linear_lower.unsqueeze(dim) if bounds.linear_lower is not None else None
        linear_upper = bounds.linear_upper.unsqueeze(dim) if bounds.linear_upper is not None else None
        bias_lower = bounds.bias_lower.unsqueeze(dim)
        bias_upper = bounds.bias_upper.unsqueeze(dim)

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
