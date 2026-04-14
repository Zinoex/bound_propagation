from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPFlatten(ForwardLBPStrategy):
    """Forward LBP strategy for flatten."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPFlatten requires input to be LinearBounds")

        start_dim = args[1] if len(args) > 1 else kwargs.get("start_dim", 0)
        end_dim = args[2] if len(args) > 2 else kwargs.get("end_dim", -1)

        bias_lower = bounds.bias_lower.flatten(start_dim, end_dim)
        bias_upper = bounds.bias_upper.flatten(start_dim, end_dim)

        return LinearBounds(
            region=bounds.region,
            linear_lower=bounds.linear_lower,
            bias_lower=bias_lower,
            linear_upper=bounds.linear_upper,
            bias_upper=bias_upper,
        )
