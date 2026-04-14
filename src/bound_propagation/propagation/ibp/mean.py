from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPMean(ForwardIBPStrategy):
    """IBP strategy for mean."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPMean requires the input to be an IntervalBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)

        lower = x_bounds.lower.mean(dim=dim, keepdim=keepdim)
        upper = x_bounds.upper.mean(dim=dim, keepdim=keepdim)

        return IntervalBounds(lower, upper)
