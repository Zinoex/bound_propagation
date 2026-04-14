from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPMin(ForwardIBPStrategy):
    """IBP strategy for amin (reduction min along dimension)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPMin requires the input to be an IntervalBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)

        if dim is not None:
            lower = torch.amin(x_bounds.lower, dim=dim, keepdim=keepdim)
            upper = torch.amin(x_bounds.upper, dim=dim, keepdim=keepdim)
        else:
            lower = torch.amin(x_bounds.lower)
            upper = torch.amin(x_bounds.upper)

        return IntervalBounds(lower, upper)
