from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPCbrt(ForwardIBPStrategy):
    """IBP strategy for cbrt (monotonic)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPCbrt requires input to be IntervalBounds")

        lower = torch.copysign(torch.pow(x_bounds.lower.abs(), 1 / 3), x_bounds.lower)
        upper = torch.copysign(torch.pow(x_bounds.upper.abs(), 1 / 3), x_bounds.upper)

        return IntervalBounds(lower, upper)
