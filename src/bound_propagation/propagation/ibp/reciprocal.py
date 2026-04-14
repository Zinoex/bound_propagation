from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPReciprocal(ForwardIBPStrategy):
    """IBP strategy for reciprocal: 1/[a, b]."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPReciprocal requires input to be IntervalBounds")

        unbounded_mask = (x_bounds.lower <= 0) & (x_bounds.upper >= 0)

        lower = 1 / x_bounds.upper
        lower = torch.where(unbounded_mask, float("-inf"), lower)

        upper = 1 / x_bounds.lower
        upper = torch.where(unbounded_mask, float("inf"), upper)

        return IntervalBounds(lower, upper)
