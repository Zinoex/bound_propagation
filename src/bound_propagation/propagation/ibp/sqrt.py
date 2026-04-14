from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPSqrt(ForwardIBPStrategy):
    """IBP strategy for sqrt (monotone for non-negative inputs)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSqrt requires input to be IntervalBounds")

        if torch.any(x_bounds.lower < 0):
            raise ValueError("sqrt requires non-negative input bounds")

        return IntervalBounds(torch.sqrt(x_bounds.lower), torch.sqrt(x_bounds.upper))
