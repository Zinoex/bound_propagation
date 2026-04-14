from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPLog(ForwardIBPStrategy):
    """IBP strategy for log (monotone for positive inputs)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPLog requires input to be IntervalBounds")

        if torch.any(x_bounds.lower <= 0):
            raise ValueError("log requires positive input bounds")

        return IntervalBounds(torch.log(x_bounds.lower), torch.log(x_bounds.upper))
