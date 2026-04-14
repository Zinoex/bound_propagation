from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPRelu(ForwardIBPStrategy):
    """IBP strategy for relu: relu([a, b]) = [max(0, a), max(0, b)]."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPRelu requires input to be IntervalBounds")

        lower = torch.clamp(x_bounds.lower, min=0.0)
        upper = torch.clamp(x_bounds.upper, min=0.0)

        return IntervalBounds(lower, upper)
