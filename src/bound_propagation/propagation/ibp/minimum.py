from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPMinimum(ForwardIBPStrategy):
    """IBP strategy for element-wise minimum (abstract and/or constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, IntervalBounds) and isinstance(right, IntervalBounds):
            return IntervalBounds(
                torch.min(left.lower, right.lower),
                torch.min(left.upper, right.upper),
            )

        if isinstance(left, IntervalBounds):
            interval, c = left, right
        elif isinstance(right, IntervalBounds):
            interval, c = right, left
        else:
            raise TypeError(f"IBPMinimum requires at least one IntervalBounds, got {type(left)} and {type(right)}")

        if not isinstance(c, torch.Tensor):
            raise TypeError(f"IBPMinimum requires the constant input to be a torch.Tensor, got {type(c)}")

        return IntervalBounds(torch.min(interval.lower, c), torch.min(interval.upper, c))
