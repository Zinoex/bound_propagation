from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPMul(ForwardIBPStrategy):
    """IBP strategy for multiplication (abstract*abstract or abstract*constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, IntervalBounds) and isinstance(right, IntervalBounds):
            ll = left.lower * right.lower
            lu = left.lower * right.upper
            ul = left.upper * right.lower
            uu = left.upper * right.upper
            lower = torch.min(torch.min(ll, lu), torch.min(ul, uu))
            upper = torch.max(torch.max(ll, lu), torch.max(ul, uu))
            return IntervalBounds(lower, upper)

        # One side is constant
        if isinstance(left, IntervalBounds):
            interval, c = left, right
        elif isinstance(right, IntervalBounds):
            interval, c = right, left
        else:
            raise TypeError(f"IBPMul requires at least one IntervalBounds, got {type(left)} and {type(right)}")

        if isinstance(c, torch.Tensor):
            lower = torch.where(c >= 0, interval.lower * c, interval.upper * c)
            upper = torch.where(c >= 0, interval.upper * c, interval.lower * c)
            return IntervalBounds(lower, upper)

        # Scalar constant
        if c >= 0:
            return IntervalBounds(interval.lower * c, interval.upper * c)
        return IntervalBounds(interval.upper * c, interval.lower * c)
