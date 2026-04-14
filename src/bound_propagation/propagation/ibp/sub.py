from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPSub(ForwardIBPStrategy):
    """IBP strategy for subtraction (all combinations of abstract/constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, IntervalBounds) and isinstance(right, IntervalBounds):
            return IntervalBounds(left.lower - right.upper, left.upper - right.lower)

        if isinstance(left, IntervalBounds):
            # abstract - constant
            return IntervalBounds(left.lower - right, left.upper - right)

        if isinstance(right, IntervalBounds):
            # constant - abstract
            return IntervalBounds(left - right.upper, left - right.lower)

        raise TypeError(f"IBPSub requires at least one IntervalBounds, got {type(left)} and {type(right)}")
