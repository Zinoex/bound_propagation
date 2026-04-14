from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPAdd(ForwardIBPStrategy):
    """IBP strategy for addition (abstract+abstract or abstract+constant)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        left, right = args[0], args[1]

        if isinstance(left, IntervalBounds) and isinstance(right, IntervalBounds):
            return IntervalBounds(left.lower + right.lower, left.upper + right.upper)

        if isinstance(left, IntervalBounds):
            return IntervalBounds(left.lower + right, left.upper + right)

        if isinstance(right, IntervalBounds):
            return IntervalBounds(left + right.lower, left + right.upper)

        raise TypeError(f"IBPAdd requires at least one IntervalBounds, got {type(left)} and {type(right)}")
