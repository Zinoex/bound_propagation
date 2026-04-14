from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPAbs(ForwardIBPStrategy):
    """IBP strategy for abs: abs([a, b]) accounts for sign-crossing."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPAbs requires input to be IntervalBounds")

        abs_lower = x_bounds.lower.abs()
        abs_upper = x_bounds.upper.abs()

        zero = torch.zeros_like(abs_lower)
        lower = torch.where(
            (x_bounds.lower < 0) & (x_bounds.upper > 0),
            zero,
            torch.min(abs_lower, abs_upper),
        )
        upper = torch.max(abs_lower, abs_upper)

        return IntervalBounds(lower, upper)
