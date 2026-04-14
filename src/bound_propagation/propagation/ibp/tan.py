from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPTan(ForwardIBPStrategy):
    """IBP strategy for tan with asymptote detection."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPTan requires input to be IntervalBounds")

        lower = torch.tan(x_bounds.lower)
        upper = torch.tan(x_bounds.upper)

        pi_over_2 = torch.pi / 2
        eps = torch.finfo(x_bounds.lower.dtype).eps * 8
        k_min = torch.ceil((x_bounds.lower - pi_over_2 - eps) / torch.pi)
        k_max = torch.floor((x_bounds.upper - pi_over_2 + eps) / torch.pi)

        contains_asymptote = k_min <= k_max
        lower[contains_asymptote] = float("-inf")
        upper[contains_asymptote] = float("inf")

        return IntervalBounds(lower, upper)
