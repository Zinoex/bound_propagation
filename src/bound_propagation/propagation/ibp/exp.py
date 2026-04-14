from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPExp(ForwardIBPStrategy):
    """IBP strategy for exp (monotone): exp([a, b]) = [exp(a), exp(b)]."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPExp requires input to be IntervalBounds")

        return IntervalBounds(torch.exp(x_bounds.lower), torch.exp(x_bounds.upper))
