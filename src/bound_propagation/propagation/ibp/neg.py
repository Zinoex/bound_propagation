from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPNeg(ForwardIBPStrategy):
    """IBP strategy for negation: -[a, b] = [-b, -a]."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPNeg requires input to be IntervalBounds")

        return IntervalBounds(-x_bounds.upper, -x_bounds.lower)
