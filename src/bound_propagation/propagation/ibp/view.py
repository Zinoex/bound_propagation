from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPView(ForwardIBPStrategy):
    """IBP strategy for view."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPView requires input to be IntervalBounds")

        # x.view(*shape) → args[1:] are individual ints
        # x.view(shape) → args[1] is a tuple
        if len(args) > 2 or (len(args) == 2 and isinstance(args[1], int)):
            size = tuple(args[1:])
        else:
            size = args[1]

        lower = x_bounds.lower.view(size)
        upper = x_bounds.upper.view(size)

        return IntervalBounds(lower, upper)
