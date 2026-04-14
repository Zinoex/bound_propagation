from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPReshape(ForwardIBPStrategy):
    """IBP strategy for reshape."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPReshape requires input to be IntervalBounds")

        # torch.reshape(x, shape) → args[1] is a tuple
        # x.reshape(*shape) → args[1:] are individual ints
        if len(args) > 2 or (len(args) == 2 and isinstance(args[1], int)):
            shape = tuple(args[1:])
        else:
            shape = args[1]

        lower = x_bounds.lower.reshape(shape)
        upper = x_bounds.upper.reshape(shape)

        return IntervalBounds(lower, upper)
