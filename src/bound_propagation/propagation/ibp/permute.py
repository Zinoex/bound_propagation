from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPPermute(ForwardIBPStrategy):
    """IBP strategy for permute."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPPermute requires the input to be an IntervalBounds")

        # torch.permute(x, dims) → args[1] is a tuple
        # x.permute(*dims) → args[1:] are individual ints
        if len(args) > 2:
            dims = tuple(args[1:])
        else:
            dims = args[1]

        lower = x_bounds.lower.permute(dims)
        upper = x_bounds.upper.permute(dims)

        return IntervalBounds(lower, upper)
