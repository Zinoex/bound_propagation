from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPSelect(ForwardIBPStrategy):
    """IBP strategy for select."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSelect requires the input to be an IntervalBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        index = args[2] if len(args) > 2 else kwargs.get("index")

        if index is None:
            raise ValueError("select requires an index argument")

        return IntervalBounds(
            x_bounds.lower.select(dim=dim, index=index),
            x_bounds.upper.select(dim=dim, index=index),
        )
