from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPTranspose(ForwardIBPStrategy):
    """IBP strategy for transpose."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPTranspose requires the input to be an IntervalBounds")

        dim0 = args[1] if len(args) > 1 else kwargs.get("dim0")
        dim1 = args[2] if len(args) > 2 else kwargs.get("dim1")

        if dim0 is None or dim1 is None:
            raise ValueError("transpose requires dim0 and dim1 arguments")

        lower = x_bounds.lower.transpose(dim0, dim1)
        upper = x_bounds.upper.transpose(dim0, dim1)

        return IntervalBounds(lower, upper)
