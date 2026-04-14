from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPSqueeze(ForwardIBPStrategy):
    """IBP strategy for squeeze."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSqueeze requires input to be IntervalBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim", None)

        lower = x_bounds.lower.squeeze(dim=dim)
        upper = x_bounds.upper.squeeze(dim=dim)

        return IntervalBounds(lower, upper)
