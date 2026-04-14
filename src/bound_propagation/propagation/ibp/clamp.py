from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPClamp(ForwardIBPStrategy):
    """IBP strategy for clamp."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPClamp requires input to be IntervalBounds")

        clamp_min = args[1] if len(args) > 1 else kwargs.get("min", None)
        clamp_max = args[2] if len(args) > 2 else kwargs.get("max", None)

        lower = torch.clamp(x_bounds.lower, min=clamp_min, max=clamp_max)
        upper = torch.clamp(x_bounds.upper, min=clamp_min, max=clamp_max)

        return IntervalBounds(lower, upper)
