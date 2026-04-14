from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPFlatten(ForwardIBPStrategy):
    """IBP strategy for flatten."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPFlatten requires input to be IntervalBounds")

        if node.op == "call_module":
            module = ctx.get_module(node.target)
            start_dim = module.start_dim
            end_dim = module.end_dim
        else:
            start_dim = args[1] if len(args) > 1 else kwargs.get("start_dim", 0)
            end_dim = args[2] if len(args) > 2 else kwargs.get("end_dim", -1)

        lower = torch.flatten(x_bounds.lower, start_dim, end_dim)
        upper = torch.flatten(x_bounds.upper, start_dim, end_dim)

        return IntervalBounds(lower, upper)
