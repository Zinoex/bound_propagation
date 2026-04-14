from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPCat(ForwardIBPStrategy):
    """IBP strategy for cat."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        tensors = args[0]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)

        if not isinstance(tensors, (list, tuple)):
            raise TypeError("IBPCat expects first argument to be a list/tuple of tensors")

        for i, b in enumerate(tensors):
            if not isinstance(b, IntervalBounds):
                raise TypeError(f"IBPCat requires all inputs to be IntervalBounds, but input {i} is {type(b)}")

        lower = torch.cat([b.lower for b in tensors], dim=dim)
        upper = torch.cat([b.upper for b in tensors], dim=dim)

        return IntervalBounds(lower, upper)
