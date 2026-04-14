from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPGetItem(ForwardLBPStrategy):
    """Forward LBP strategy for getitem (operator.getitem)."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]
        index = args[1]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPGetItem requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        lower = lower[index]
        upper = upper[index]

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
