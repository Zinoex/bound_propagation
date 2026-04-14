from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPReshape(ForwardLBPStrategy):
    """Forward LBP strategy for reshape."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPReshape requires input to be LinearBounds")

        target_shape = args[1:]

        bias_lower = bounds.bias_lower.reshape(target_shape)
        bias_upper = bounds.bias_upper.reshape(target_shape)

        return LinearBounds(
            region=bounds.region,
            linear_lower=bounds.linear_lower,
            bias_lower=bias_lower,
            linear_upper=bounds.linear_upper,
            bias_upper=bias_upper,
        )
