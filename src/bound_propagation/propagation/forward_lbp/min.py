from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPMin(ForwardLBPStrategy):
    """Forward LBP strategy for min reduction."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPMin requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        lower, upper = bounds.concretize()

        if dim is not None:
            lower = lower.min(dim=dim, keepdim=keepdim).values
            upper = upper.min(dim=dim, keepdim=keepdim).values
        else:
            lower = lower.min()
            upper = upper.min()

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
