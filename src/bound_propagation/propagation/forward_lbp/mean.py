from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPMean(ForwardLBPStrategy):
    """Forward LBP strategy for mean reduction."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPMean requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        lower, upper = bounds.concretize()
        lower = lower.mean(dim=dim, keepdim=keepdim)
        upper = upper.mean(dim=dim, keepdim=keepdim)

        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=lower,
            linear_upper=[],
            bias_upper=upper,
        )
