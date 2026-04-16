from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPMax(ForwardLBPStrategy):
    """Forward LBP strategy for max reduction."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPMax requires input to be LinearBounds")

        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        keepdim = kwargs.get("keepdim", False)

        lower, upper = bounds.concretize()

        if dim is not None:
            lower = lower.amax(dim=dim, keepdim=keepdim)
            upper = upper.amax(dim=dim, keepdim=keepdim)
        else:
            lower = lower.amax()
            upper = upper.amax()

        return LinearBounds(
            regions=[],
            linear_lower=[],
            bias_lower=lower,
            linear_upper=[],
            bias_upper=upper,
        )
