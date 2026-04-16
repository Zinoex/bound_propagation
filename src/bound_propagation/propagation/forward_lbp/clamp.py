from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from ..linear_relaxations.clamp import compute_clamp_relaxation
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPClamp(ForwardLBPStrategy):
    """Forward LBP strategy for clamp using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPClamp requires input to be LinearBounds")

        min_val = args[1] if len(args) > 1 else kwargs.get("min")
        max_val = args[2] if len(args) > 2 else kwargs.get("max")

        lower, upper = bounds.concretize()
        relaxation = compute_clamp_relaxation(lower, upper, min_val, max_val)
        return relaxation.forward_compose([bounds])
