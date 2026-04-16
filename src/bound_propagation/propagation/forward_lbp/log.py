from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from ..linear_relaxations.log import compute_log_relaxation
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPLog(ForwardLBPStrategy):
    """Forward LBP strategy for log using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPLog requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_log_relaxation(lower, upper)
        return relaxation.forward_compose([bounds])
