from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from ..linear_relaxations.sigmoid import compute_sigmoid_relaxation
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPSigmoid(ForwardLBPStrategy):
    """Forward LBP strategy for sigmoid using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPSigmoid requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_sigmoid_relaxation(lower, upper)
        return relaxation.forward_compose([bounds])
