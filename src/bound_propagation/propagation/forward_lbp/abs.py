from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..linear_relaxations.abs import compute_abs_relaxation
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch.fx as fx

    from ..context import PropagationContext


class ForwardLBPAbs(ForwardLBPStrategy):
    """Forward LBP strategy for abs using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPAbs requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        relaxation = compute_abs_relaxation(lower, upper)
        return relaxation.forward_compose([bounds])
