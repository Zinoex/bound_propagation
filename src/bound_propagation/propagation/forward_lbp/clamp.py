from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from ..linear_relaxations.clamp import compute_clamp_alpha_beta
from .base import ForwardLBPStrategy
from .utils import apply_linear_relaxation

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
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)
        return apply_linear_relaxation(bounds, alpha_lower, beta_lower, alpha_upper, beta_upper)
