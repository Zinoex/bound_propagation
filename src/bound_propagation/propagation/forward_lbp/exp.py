from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from ..linear_relaxations.exp import compute_exp_alpha_beta
from .base import ForwardLBPStrategy
from .utils import apply_linear_relaxation

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPExp(ForwardLBPStrategy):
    """Forward LBP strategy for exp using linear relaxation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPExp requires input to be LinearBounds")

        lower, upper = bounds.concretize()
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_exp_alpha_beta(lower, upper)
        return apply_linear_relaxation(bounds, alpha_lower, beta_lower, alpha_upper, beta_upper)
