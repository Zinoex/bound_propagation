from __future__ import annotations

from typing import TYPE_CHECKING

import torch.fx as fx

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class ForwardLBPNeg(ForwardLBPStrategy):
    """Forward LBP strategy for negation."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> LinearBounds:
        args, kwargs = ctx.resolve_args(node)
        bounds = args[0]

        if not isinstance(bounds, LinearBounds):
            raise TypeError("ForwardLBPNeg requires input to be LinearBounds")

        return LinearBounds(
            regions=bounds.regions,
            linear_lower=[-linear for linear in bounds.linear_uppers],
            bias_lower=-bounds.bias_upper,
            linear_upper=[-linear for linear in bounds.linear_lowers],
            bias_upper=-bounds.bias_lower,
            input_ids=bounds.input_ids,
        )
