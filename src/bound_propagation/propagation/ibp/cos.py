from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPCos(ForwardIBPStrategy):
    """IBP strategy for cos with peak/trough analysis."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPCos requires input to be IntervalBounds")

        two_pi = 2 * torch.pi
        pi = torch.pi

        includes_peak = torch.floor(x_bounds.upper / two_pi) >= torch.ceil(x_bounds.lower / two_pi)

        includes_trough = torch.floor((x_bounds.upper - pi) / two_pi) >= torch.ceil((x_bounds.lower - pi) / two_pi)

        cos_lower = torch.cos(x_bounds.lower)
        cos_upper = torch.cos(x_bounds.upper)

        lower = torch.where(
            includes_trough,
            torch.tensor(-1.0, device=x_bounds.lower.device, dtype=x_bounds.lower.dtype),
            torch.min(cos_lower, cos_upper),
        )
        upper = torch.where(
            includes_peak,
            torch.tensor(1.0, device=x_bounds.upper.device, dtype=x_bounds.upper.dtype),
            torch.max(cos_lower, cos_upper),
        )

        return IntervalBounds(lower, upper)
