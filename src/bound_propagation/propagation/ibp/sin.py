from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.fx as fx

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ..context import PropagationContext


class IBPSin(ForwardIBPStrategy):
    """IBP strategy for sin with peak/trough analysis."""

    def propagate_forward(
        self,
        node: fx.Node,
        ctx: PropagationContext,
    ) -> IntervalBounds:
        args, kwargs = ctx.resolve_args(node)
        x_bounds = args[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSin requires input to be IntervalBounds")

        two_pi = 2 * torch.pi
        pi = torch.pi
        pi_over_2 = pi / 2
        three_pi_over_2 = 3 * pi / 2

        includes_peak = torch.floor((x_bounds.upper - pi_over_2) / two_pi) >= torch.ceil(
            (x_bounds.lower - pi_over_2) / two_pi
        )

        includes_trough = torch.floor((x_bounds.upper - three_pi_over_2) / two_pi) >= torch.ceil(
            (x_bounds.lower - three_pi_over_2) / two_pi
        )

        sin_lower = torch.sin(x_bounds.lower)
        sin_upper = torch.sin(x_bounds.upper)

        lower = torch.where(
            includes_trough,
            torch.tensor(-1.0, device=x_bounds.lower.device, dtype=x_bounds.lower.dtype),
            torch.min(sin_lower, sin_upper),
        )
        upper = torch.where(
            includes_peak,
            torch.tensor(1.0, device=x_bounds.upper.device, dtype=x_bounds.upper.dtype),
            torch.max(sin_lower, sin_upper),
        )

        return IntervalBounds(lower, upper)
