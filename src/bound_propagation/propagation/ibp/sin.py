from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPSin(ForwardIBPStrategy):
    """IBP strategy for SIN operation: sin([a, b]) = [sin(a), sin(b)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"sin requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSin requires input to be IntervalBounds")

        # Analyze regimes (by the batch) based on the period of sine
        # Lower:
        # - if it includes a trough (2k+1)*pi/2, then the minimum is -1
        # - else if it includes a peak 2k*pi, then the minimum is minimum of sin(lower) and sin(upper)

        # Upper:
        # - if it includes a peak 2k*pi, then the maximum is 1
        # - else if it includes a trough (2k+1)*pi/2, then the maximum is maximum of sin(lower) and sin(upper)

        two_pi = 2 * torch.pi
        lower_mod = torch.remainder(x_bounds.lower, two_pi)
        upper_mod = torch.remainder(x_bounds.upper, two_pi)
        includes_peak = (lower_mod <= 0) & (upper_mod >= 0) | (lower_mod <= two_pi) & (upper_mod >= two_pi)
        includes_trough = (lower_mod <= torch.pi / 2) & (upper_mod >= torch.pi / 2)

        sin_lower = torch.sin(x_bounds.lower)
        sin_upper = torch.sin(x_bounds.upper)

        lower = torch.where(
            includes_trough,
            torch.tensor(-1.0, device=x_bounds.lower.device),
            torch.min(sin_lower, sin_upper),
        )
        upper = torch.where(
            includes_peak,
            torch.tensor(1.0, device=x_bounds.lower.device),
            torch.max(sin_lower, sin_upper),
        )

        return IntervalBounds(lower, upper)
