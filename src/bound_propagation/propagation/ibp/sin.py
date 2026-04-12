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

        # Analyze regimes based on the period of sine
        # Peak (max = 1) occurs at π/2 + 2k*π for integer k
        # Trough (min = -1) occurs at 3π/2 + 2k*π for integer k
        #
        # An interval [a, b] contains a peak if: floor((b - π/2) / 2π) >= ceil((a - π/2) / 2π)
        # An interval [a, b] contains a trough if: floor((b - 3π/2) / 2π) >= ceil((a - 3π/2) / 2π)

        two_pi = 2 * torch.pi
        pi = torch.pi
        pi_over_2 = pi / 2
        three_pi_over_2 = 3 * pi / 2

        # Check for peak: does [a, b] contain any π/2 + 2k*π?
        includes_peak = torch.floor((x_bounds.upper - pi_over_2) / two_pi) >= torch.ceil(
            (x_bounds.lower - pi_over_2) / two_pi
        )

        # Check for trough: does [a, b] contain any 3π/2 + 2k*π?
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
