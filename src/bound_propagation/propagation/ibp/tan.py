from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPTan(ForwardIBPStrategy):
    """IBP strategy for TAN operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:

        if len(input_bounds) != 1:
            raise ValueError(f"tan requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPTan requires input to be IntervalBounds")

        # Tan is monotonical increasing between its asymptotes, so we can compute
        # the bounds by applying tan to the endpoints and use [-inf, inf] if the interval
        # contains an asymptote.
        lower = torch.tan(x_bounds.lower)
        upper = torch.tan(x_bounds.upper)

        # Check if the interval contains an asymptote of tan, which occurs at (2k+1)*pi/2 for k in Z
        pi_over_2 = torch.pi / 2
        eps = torch.finfo(x_bounds.lower.dtype).eps * 8
        k_min = torch.ceil((x_bounds.lower - pi_over_2 - eps) / torch.pi)
        k_max = torch.floor((x_bounds.upper - pi_over_2 + eps) / torch.pi)

        contains_asymptote = k_min <= k_max
        lower[contains_asymptote] = float("-inf")
        upper[contains_asymptote] = float("inf")

        return IntervalBounds(lower, upper)
