from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPSqrt(ForwardIBPStrategy):
    """IBP strategy for SQRT operation: sqrt([a, b]) = [sqrt(a), sqrt(b)] for a, b >= 0."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:

        if len(input_bounds) != 1:
            raise ValueError(f"sqrt requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSqrt requires input to be IntervalBounds")

        # Check for non-positive inputs
        if torch.any(x_bounds.lower < 0):
            raise ValueError("sqrt requires non-negative input bounds")

        # Sqrt is monotonic for non-negative inputs
        lower = torch.sqrt(x_bounds.lower)
        upper = torch.sqrt(x_bounds.upper)

        return IntervalBounds(lower, upper)
