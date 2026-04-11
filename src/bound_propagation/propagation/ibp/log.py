from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPLog(ForwardIBPStrategy):
    """IBP strategy for LOG operation: log([a, b]) = [log(a), log(b)] for a, b > 0."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:

        if len(input_bounds) != 1:
            raise ValueError(f"log requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPLog requires input to be IntervalBounds")

        # Check for non-positive inputs
        if torch.any(x_bounds.lower <= 0):
            raise ValueError("log requires positive input bounds")

        # Log is monotonic for positive inputs
        lower = torch.log(x_bounds.lower)
        upper = torch.log(x_bounds.upper)

        return IntervalBounds(lower, upper)
