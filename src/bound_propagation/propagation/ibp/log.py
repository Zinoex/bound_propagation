from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPLogStrategy(ForwardIBPStrategy):
    """IBP strategy for LOG operation: log([a, b]) = [log(a), log(b)] for a, b > 0."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds],
    ) -> IntervalBounds:

        if len(input_bounds) != 1:
            raise ValueError(f"LOG requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]

        # Check for non-positive inputs
        if torch.any(x_bounds.lower <= 0):
            raise ValueError("LOG requires positive input bounds")

        # Log is monotonic for positive inputs
        lower = torch.log(x_bounds.lower)
        upper = torch.log(x_bounds.upper)

        return IntervalBounds(x_bounds.region, lower, upper)
