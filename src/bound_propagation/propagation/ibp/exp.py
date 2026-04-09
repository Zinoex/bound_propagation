from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import IntervalBoundingStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPExpStrategy(IntervalBoundingStrategy):
    """IBP strategy for EXP operation: exp([a, b]) = [exp(a), exp(b)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"EXP requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]

        # Exp is monotonic
        lower = torch.exp(x_bounds.lower)
        upper = torch.exp(x_bounds.upper)

        return IntervalBounds(x_bounds.region, lower, upper)
