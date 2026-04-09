from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPSubStrategy(ForwardIBPStrategy):
    """IBP strategy for SUB operation: [a, b] - [c, d] = [a - d, b - c]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"SUB requires 2 inputs, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]
        y_bounds: IntervalBounds = input_bounds[1]

        # Interval subtraction
        lower = x_bounds.lower - y_bounds.upper
        upper = x_bounds.upper - y_bounds.lower

        return IntervalBounds(x_bounds.region, lower, upper)
