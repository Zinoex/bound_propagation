from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import IntervalBounds
from .base import IntervalBoundingStrategy
from .utils import verify_interval_bounds

if TYPE_CHECKING:
    from ...ir import Node


class IBPAddStrategy(IntervalBoundingStrategy):
    """IBP strategy for ADD operation: [a, b] + [c, d] = [a + c, b + d]."""

    def compute_bounds(
        self,
        node: Node,
        input_bounds: list[IntervalBounds],
    ) -> IntervalBounds:
        verify_interval_bounds(input_bounds)

        if len(input_bounds) != 2:
            raise ValueError(f"ADD requires 2 inputs, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]
        y_bounds: IntervalBounds = input_bounds[1]

        # Interval addition
        lower = x_bounds.lower + y_bounds.lower
        upper = x_bounds.upper + y_bounds.upper

        return IntervalBounds(x_bounds.region, lower, upper)
