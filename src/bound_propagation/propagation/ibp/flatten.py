from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPFlattenStrategy(ForwardIBPStrategy):
    """IBP strategy for FLATTEN operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"FLATTEN requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]

        # Flatten all dimensions
        lower = x_bounds.lower.flatten()
        upper = x_bounds.upper.flatten()

        return IntervalBounds(lower, upper)
