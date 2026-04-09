from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import IntervalBounds
from .base import IntervalBoundingStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPReshapeStrategy(IntervalBoundingStrategy):
    """IBP strategy for RESHAPE operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"RESHAPE requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]

        # Get target shape from node attributes
        target_shape = node.attributes.get("shape")
        if target_shape is None:
            raise ValueError("RESHAPE node missing shape attribute")

        # Reshape bounds
        lower = x_bounds.lower.reshape(target_shape)
        upper = x_bounds.upper.reshape(target_shape)

        return IntervalBounds(x_bounds.region, lower, upper)
