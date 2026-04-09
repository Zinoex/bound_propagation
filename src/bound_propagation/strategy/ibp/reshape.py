from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import IntervalBounds
from ..strategy import ForwardBoundingStrategy
from .utils import verify_interval_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class IBPReshapeStrategy(ForwardBoundingStrategy):
    """IBP strategy for RESHAPE operation."""

    @property
    def method_name(self) -> str:
        return "ibp"

    def compute_bounds(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> AbstractBounds:
        verify_interval_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"RESHAPE requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]  # ty:ignore[invalid-assignment]

        # Get target shape from node attributes
        target_shape = node.attributes.get("shape")
        if target_shape is None:
            raise ValueError("RESHAPE node missing shape attribute")

        # Reshape bounds
        lower = x_bounds.lower.reshape(target_shape)
        upper = x_bounds.upper.reshape(target_shape)

        return IntervalBounds(x_bounds.region, lower, upper)
