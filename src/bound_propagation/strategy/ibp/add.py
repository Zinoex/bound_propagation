"""IBP strategy for ADD operation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import IntervalBounds
from ..strategy import BoundingStrategy
from .utils import verify_interval_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class IBPAddStrategy(BoundingStrategy):
    """IBP strategy for ADD operation: [a,b] + [c,d] = [a+c, b+d]."""

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

        if len(input_bounds) != 2:
            raise ValueError(f"ADD requires 2 inputs, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]  # ty:ignore[invalid-assignment]
        y_bounds: IntervalBounds = input_bounds[1]  # ty:ignore[invalid-assignment]

        # Interval addition
        lower = x_bounds.lower + y_bounds.lower
        upper = x_bounds.upper + y_bounds.upper

        return IntervalBounds(x_bounds.region, lower, upper)
