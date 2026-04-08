"""IBP strategy for FLATTEN operation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import IntervalBounds
from ..strategy import BoundingStrategy
from .utils import verify_interval_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class IBPFlattenStrategy(BoundingStrategy):
    """IBP strategy for FLATTEN operation."""

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
            raise ValueError(f"FLATTEN requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]  # ty:ignore[invalid-assignment]

        # Flatten all dimensions
        lower = x_bounds.lower.flatten()
        upper = x_bounds.upper.flatten()

        return IntervalBounds(x_bounds.region, lower, upper)
