"""IBP strategy for DIV operation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from ..strategy import BoundingStrategy
from .utils import verify_interval_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class IBPDivStrategy(BoundingStrategy):
    """IBP strategy for DIV operation: [a,b] / [c,d]."""

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
            raise ValueError(f"DIV requires 2 inputs, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]  # ty:ignore[invalid-assignment]
        y_bounds: IntervalBounds = input_bounds[1]  # ty:ignore[invalid-assignment]

        # Check if divisor can be zero
        if torch.any((y_bounds.lower <= 0) & (y_bounds.upper >= 0)):
            # Division by interval containing zero - return unbounded
            return IntervalBounds.unbounded(x_bounds.shape)

        # Compute all four quotients
        ll = x_bounds.lower / y_bounds.lower
        lu = x_bounds.lower / y_bounds.upper
        ul = x_bounds.upper / y_bounds.lower
        uu = x_bounds.upper / y_bounds.upper

        # Take min and max
        lower = torch.min(torch.min(ll, lu), torch.min(ul, uu))
        upper = torch.max(torch.max(ll, lu), torch.max(ul, uu))

        return IntervalBounds(x_bounds.region, lower, upper)
