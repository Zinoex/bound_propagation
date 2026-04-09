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


class IBPMulStrategy(BoundingStrategy):
    """IBP strategy for MUL operation: [a,b] * [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]."""

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
            raise ValueError(f"MUL requires 2 inputs, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]  # ty:ignore[invalid-assignment]
        y_bounds: IntervalBounds = input_bounds[1]  # ty:ignore[invalid-assignment]

        # Compute all four products
        ll = x_bounds.lower * y_bounds.lower
        lu = x_bounds.lower * y_bounds.upper
        ul = x_bounds.upper * y_bounds.lower
        uu = x_bounds.upper * y_bounds.upper

        # Take min and max across all products
        lower = torch.min(torch.min(ll, lu), torch.min(ul, uu))
        upper = torch.max(torch.max(ll, lu), torch.max(ul, uu))

        return IntervalBounds(x_bounds.region, lower, upper)
