from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from ..strategy import ForwardBoundingStrategy
from .utils import verify_interval_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class IBPLogStrategy(ForwardBoundingStrategy):
    """IBP strategy for LOG operation: log([a,b]) = [log(a), log(b)] for a,b > 0."""

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
            raise ValueError(f"LOG requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]  # ty:ignore[invalid-assignment]

        # Check for non-positive inputs
        if torch.any(x_bounds.lower <= 0):
            raise ValueError("LOG requires positive input bounds")

        # Log is monotonic for positive inputs
        lower = torch.log(x_bounds.lower)
        upper = torch.log(x_bounds.upper)

        return IntervalBounds(x_bounds.region, lower, upper)
