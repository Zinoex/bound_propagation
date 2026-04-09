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


class IBPExpStrategy(ForwardBoundingStrategy):
    """IBP strategy for EXP operation: exp([a,b]) = [exp(a), exp(b)]."""

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
            raise ValueError(f"EXP requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]  # ty:ignore[invalid-assignment]

        # Exp is monotonic
        lower = torch.exp(x_bounds.lower)
        upper = torch.exp(x_bounds.upper)

        return IntervalBounds(x_bounds.region, lower, upper)
