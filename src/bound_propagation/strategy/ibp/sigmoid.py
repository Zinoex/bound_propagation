"""IBP strategy for SIGMOID activation."""

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


class IBPSigmoidStrategy(BoundingStrategy):
    """IBP strategy for SIGMOID activation: sigmoid([a,b]) = [sigmoid(a), sigmoid(b)]."""

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
            raise ValueError(f"SIGMOID requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]  # ty:ignore[invalid-assignment]

        # Sigmoid is monotonic, so just apply to bounds
        lower = torch.sigmoid(x_bounds.lower)
        upper = torch.sigmoid(x_bounds.upper)

        return IntervalBounds(x_bounds.region, lower, upper)
