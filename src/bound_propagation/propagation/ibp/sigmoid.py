from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import IntervalBoundingStrategy
from .utils import verify_interval_bounds

if TYPE_CHECKING:
    from ...ir import Node


class IBPSigmoidStrategy(IntervalBoundingStrategy):
    """IBP strategy for SIGMOID activation: sigmoid([a, b]) = [sigmoid(a), sigmoid(b)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds],
    ) -> IntervalBounds:
        verify_interval_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"SIGMOID requires 1 input, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]

        # Sigmoid is monotonic, so just apply to bounds
        lower = torch.sigmoid(x_bounds.lower)
        upper = torch.sigmoid(x_bounds.upper)

        return IntervalBounds(x_bounds.region, lower, upper)
