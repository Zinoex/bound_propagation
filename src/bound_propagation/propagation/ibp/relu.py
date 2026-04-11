from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node

class IBPRelu(ForwardIBPStrategy):
    """IBP strategy for RELU activation: relu([a, b]) = [max(0, a), max(0, b)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"relu requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPRelu requires input to be IntervalBounds")

        # ReLU clamps lower bound to 0 and keeps upper as is if positive
        lower = torch.clamp(x_bounds.lower, min=0.0)
        upper = torch.clamp(x_bounds.upper, min=0.0)

        return IntervalBounds(lower, upper)
