from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPSigmoid(ForwardIBPStrategy):
    """IBP strategy for SIGMOID activation: sigmoid([a, b]) = [sigmoid(a), sigmoid(b)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"sigmoid requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSigmoid requires input to be IntervalBounds")

        # Sigmoid is monotonic, so just apply to bounds
        lower = torch.sigmoid(x_bounds.lower)
        upper = torch.sigmoid(x_bounds.upper)

        return IntervalBounds(lower, upper)
