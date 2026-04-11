from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPFlatten(ForwardIBPStrategy):
    """IBP strategy for FLATTEN operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"flatten requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPFlatten requires input to be IntervalBounds")

        # Flatten all dimensions
        lower = x_bounds.lower.flatten()
        upper = x_bounds.upper.flatten()

        return IntervalBounds(lower, upper)
