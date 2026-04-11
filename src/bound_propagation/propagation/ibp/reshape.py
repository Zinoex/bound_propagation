from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPReshape(ForwardIBPStrategy):
    """IBP strategy for RESHAPE operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"RESHAPE requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPReshape requires input to be IntervalBounds")

        # Get target shape from node attributes
        target_shape = node.attributes.get("shape")
        if target_shape is None:
            raise ValueError("reshape node missing shape attribute")

        # Reshape bounds
        lower = x_bounds.lower.reshape(target_shape)
        upper = x_bounds.upper.reshape(target_shape)

        return IntervalBounds(lower, upper)
