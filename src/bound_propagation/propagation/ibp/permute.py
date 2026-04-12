from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPPermute(ForwardIBPStrategy):
    """IBP strategy for PERMUTE operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"permute requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]
        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPPermute requires the input to be an IntervalBounds")

        dims = node.attributes.get("dims")

        if dims is None:
            raise ValueError("permute requires 'dims' attribute")

        # Interval permute
        lower = x_bounds.lower.permute(dims)
        upper = x_bounds.upper.permute(dims)

        return IntervalBounds(lower, upper)
