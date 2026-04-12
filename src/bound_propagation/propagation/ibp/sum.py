from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPSum(ForwardIBPStrategy):
    """IBP strategy for SUM operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"sum requires 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], IntervalBounds):
            raise TypeError("IBPSum requires the input to be an IntervalBounds")

        x_bounds: IntervalBounds = input_bounds[0]

        dim = node.attributes.get("dim")
        keep_dim = node.attributes.get("keepdim", False)

        lower = torch.sum(x_bounds.lower, dim, keepdim=keep_dim)
        upper = torch.sum(x_bounds.upper, dim, keepdim=keep_dim)

        return IntervalBounds(lower, upper)
