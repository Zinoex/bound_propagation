from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPView(ForwardIBPStrategy):
    """IBP strategy for VIEW operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"view requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPView requires input to be IntervalBounds")

        # Get target shape from node attributes
        size = node.attributes.get("size")
        if size is None:
            raise ValueError("view node missing size attribute")

        # View bounds
        lower = x_bounds.lower.view(size)
        upper = x_bounds.upper.view(size)

        return IntervalBounds(lower, upper)
