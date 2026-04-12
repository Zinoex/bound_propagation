from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPMean(ForwardIBPStrategy):
    """IBP strategy for MEAN operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"mean requires 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], IntervalBounds):
            raise TypeError("IBPMean requires the input to be an IntervalBounds")

        x_bounds: IntervalBounds = input_bounds[0]
        dim = node.attributes.get("dim", 0)

        # Interval mean
        lower = x_bounds.lower.mean(dim=dim)
        upper = x_bounds.upper.mean(dim=dim)

        return IntervalBounds(lower, upper)
