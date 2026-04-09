from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPMulStrategy(ForwardIBPStrategy):
    """IBP strategy for MUL operation: [a, b] * [c, d] = [min(a * c, a * d, b * c, b * d), max(a * c, a * d, b * c, b * d)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"MUL requires 2 inputs, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]
        y_bounds: IntervalBounds = input_bounds[1]

        # Compute all four products
        ll = x_bounds.lower * y_bounds.lower
        lu = x_bounds.lower * y_bounds.upper
        ul = x_bounds.upper * y_bounds.lower
        uu = x_bounds.upper * y_bounds.upper

        # Take min and max across all products
        lower = torch.min(torch.min(ll, lu), torch.min(ul, uu))
        upper = torch.max(torch.max(ll, lu), torch.max(ul, uu))

        return IntervalBounds(x_bounds.region, lower, upper)
