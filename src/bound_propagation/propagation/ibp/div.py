from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPDivStrategy(ForwardIBPStrategy):
    """IBP strategy for DIV operation: [a, b] / [c, d]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"DIV requires 2 inputs, got {len(input_bounds)}")

        x_bounds: IntervalBounds = input_bounds[0]
        y_bounds: IntervalBounds = input_bounds[1]

        # Check if divisor can be zero
        if torch.any((y_bounds.lower <= 0) & (y_bounds.upper >= 0)):
            # Division by interval containing zero - return unbounded
            return IntervalBounds.unbounded_like(x_bounds)

        # Compute all four quotients
        ll = x_bounds.lower / y_bounds.lower
        lu = x_bounds.lower / y_bounds.upper
        ul = x_bounds.upper / y_bounds.lower
        uu = x_bounds.upper / y_bounds.upper

        # Take min and max
        lower = torch.min(torch.min(ll, lu), torch.min(ul, uu))
        upper = torch.max(torch.max(ll, lu), torch.max(ul, uu))

        return IntervalBounds(lower, upper)
