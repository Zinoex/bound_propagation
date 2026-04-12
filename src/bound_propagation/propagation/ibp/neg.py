from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPNeg(ForwardIBPStrategy):
    """IBP strategy for NEG operation: -[a, b] = [-b, -a]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"neg requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]
        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPNeg requires the input to be an IntervalBounds")

        # Interval negation
        lower = -x_bounds.upper
        upper = -x_bounds.lower

        return IntervalBounds(lower, upper)
