from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPAbs(ForwardIBPStrategy):
    """
    IBP strategy for ABS operation:
    abs([a, b]) = if a < 0 and b > 0
        then [0, max(abs(a), abs(b))]
        else [min(abs(a), abs(b)), max(abs(a), abs(b))].
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"abs requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPAbs requires input to be IntervalBounds")

        abs_lower = x_bounds.lower.abs()
        abs_upper = x_bounds.upper.abs()

        zero = torch.zeros_like(abs_lower)
        lower = torch.where((x_bounds.lower < 0) & (x_bounds.upper > 0), zero, torch.min(abs_lower, abs_upper))
        upper = torch.max(abs_lower, abs_upper)

        return IntervalBounds(lower, upper)
