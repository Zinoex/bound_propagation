from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPMinimum(ForwardIBPStrategy):
    """IBP strategy for MIN operation: min([a, b], [c, d]) = [min(a, c), min(b, d)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"minimum requires 2 inputs, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], IntervalBounds) or not isinstance(input_bounds[1], IntervalBounds):
            raise TypeError("IBPMinimum requires both inputs to be IntervalBounds")

        x_bounds: IntervalBounds = input_bounds[0]
        y_bounds: IntervalBounds = input_bounds[1]

        # Interval
        lower = torch.min(x_bounds.lower, y_bounds.lower)
        upper = torch.min(x_bounds.upper, y_bounds.upper)

        return IntervalBounds(lower, upper)


class IBPMinimumWithConstant(ForwardIBPStrategy):
    """IBP strategy for MIN when at least one input is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"minimum requires 2 inputs, got {len(input_bounds)}")

        left = input_bounds[0]
        right = input_bounds[1]

        if isinstance(left, IntervalBounds):
            x, c = left, right
        elif isinstance(right, IntervalBounds):
            x, c = right, left
        else:
            raise TypeError(
                f"IBPMinimumWithConstant requires one input to be IntervalBounds "
                f"the other input to be torch.Tensor, got {type(left)} and {type(right)}"
            )

        if not isinstance(c, torch.Tensor):
            raise TypeError(f"IBPMinimumWithConstant requires the constant input to be a torch.Tensor, got {type(c)}")

        # Min constant to interval
        lower = torch.min(x.lower, c)
        upper = torch.min(x.upper, c)

        return IntervalBounds(lower, upper)
