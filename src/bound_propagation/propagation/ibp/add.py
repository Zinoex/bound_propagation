from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPAdd(ForwardIBPStrategy):
    """IBP strategy for ADD operation: [a, b] + [c, d] = [a + c, b + d]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"add requires 2 inputs, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], IntervalBounds) or not isinstance(input_bounds[1], IntervalBounds):
            raise TypeError("IBPAdd requires both inputs to be IntervalBounds")

        x_bounds: IntervalBounds = input_bounds[0]
        y_bounds: IntervalBounds = input_bounds[1]

        # Interval addition
        lower = x_bounds.lower + y_bounds.lower
        upper = x_bounds.upper + y_bounds.upper

        return IntervalBounds(lower, upper)


class IBPAddWithConstant(ForwardIBPStrategy):
    """IBP strategy for ADD when at least one input is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"add requires 2 inputs, got {len(input_bounds)}")

        left = input_bounds[0]
        right = input_bounds[1]

        if isinstance(left, IntervalBounds):
            x, c = left, right
        elif isinstance(right, IntervalBounds):
            x, c = right, left
        else:
            raise TypeError(
                "IBPAddWithConstant requires one input to be IntervalBounds "
                "and the other input to be torch.Tensor or Number, "
                f"got {type(left)} and {type(right)}"
            )

        c = cast(torch.Tensor | torch.types.Number, c)

        # Add constant to interval
        lower = x.lower + c
        upper = x.upper + c

        return IntervalBounds(lower, upper)
