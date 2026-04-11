from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPMul(ForwardIBPStrategy):
    """IBP strategy for MUL operation: [a, b] * [c, d] = [min(a * c, a * d, b * c, b * d), max(a * c, a * d, b * c, b * d)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"mul requires 2 inputs, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], IntervalBounds) or not isinstance(input_bounds[1], IntervalBounds):
            raise TypeError("IBPMul requires both inputs to be IntervalBounds")

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

        return IntervalBounds(lower, upper)


class IBPMulWithConstant(ForwardIBPStrategy):
    """IBP strategy for MUL when at least one input is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"mul requires 2 inputs, got {len(input_bounds)}")

        x = input_bounds[0]
        y = input_bounds[1]

        if isinstance(x, IntervalBounds):
            interval = x
            c = y
        elif isinstance(y, IntervalBounds):
            interval = y
            c = x
        else:
            raise TypeError("IBPMulWithConstant requires at least one input to be IntervalBounds")

        if isinstance(c, torch.Tensor):
            lower = torch.where(
                c >= 0,
                interval.lower * c,
                interval.upper * c,
            )
            upper = torch.where(
                c >= 0,
                interval.upper * c,
                interval.lower * c,
            )
            return IntervalBounds(lower, upper)
        elif isinstance(c, torch.types.Number):
            if c >= 0:
                return IntervalBounds(interval.lower * c, interval.upper * c)
            else:
                return IntervalBounds(interval.upper * c, interval.lower * c)
        else:
            raise TypeError(f"Constant input must be torch.Tensor or Number, got {type(c)}")
