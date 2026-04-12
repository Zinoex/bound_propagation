from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPSub(ForwardIBPStrategy):
    """IBP strategy for SUB operation: [a, b] - [c, d] = [a - d, b - c]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"sub requires 2 inputs, got {len(input_bounds)}")

        x_bounds = input_bounds[0]
        y_bounds = input_bounds[1]

        if not isinstance(x_bounds, IntervalBounds) or not isinstance(y_bounds, IntervalBounds):
            raise TypeError("IBPSub requires both inputs to be IntervalBounds")

        # Interval subtraction
        lower = x_bounds.lower - y_bounds.upper
        upper = x_bounds.upper - y_bounds.lower

        return IntervalBounds(lower, upper)


class IBPSubConstantRight(ForwardIBPStrategy):
    """IBP strategy for SUB when the second input is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"sub requires 2 inputs, got {len(input_bounds)}")

        x = input_bounds[0]
        c = input_bounds[1]

        if not isinstance(x, IntervalBounds) or isinstance(c, IntervalBounds):
            raise TypeError("IBPSubConstantRight requires the first input to be IntervalBounds and the second input to be torch.Tensor or Number")

        lower = x.lower - c
        upper = x.upper - c

        return IntervalBounds(lower, upper)


class IBPSubConstantLeft(ForwardIBPStrategy):
    """IBP strategy for SUB when the first input is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"sub requires 2 inputs, got {len(input_bounds)}")

        c = input_bounds[0]
        x = input_bounds[1]

        if not isinstance(x, IntervalBounds) or isinstance(c, IntervalBounds):
            raise TypeError("IBPSubConstantLeft requires the second input to be IntervalBounds and the first input to be torch.Tensor or Number")

        lower = c - x.upper
        upper = c - x.lower

        return IntervalBounds(lower, upper)
