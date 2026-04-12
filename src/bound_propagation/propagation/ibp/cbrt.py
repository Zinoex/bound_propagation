from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPCbrt(ForwardIBPStrategy):
    """IBP strategy for CBRt operation: cbrt([a, b]) = [cbrt(a), cbrt(b)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"cbrt requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPCbrt requires input to be IntervalBounds")

        # Cbrt is monotonic; however, we need to handle negative values correctly
        # since PyTorch's pow with fractional exponents can return NaN for negative bases.
        # We can use torch.copysign to preserve the sign after taking the cube root of the absolute value.
        lower = torch.copysign(torch.pow(x_bounds.lower.abs(), 1 / 3), x_bounds.lower)
        upper = torch.copysign(torch.pow(x_bounds.upper.abs(), 1 / 3), x_bounds.upper)

        return IntervalBounds(lower, upper)
