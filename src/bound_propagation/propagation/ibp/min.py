from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPMin(ForwardIBPStrategy):
    """IBP strategy for MIN operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"min requires 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], IntervalBounds):
            raise TypeError("IBPMin requires the input to be an IntervalBounds")

        x_bounds: IntervalBounds = input_bounds[0]

        dim = node.attributes.get("dim")
        if dim is not None:
            # If dim is specified, we take the min across that dimension
            lower = torch.min(x_bounds.lower, dim=dim).values
            upper = torch.min(x_bounds.upper, dim=dim).values
        else:
            # Interval min
            lower = torch.min(x_bounds.lower)
            upper = torch.min(x_bounds.upper)

        return IntervalBounds(lower, upper)
