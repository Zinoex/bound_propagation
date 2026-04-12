from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPMax(ForwardIBPStrategy):
    """IBP strategy for MEAN operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"max requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPMax requires the input to be an IntervalBounds")

        dim = node.attributes.get("dim")
        if dim is not None:
            # If dim is specified, we take the max across that dimension
            lower = torch.max(x_bounds.lower, dim=dim).values
            upper = torch.max(x_bounds.upper, dim=dim).values
        else:
            # Otherwise, we take the elementwise max
            lower = torch.max(x_bounds.lower)
            upper = torch.max(x_bounds.upper)

        return IntervalBounds(lower, upper)
