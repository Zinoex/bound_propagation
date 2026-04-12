from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPTranspose(ForwardIBPStrategy):
    """IBP strategy for TRANSPOSE operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"transpose requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]
        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPTranspose requires the input to be an IntervalBounds")

        dim0 = node.attributes.get("dim0")
        dim1 = node.attributes.get("dim1")

        if dim0 is None or dim1 is None:
            raise ValueError("transpose requires 'dim   0' and 'dim1' attributes")

        # Interval transpose
        lower = x_bounds.lower.transpose(dim0, dim1)
        upper = x_bounds.upper.transpose(dim0, dim1)

        return IntervalBounds(lower, upper)
