from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPUnsqueeze(ForwardIBPStrategy):
    """IBP strategy for UNSQUEEZE operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"UNSQUEEZE requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPUnsqueeze requires input to be IntervalBounds")

        dim = node.attributes.get("dim", None)

        if dim is None:
            raise ValueError("UNSQUEEZE requires 'dim' attribute")

        # Unsqueeze bounds
        lower = x_bounds.lower.unsqueeze(dim=dim)
        upper = x_bounds.upper.unsqueeze(dim=dim)

        return IntervalBounds(lower, upper)
