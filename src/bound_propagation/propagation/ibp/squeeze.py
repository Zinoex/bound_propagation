from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPSqueeze(ForwardIBPStrategy):
    """IBP strategy for SQUEEZE operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"SQUEEZE requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSqueeze requires input to be IntervalBounds")

        dim = node.attributes.get("dim", None)

        # Squeeze bounds
        lower = x_bounds.lower.squeeze(dim=dim)
        upper = x_bounds.upper.squeeze(dim=dim)

        return IntervalBounds(lower, upper)
