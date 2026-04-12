from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPClamp(ForwardIBPStrategy):
    """IBP strategy for CLAMP activation: clamp([a, b]) = [clamp(a), clamp(b)]."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"clamp requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]

        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPClamp requires input to be IntervalBounds")

        clamp_min = node.attributes.get("min", None)
        clamp_max = node.attributes.get("max", None)

        # clamps lower bound to 0 and keeps upper as is if positive
        lower = torch.clamp(x_bounds.lower, min=clamp_min, max=clamp_max)
        upper = torch.clamp(x_bounds.upper, min=clamp_min, max=clamp_max)

        return IntervalBounds(lower, upper)
