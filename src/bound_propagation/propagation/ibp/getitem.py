from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPGetItem(ForwardIBPStrategy):
    """IBP strategy for GETITEM operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"getitem requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]
        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPGetItem requires the input to be an IntervalBounds")

        item = node.attributes.get("item")

        if item is None:
            raise ValueError("getitem requires 'item' attribute")

        # Interval getitem
        return x_bounds[item]
