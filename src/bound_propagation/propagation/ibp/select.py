from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import IntervalBounds
from .base import ForwardIBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class IBPSelect(ForwardIBPStrategy):
    """IBP strategy for SELECT operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[IntervalBounds | torch.Tensor | torch.types.Number],
    ) -> IntervalBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"select requires 1 input, got {len(input_bounds)}")

        x_bounds = input_bounds[0]
        if not isinstance(x_bounds, IntervalBounds):
            raise TypeError("IBPSelect requires the input to be an IntervalBounds")

        index = node.attributes.get("index")
        dim = node.attributes.get("dim", 0)

        if index is None or dim is None:
            raise ValueError("select requires 'index' and 'dim' attributes")

        # Interval select
        return IntervalBounds(x_bounds.lower.select(dim=dim, index=index), x_bounds.upper.select(dim=dim, index=index))
