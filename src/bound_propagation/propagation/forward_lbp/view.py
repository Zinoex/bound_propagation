from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPViewStrategy(ForwardLBPStrategy):
    """Forward LBP strategy for VIEW operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"VIEW requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPViewStrategy requires input to be LinearBounds")

        bounds: LinearBounds = input_bounds[0]
        shape = node.attributes.get("shape")

        # View preserves linear structure
        linear_lower = bounds.linear_lower.view(*shape) if bounds.linear_lower is not None else None
        linear_upper = bounds.linear_upper.view(*shape) if bounds.linear_upper is not None else None
        bias_lower = bounds.bias_lower.view(*shape)
        bias_upper = bounds.bias_upper.view(*shape)

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
