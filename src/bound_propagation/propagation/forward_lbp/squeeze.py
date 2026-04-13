from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPSqueezeStrategy(ForwardLBPStrategy):
    """Forward LBP strategy for SQUEEZE operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"SQUEEZE requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPSqueezeStrategy requires input to be LinearBounds")

        bounds: LinearBounds = input_bounds[0]
        dim = node.attributes.get("dim")

        # Squeeze preserves linear structure
        if dim is not None:
            linear_lower = bounds.linear_lower.squeeze(dim) if bounds.linear_lower is not None else None
            linear_upper = bounds.linear_upper.squeeze(dim) if bounds.linear_upper is not None else None
            bias_lower = bounds.bias_lower.squeeze(dim)
            bias_upper = bounds.bias_upper.squeeze(dim)
        else:
            linear_lower = bounds.linear_lower.squeeze() if bounds.linear_lower is not None else None
            linear_upper = bounds.linear_upper.squeeze() if bounds.linear_upper is not None else None
            bias_lower = bounds.bias_lower.squeeze()
            bias_upper = bounds.bias_upper.squeeze()

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
