from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPUnsqueeze(ForwardLBPStrategy):
    """Forward LBP strategy for UNSQUEEZE operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"unsqueeze requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPUnsqueeze requires input to be LinearBounds")

        bounds = input_bounds[0]
        dim = node.attributes.get("dim", 0)

        # Unsqueeze preserves linear structure
        linear_lower = bounds.linear_lower.unsqueeze(dim) if bounds.linear_lower is not None else None
        linear_upper = bounds.linear_upper.unsqueeze(dim) if bounds.linear_upper is not None else None
        bias_lower = bounds.bias_lower.unsqueeze(dim)
        bias_upper = bounds.bias_upper.unsqueeze(dim)

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
