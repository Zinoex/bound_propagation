from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPSum(ForwardLBPStrategy):
    """Forward LBP strategy for SUM operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"sum requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPSum requires input to be LinearBounds")

        bounds = input_bounds[0]

        dim = node.attributes.get("dim")
        keep_dim = node.attributes.get("keepdim", False)

        # Sum is a linear operation, so we can preserve linear structure
        # For simplicity, concretize and apply sum
        lower, upper = bounds.concretize()

        lower = lower.sum(dim, keepdim=keep_dim)
        upper = upper.sum(dim, keepdim=keep_dim)

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
