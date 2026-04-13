from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPMaxStrategy(ForwardLBPStrategy):
    """Forward LBP strategy for MAX reduction operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"MAX requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPMaxStrategy requires input to be LinearBounds")

        bounds = input_bounds[0]

        dim = node.attributes.get("dim")
        keep_dim = node.attributes.get("keepdim", False)

        # Concretize and apply max
        lower, upper = bounds.concretize()

        if dim is not None:
            lower = lower.max(dim=dim, keepdim=keep_dim).values
            upper = upper.max(dim=dim, keepdim=keep_dim).values
        else:
            lower = lower.max()
            upper = upper.max()

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
