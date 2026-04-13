from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPMinStrategy(ForwardLBPStrategy):
    """Forward LBP strategy for MIN reduction operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"MIN requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPMinStrategy requires input to be LinearBounds")

        bounds = input_bounds[0]

        dim = node.attributes.get("dim")
        keep_dim = node.attributes.get("keepdim", False)

        # Concretize and apply min
        lower, upper = bounds.concretize()

        if dim is not None:
            lower = lower.min(dim=dim, keepdim=keep_dim).values
            upper = upper.min(dim=dim, keepdim=keep_dim).values
        else:
            lower = lower.min()
            upper = upper.min()

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
