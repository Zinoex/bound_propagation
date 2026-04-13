from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPMeanStrategy(ForwardLBPStrategy):
    """Forward LBP strategy for MEAN operation."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"MEAN requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPMeanStrategy requires input to be LinearBounds")

        bounds = input_bounds[0]
        dim = node.attributes.get("dim", 0)

        # Mean is a linear operation, so we can preserve linear structure
        # For simplicity, concretize and apply mean
        lower, upper = bounds.concretize()

        lower = lower.mean(dim=dim)
        upper = upper.mean(dim=dim)

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
