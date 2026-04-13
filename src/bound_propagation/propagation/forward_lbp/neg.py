from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPNegStrategy(ForwardLBPStrategy):
    """Forward LBP strategy for NEG operation (negation)."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"NEG requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPNegStrategy requires input to be LinearBounds")

        bounds: LinearBounds = input_bounds[0]

        # Negation flips lower and upper bounds and their linear coefficients
        return LinearBounds(
            region=bounds.region,
            linear_lower=-bounds.linear_upper if bounds.linear_upper is not None else None,
            bias_lower=-bounds.bias_upper,
            linear_upper=-bounds.linear_lower if bounds.linear_lower is not None else None,
            bias_upper=-bounds.bias_lower,
        )
