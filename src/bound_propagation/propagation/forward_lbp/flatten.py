from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPFlattenStrategy(ForwardLBPStrategy):
    """
    Forward LBP strategy for FLATTEN operation.

    Flatten converts to 1D shape without changing values.
    Linear coefficients remain unchanged, bias terms are flattened.
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"flatten requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPFlattenStrategy requires input to be LinearBounds")

        bounds = input_bounds[0]

        # Flatten bias terms
        bias_lower = bounds.bias_lower.flatten()
        bias_upper = bounds.bias_upper.flatten()

        return LinearBounds(
            region=bounds.region,
            linear_lower=bounds.linear_lower,  # Same reference to input
            bias_lower=bias_lower,
            linear_upper=bounds.linear_upper,  # Same reference to input
            bias_upper=bias_upper,
        )
