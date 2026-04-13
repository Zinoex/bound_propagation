from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPReshape(ForwardLBPStrategy):
    """
    Forward LBP strategy for RESHAPE operation.

    Reshape doesn't change values, only their shape.
    Linear coefficients remain unchanged, bias terms are reshaped.
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 1:
            raise ValueError(f"reshape requires exactly 1 input, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds):
            raise TypeError("ForwardLBPReshape requires input to be LinearBounds")

        bounds = input_bounds[0]

        # Get target shape from node attributes
        target_shape = node.attributes.get("shape")
        if target_shape is None:
            raise ValueError(f"RESHAPE node {node.id} missing 'shape' attribute")

        # Reshape bias terms (linear coefficients stay the same as they reference input)
        bias_lower = bounds.bias_lower.reshape(target_shape)
        bias_upper = bounds.bias_upper.reshape(target_shape)

        return LinearBounds(
            region=bounds.region,
            linear_lower=bounds.linear_lower,  # Same reference to input
            bias_lower=bias_lower,
            linear_upper=bounds.linear_upper,  # Same reference to input
            bias_upper=bias_upper,
        )
