from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPMaximum(ForwardLBPStrategy):
    """Forward LBP strategy for MAXIMUM operation (element-wise max of two abstract inputs)."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"maximum requires exactly 2 inputs, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds) or not isinstance(input_bounds[1], LinearBounds):
            raise TypeError("ForwardLBPMaximum requires both inputs to be LinearBounds")

        bounds_a: LinearBounds = input_bounds[0]
        bounds_b: LinearBounds = input_bounds[1]

        # Concretize both bounds
        lower_a, upper_a = bounds_a.concretize()
        lower_b, upper_b = bounds_b.concretize()

        # Element-wise maximum bounds
        lower = torch.maximum(lower_a, lower_b)
        upper = torch.maximum(upper_a, upper_b)

        # Return as constant LinearBounds (lose linear dependency)
        return LinearBounds(
            region=bounds_a.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )


class ForwardLBPMaximumWithConstant(ForwardLBPStrategy):
    """Forward LBP strategy for MAXIMUM when at least one input is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"maximum requires exactly 2 inputs, got {len(input_bounds)}")

        left = input_bounds[0]
        right = input_bounds[1]

        if isinstance(left, LinearBounds):
            bounds, constant = left, right
        elif isinstance(right, LinearBounds):
            bounds, constant = right, left
        else:
            raise TypeError(
                f"ForwardLBPMaximumWithConstant requires one input to be LinearBounds and the other to be "
                f"torch.Tensor or Number, got {type(left)} and {type(right)}"
            )

        # Concretize bounds
        lower, upper = bounds.concretize()

        # Element-wise maximum with constant
        lower = torch.maximum(lower, constant)
        upper = torch.maximum(upper, constant)

        # Return as constant LinearBounds
        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower,
            linear_upper=None,
            bias_upper=upper,
        )
