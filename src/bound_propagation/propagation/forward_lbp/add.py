from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    from ...ir import Node


class ForwardLBPAdd(ForwardLBPStrategy):
    """
    Forward LBP strategy for ADD operation.

    For addition z = x + y:
    - Lower: W_l^z @ x0 + b_l^z = W_l^x @ x0 + b_l^x + W_l^y @ x0 + b_l^y
    - Upper: W_u^z @ x0 + b_u^z = W_u^x @ x0 + b_u^x + W_u^y @ x0 + b_u^y

    This is exact (no relaxation needed for linear operations).
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        """
        Compute forward LBP bounds for addition.

        Args:
            node: The add node
            input_bounds: List of two LinearBounds for the operands

        Returns:
            LinearBounds for the sum
        """
        if len(input_bounds) != 2:
            raise ValueError(f"add requires exactly 2 inputs, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds) or not isinstance(input_bounds[1], LinearBounds):
            raise TypeError("ForwardLBPAdd requires both inputs to be LinearBounds")

        bounds_a = input_bounds[0]
        bounds_b = input_bounds[1]

        # Add linear coefficients
        if bounds_a.linear_lower is not None and bounds_b.linear_lower is not None:
            linear_lower = bounds_a.linear_lower + bounds_b.linear_lower
        elif bounds_a.linear_lower is not None:
            linear_lower = bounds_a.linear_lower
        elif bounds_b.linear_lower is not None:
            linear_lower = bounds_b.linear_lower
        else:
            linear_lower = None

        if bounds_a.linear_upper is not None and bounds_b.linear_upper is not None:
            linear_upper = bounds_a.linear_upper + bounds_b.linear_upper
        elif bounds_a.linear_upper is not None:
            linear_upper = bounds_a.linear_upper
        elif bounds_b.linear_upper is not None:
            linear_upper = bounds_b.linear_upper
        else:
            linear_upper = None

        # Add bias terms
        bias_lower = bounds_a.bias_lower + bounds_b.bias_lower
        bias_upper = bounds_a.bias_upper + bounds_b.bias_upper

        return LinearBounds(
            region=bounds_a.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )


class ForwardLBPAddWithConstant(ForwardLBPStrategy):
    """Forward LBP strategy for ADD when at least one input is constant."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"add requires exactly 2 inputs, got {len(input_bounds)}")

        left = input_bounds[0]
        right = input_bounds[1]

        if isinstance(left, LinearBounds):
            x, c = left, right
        elif isinstance(right, LinearBounds):
            x, c = right, left
        else:
            raise TypeError(
                f"ForwardLBPAddWithConstant requires one input to be LinearBounds and the other to be "
                f"torch.Tensor or Number, got {type(left)} and {type(right)}"
            )

        c = cast(torch.Tensor | torch.types.Number, c)

        # Adding a constant: just add to bias terms
        bias_lower = x.bias_lower + c
        bias_upper = x.bias_upper + c

        return LinearBounds(
            region=x.region,
            linear_lower=x.linear_lower,
            bias_lower=bias_lower,
            linear_upper=x.linear_upper,
            bias_upper=bias_upper,
        )
