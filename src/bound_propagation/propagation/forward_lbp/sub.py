from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPSubStrategy(ForwardLBPStrategy):
    """
    Forward LBP strategy for SUB operation.

    For subtraction z = x - y:
    - Lower: W_l^z = W_l^x - W_u^y, b_l^z = b_l^x - b_u^y
    - Upper: W_u^z = W_u^x - W_l^y, b_u^z = b_u^x - b_l^y

    This is exact (no relaxation needed for linear operations).
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"SUB requires exactly 2 inputs, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds) or not isinstance(input_bounds[1], LinearBounds):
            raise TypeError("ForwardLBPSubStrategy requires both inputs to be LinearBounds")

        bounds_a: LinearBounds = input_bounds[0]
        bounds_b: LinearBounds = input_bounds[1]

        # Subtract linear coefficients (lower - upper for lower bound)
        if bounds_a.linear_lower is not None and bounds_b.linear_upper is not None:
            linear_lower = bounds_a.linear_lower - bounds_b.linear_upper
        elif bounds_a.linear_lower is not None:
            linear_lower = bounds_a.linear_lower
        elif bounds_b.linear_upper is not None:
            linear_lower = -bounds_b.linear_upper
        else:
            linear_lower = None

        if bounds_a.linear_upper is not None and bounds_b.linear_lower is not None:
            linear_upper = bounds_a.linear_upper - bounds_b.linear_lower
        elif bounds_a.linear_upper is not None:
            linear_upper = bounds_a.linear_upper
        elif bounds_b.linear_lower is not None:
            linear_upper = -bounds_b.linear_lower
        else:
            linear_upper = None

        # Subtract bias terms
        bias_lower = bounds_a.bias_lower - bounds_b.bias_upper
        bias_upper = bounds_a.bias_upper - bounds_b.bias_lower

        return LinearBounds(
            region=bounds_a.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )


class ForwardLBPSubConstantRight(ForwardLBPStrategy):
    """Forward LBP strategy for SUB when the second input is constant: x - c."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"SUB requires exactly 2 inputs, got {len(input_bounds)}")

        x = input_bounds[0]
        c = input_bounds[1]

        if not isinstance(x, LinearBounds) or isinstance(c, LinearBounds):
            raise TypeError(
                "ForwardLBPSubConstantRight requires the first input to be LinearBounds "
                "and the second input to be torch.Tensor or Number"
            )

        # Subtract constant from bias terms
        bias_lower = x.bias_lower - c
        bias_upper = x.bias_upper - c

        return LinearBounds(
            region=x.region,
            linear_lower=x.linear_lower,
            bias_lower=bias_lower,
            linear_upper=x.linear_upper,
            bias_upper=bias_upper,
        )


class ForwardLBPSubConstantLeft(ForwardLBPStrategy):
    """Forward LBP strategy for SUB when the first input is constant: c - x."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"SUB requires exactly 2 inputs, got {len(input_bounds)}")

        c = input_bounds[0]
        x = input_bounds[1]

        if not isinstance(x, LinearBounds) or isinstance(c, LinearBounds):
            raise TypeError(
                "ForwardLBPSubConstantLeft requires the second input to be LinearBounds "
                "and the first input to be torch.Tensor or Number"
            )

        # c - x: flip signs and bounds
        linear_lower = -x.linear_upper if x.linear_upper is not None else None
        linear_upper = -x.linear_lower if x.linear_lower is not None else None
        bias_lower = c - x.bias_upper
        bias_upper = c - x.bias_lower

        return LinearBounds(
            region=x.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
