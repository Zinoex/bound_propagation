from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import ForwardLBPStrategy

if TYPE_CHECKING:
    import torch

    from ...ir import Node


class ForwardLBPMatmul(ForwardLBPStrategy):
    """
    Forward LBP strategy for MATMUL when both inputs are abstract.

    For matrix multiplication z = x @ y where both have linear dependencies:
    - Not yet supported, should concretize to intervals
    """

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"matmul requires exactly 2 inputs, got {len(input_bounds)}")

        if not isinstance(input_bounds[0], LinearBounds) or not isinstance(input_bounds[1], LinearBounds):
            raise TypeError("ForwardLBPMatmul requires both inputs to be LinearBounds")

        # Both vary - need to concretize
        raise NotImplementedError(
            "LBP matmul with two varying operands not yet supported. Use constant weights or switch to IBP method."
        )


class ForwardLBPMatmulConstant(ForwardLBPStrategy):
    """Forward LBP strategy for MATMUL when the second input is constant: x @ W."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"MATMUL requires exactly 2 inputs, got {len(input_bounds)}")

        x = input_bounds[0]
        weight = input_bounds[1]

        if not isinstance(x, LinearBounds) or not isinstance(weight, torch.Tensor):
            raise TypeError(
                "ForwardLBPMatmulConstant requires the first input to be LinearBounds "
                "and the second input to be torch.Tensor"
            )

        return self._matmul_by_constant(x, weight)


class ForwardLBPConstantMatmul(ForwardLBPStrategy):
    """Forward LBP strategy for MATMUL when the first input is constant: W @ x."""

    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number],
    ) -> LinearBounds:
        if len(input_bounds) != 2:
            raise ValueError(f"matmul requires exactly 2 inputs, got {len(input_bounds)}")

        weight = input_bounds[0]
        x = input_bounds[1]

        if not isinstance(weight, torch.Tensor) or not isinstance(x, LinearBounds):
            raise TypeError(
                "ForwardLBPConstantMatmul requires the first input to be torch.Tensor "
                "and the second input to be LinearBounds"
            )

        # W @ x: need different computation
        # z = W @ x, where x has linear bounds
        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)

        # Lower bound
        if x.linear_lower is not None and x.linear_upper is not None:
            linear_lower = weight_pos @ x.linear_lower + weight_neg @ x.linear_upper
        elif x.linear_lower is not None:
            linear_lower = weight_pos @ x.linear_lower + weight_neg @ x.linear_lower
        elif x.linear_upper is not None:
            linear_lower = weight_pos @ x.linear_upper + weight_neg @ x.linear_upper
        else:
            linear_lower = None

        bias_lower = weight_pos @ x.bias_lower + weight_neg @ x.bias_upper

        # Upper bound
        if x.linear_lower is not None and x.linear_upper is not None:
            linear_upper = weight_pos @ x.linear_upper + weight_neg @ x.linear_lower
        elif x.linear_upper is not None:
            linear_upper = weight_pos @ x.linear_upper + weight_neg @ x.linear_upper
        elif x.linear_lower is not None:
            linear_upper = weight_pos @ x.linear_lower + weight_neg @ x.linear_lower
        else:
            linear_upper = None

        bias_upper = weight_pos @ x.bias_upper + weight_neg @ x.bias_lower

        return LinearBounds(
            region=x.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )

    def _matmul_by_constant(self, bounds: LinearBounds, weight: torch.Tensor) -> LinearBounds:
        """
        Multiply linear bounds by a constant matrix.

        Args:
            bounds: LinearBounds to multiply
            weight: Constant weight matrix

        Returns:
            Transformed LinearBounds
        """
        # z = x @ W, where x has linear bounds
        # Lower: W_l^z @ x0 + b_l^z = W_l^x @ W @ x0 + b_l^x @ W
        # Need to handle positive/negative weights for tight bounds

        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)

        # Lower bound
        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            # Use lower for positive weights, upper for negative
            linear_lower = bounds.linear_lower @ weight_pos + bounds.linear_upper @ weight_neg
        elif bounds.linear_lower is not None:
            linear_lower = bounds.linear_lower @ weight_pos + bounds.linear_lower @ weight_neg
        elif bounds.linear_upper is not None:
            linear_lower = bounds.linear_upper @ weight_pos + bounds.linear_upper @ weight_neg
        else:
            linear_lower = None

        bias_lower = bounds.bias_lower @ weight_pos + bounds.bias_upper @ weight_neg

        # Upper bound
        if bounds.linear_lower is not None and bounds.linear_upper is not None:
            # Use upper for positive weights, lower for negative
            linear_upper = bounds.linear_upper @ weight_pos + bounds.linear_lower @ weight_neg
        elif bounds.linear_upper is not None:
            linear_upper = bounds.linear_upper @ weight_pos + bounds.linear_upper @ weight_neg
        elif bounds.linear_lower is not None:
            linear_upper = bounds.linear_lower @ weight_pos + bounds.linear_lower @ weight_neg
        else:
            linear_upper = None

        bias_upper = bounds.bias_upper @ weight_pos + bounds.bias_lower @ weight_neg

        return LinearBounds(
            region=bounds.region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
        )
