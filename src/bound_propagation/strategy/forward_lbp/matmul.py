from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from ..strategy import BoundingStrategy
from .utils import verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class ForwardLBPMatmulStrategy(BoundingStrategy):
    """
    Forward LBP strategy for MATMUL operation.

    For matrix multiplication z = x @ y:
    - If y is constant: exact linear transformation
    - If both vary: concretize to intervals(?)
    """

    @property
    def method_name(self) -> str:
        """Return the method name for this strategy."""
        return "forward"

    def compute_bounds(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> AbstractBounds:
        """
        Compute forward LBP bounds for matmul.

        Args:
            node: The MATMUL node
            input_bounds: List of two LinearBounds for the operands
            config: Strategy configuration

        Returns:
            LinearBounds for the product
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 2:
            raise ValueError(f"MATMUL requires exactly 2 inputs, got {len(input_bounds)}")

        bounds_a: LinearBounds = input_bounds[0]
        bounds_b: LinearBounds = input_bounds[1]

        # Check if second operand is constant (common case: x @ W)
        b_is_constant = bounds_b.linear_lower is None and bounds_b.linear_upper is None

        if b_is_constant:
            # y is constant: z = x @ W
            weight = bounds_b.bias_lower  # The constant weight matrix
            return self._matmul_by_constant(bounds_a, weight)
        else:
            # Both vary - need to concretize
            raise NotImplementedError(
                "LBP matmul with two varying operands not yet supported. "
                "Use constant weights or switch to IBP method."
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

        weight_pos = torch.clamp(weight, min=0)
        weight_neg = torch.clamp(weight, max=0)

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
