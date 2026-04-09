"""
CROWN strategy for addition operation.

Addition is a linear operation, so CROWN bounds are exact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..strategy import BoundingStrategy
from .utils import verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class CROWNAddStrategy(BoundingStrategy):
    """
    CROWN strategy for ADD operation.

    For addition z = x + y:
    - Lower: W_l^z @ x0 + b_l^z = W_l^x @ x0 + b_l^x + W_l^y @ x0 + b_l^y
    - Upper: W_u^z @ x0 + b_u^z = W_u^x @ x0 + b_u^x + W_u^y @ x0 + b_u^y

    This is exact (no relaxation needed for linear operations).
    """

    @property
    def method_name(self) -> str:
        """Return the method name for this strategy."""
        return "crown"

    def compute_bounds(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> AbstractBounds:
        """
        Compute CROWN bounds for addition.

        Args:
            node: The ADD node
            input_bounds: List of two LinearBounds for the operands
            config: Strategy configuration

        Returns:
            LinearBounds for the sum
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 2:
            raise ValueError(f"ADD requires exactly 2 inputs, got {len(input_bounds)}")

        bounds_a: LinearBounds = input_bounds[0]
        bounds_b: LinearBounds = input_bounds[1]

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
