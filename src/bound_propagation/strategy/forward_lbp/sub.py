from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..strategy import BoundingStrategy
from .utils import verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class ForwardCrownSubStrategy(BoundingStrategy):
    """
    Forward CROWN strategy for SUB operation.

    For subtraction z = x - y:
    - Lower: W_l^z = W_l^x - W_u^y, b_l^z = b_l^x - b_u^y
    - Upper: W_u^z = W_u^x - W_l^y, b_u^z = b_u^x - b_l^y

    This is exact (no relaxation needed for linear operations).
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
        Compute forward CROWN bounds for subtraction.

        Args:
            node: The SUB node
            input_bounds: List of two LinearBounds for the operands
            config: Strategy configuration

        Returns:
            LinearBounds for the difference
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 2:
            raise ValueError(f"SUB requires exactly 2 inputs, got {len(input_bounds)}")

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
