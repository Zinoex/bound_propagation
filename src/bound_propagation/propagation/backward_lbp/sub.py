from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..strategy import ForwardBoundingStrategy
from .utils import verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class BackwardLBPSubStrategy(ForwardBoundingStrategy):
    """
    Backward LBP strategy for SUB operation.

    For subtraction z = x - y, in backward mode we have bounds on z
    and need to propagate them back to x and y.

    This is exact (no relaxation needed for linear operations).
    """

    @property
    def method_name(self) -> str:
        """Return the method name for this strategy."""
        return "backward"

    def compute_bounds(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> AbstractBounds:
        """
        Compute backward LBP bounds for subtraction.

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

        lower_regions, linear_lower, input_ids = LinearBounds.combine_linear_terms(
            [(bounds_a, "lower", 1.0), (bounds_b, "upper", -1.0)]
        )
        upper_regions, linear_upper, upper_input_ids = LinearBounds.combine_linear_terms(
            [(bounds_a, "upper", 1.0), (bounds_b, "lower", -1.0)]
        )

        if input_ids != upper_input_ids:
            raise ValueError(f"Lower and upper input IDs must match, got {input_ids} vs {upper_input_ids}")

        # Subtract bias terms
        bias_lower = bounds_a.bias_lower - bounds_b.bias_upper
        bias_upper = bounds_a.bias_upper - bounds_b.bias_lower

        return LinearBounds(
            regions=lower_regions or upper_regions,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=input_ids,
        )
