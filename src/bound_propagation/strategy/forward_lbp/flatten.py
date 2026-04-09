from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..strategy import ForwardBoundingStrategy
from .utils import verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class ForwardLBPFlattenStrategy(ForwardBoundingStrategy):
    """
    Forward LBP strategy for FLATTEN operation.

    Flatten converts to 1D shape without changing values.
    Linear coefficients remain unchanged, bias terms are flattened.
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
        Compute forward LBP bounds for flatten.

        Args:
            node: The FLATTEN node
            input_bounds: List with one LinearBounds for the input
            config: Strategy configuration

        Returns:
            LinearBounds for the flattened output
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"FLATTEN requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

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
