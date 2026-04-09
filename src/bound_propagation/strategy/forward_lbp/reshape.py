from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..strategy import BoundingStrategy
from .utils import verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class ForwardLBPReshapeStrategy(BoundingStrategy):
    """
    Forward LBP strategy for RESHAPE operation.

    Reshape doesn't change values, only their shape.
    Linear coefficients remain unchanged, bias terms are reshaped.
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
        Compute forward LBP bounds for reshape.

        Args:
            node: The RESHAPE node
            input_bounds: List with one LinearBounds for the input
            config: Strategy configuration

        Returns:
            LinearBounds for the reshaped output
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"RESHAPE requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

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
