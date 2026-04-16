from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..strategy import ForwardBoundingStrategy
from .utils import apply_linear_relaxation_backward, compute_relu_relaxation, verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class BackwardLBPReluStrategy(ForwardBoundingStrategy):
    """
    Backward LBP strategy for RELU operation.

    For ReLU y = max(0, x), we use adaptive linear relaxations in backward mode.
    The relaxations are the same as forward mode, but applied via backward composition.
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
        Compute backward LBP bounds for ReLU.

        Args:
            node: The RELU node
            input_bounds: List with one LinearBounds for the input
            config: Strategy configuration

        Returns:
            LinearBounds for the output
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"RELU requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

        # Concretize to get interval bounds for determining ReLU behavior
        lower, upper = bounds.concretize()

        # Compute alpha/beta parameters for ReLU relaxation
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_relu_relaxation(lower, upper, adaptive=False)

        # Apply the linear relaxation to the bounds using backward composition
        return apply_linear_relaxation_backward(bounds, alpha_lower, beta_lower, alpha_upper, beta_upper)
