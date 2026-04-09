from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..strategy import ForwardBoundingStrategy
from .utils import apply_linear_relaxation, compute_tanh_alpha_beta, verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class ForwardLBPTanhStrategy(ForwardBoundingStrategy):
    """
    Forward LBP strategy for TANH operation.

    Uses adaptive linear relaxations based on the input bounds regime.
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
        Compute forward LBP bounds for tanh.

        Args:
            node: The TANH node
            input_bounds: List with one LinearBounds for the input
            config: Strategy configuration

        Returns:
            LinearBounds for the output
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"TANH requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

        # Concretize to get interval bounds for determining relaxation
        lower, upper = bounds.concretize()

        # Compute alpha/beta parameters for tanh relaxation
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_tanh_alpha_beta(
            lower, upper
        )

        # Apply the linear relaxation to the bounds
        return apply_linear_relaxation(
            bounds, alpha_lower, beta_lower, alpha_upper, beta_upper
        )
