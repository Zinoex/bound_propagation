from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from ..strategy import ForwardBoundingStrategy
from .utils import apply_linear_relaxation_backward, compute_sigmoid_relaxation, verify_linear_bounds

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class BackwardLBPSigmoidStrategy(ForwardBoundingStrategy):
    """
    Backward LBP strategy for SIGMOID operation.

    For sigmoid y = σ(x), we use adaptive linear relaxations in backward mode.
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
        Compute backward LBP bounds for sigmoid.

        Args:
            node: The SIGMOID node
            input_bounds: List with one LinearBounds for the input
            config: Strategy configuration

        Returns:
            LinearBounds for the output
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"SIGMOID requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

        # Concretize to get interval bounds for determining relaxation
        lower, upper = bounds.concretize()

        # Compute alpha/beta parameters for sigmoid relaxation
        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_relaxation(lower, upper)

        # Apply the linear relaxation to the bounds using backward composition
        return apply_linear_relaxation_backward(bounds, alpha_lower, beta_lower, alpha_upper, beta_upper)
