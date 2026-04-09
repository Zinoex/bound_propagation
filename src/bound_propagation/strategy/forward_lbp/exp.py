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


class ForwardCrownExpStrategy(BoundingStrategy):
    """
    Forward CROWN strategy for EXP operation.

    Exponential is monotonic, so we apply it to concretized bounds.
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
        Compute forward CROWN bounds for exp.

        Args:
            node: The EXP node
            input_bounds: List with one LinearBounds for the input
            config: Strategy configuration

        Returns:
            LinearBounds for the output
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"EXP requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

        # Concretize and apply exp (monotonic function)
        lower, upper = bounds.concretize()
        lower_out = torch.exp(lower)
        upper_out = torch.exp(upper)

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower_out,
            linear_upper=None,
            bias_upper=upper_out,
        )
