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


class ForwardCrownTanhStrategy(BoundingStrategy):
    """
    Forward CROWN strategy for TANH operation.

    Concretizes input bounds, applies tanh, and returns constant bounds.
    Could be improved with adaptive linear relaxations.
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
        Compute forward CROWN bounds for tanh.

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

        # Concretize and apply tanh
        lower, upper = bounds.concretize()
        lower_out = torch.tanh(lower)
        upper_out = torch.tanh(upper)

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower_out,
            linear_upper=None,
            bias_upper=upper_out,
        )
