"""
CROWN strategy for log operation.

Logarithm is monotonic for positive inputs.
"""

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


class CROWNLogStrategy(BoundingStrategy):
    """
    CROWN strategy for LOG operation.

    Logarithm is monotonic for positive inputs.
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
        Compute CROWN bounds for log.

        Args:
            node: The LOG node
            input_bounds: List with one LinearBounds for the input
            config: Strategy configuration

        Returns:
            LinearBounds for the output
        """
        verify_linear_bounds(input_bounds)

        if len(input_bounds) != 1:
            raise ValueError(f"LOG requires exactly 1 input, got {len(input_bounds)}")

        bounds: LinearBounds = input_bounds[0]

        # Concretize and apply log (monotonic function)
        lower, upper = bounds.concretize()

        # Ensure positive inputs
        lower = torch.clamp(lower, min=1e-8)
        upper = torch.clamp(upper, min=1e-8)

        lower_out = torch.log(lower)
        upper_out = torch.log(upper)

        return LinearBounds(
            region=bounds.region,
            linear_lower=None,
            bias_lower=lower_out,
            linear_upper=None,
            bias_upper=upper_out,
        )
