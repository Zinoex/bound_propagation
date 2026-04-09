"""
Backward propagation strategy for ADD operation.

For z = x + y, backward propagation passes bounds through unchanged to both inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...bounds import LinearBounds
from ..backward_strategy import BackwardBoundingStrategy

if TYPE_CHECKING:
    from ...bounds import AbstractBounds
    from ...ir import Node
    from ..config import StrategyConfig


class BackwardAddStrategy(BackwardBoundingStrategy):
    """
    Backward propagation strategy for ADD operation.

    For z = x + y:
    - ∂z/∂x = 1, ∂z/∂y = 1
    - A_x += A_z
    - A_y += A_z
    
    Both inputs receive the full linear bounds from the output.
    """

    @property
    def method_name(self) -> str:
        """Return method name."""
        return "backward"

    def propagate_backward(
        self,
        node: Node,
        input_idx: int,
        output_bounds: LinearBounds,
        concrete_input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> LinearBounds:
        """
        Propagate bounds backward through ADD.

        For addition, both inputs get the same contribution (no scaling needed).

        Args:
            node: The ADD node
            input_idx: Index of input (0 for x, 1 for y)
            output_bounds: Linear bounds on the sum (A_z, Ā_z)
            concrete_input_bounds: Concrete bounds (not used for linear ops)
            config: Strategy configuration

        Returns:
            Contribution to the specified input (same as output_bounds)
        """
        if input_idx not in (0, 1):
            raise ValueError(f"ADD expects 2 inputs, got input_idx={input_idx}")

        # For addition, both inputs receive the full output bounds
        # This is exact (no approximation)
        return output_bounds
