from __future__ import annotations

from typing import TYPE_CHECKING

from ...bounds import LinearBounds
from .base import BackwardLBPBoundingStrategy

if TYPE_CHECKING:
    from ...ir import Node


class BackwardLBPAddStrategy(BackwardLBPBoundingStrategy):
    """
    Backward LBP strategy for ADD operation.

    For addition z = x + y, in backward mode we have bounds on z
    and need to propagate them back to x and y.

    This is exact (no relaxation needed for linear operations).
    """

    def propagate_backwards(
        self,
        node: Node,
        output_bounds: LinearBounds,
    ) -> list[LinearBounds]:
        return [output_bounds, output_bounds]
