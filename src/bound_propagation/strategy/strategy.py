"""
Abstract bounding strategy interface.

Defines the contract for all bounding strategies (IBP, forward, backward, LBP, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..bounds import AbstractBounds
    from ..ir import Node
    from .config import StrategyConfig


class BoundingStrategy(ABC):
    """
    Abstract base class for bounding strategies.

    A bounding strategy computes bounds for a single operation node in the graph.
    Different strategies implement different propagation methods:
    - IBP (Interval Bound Propagation): Forward interval arithmetic
    - Forward: Forward linear bound propagation with alpha-beta parameterization
    - Backward: Backward linear bound propagation (LBP-style)
    - LBP-IBP: Hybrid method combining interval and linear bounds

    Each strategy is responsible for computing the output bounds of one node
    given the input bounds of its input nodes.

    The strategy pattern allows easy extension with new propagation methods
    and mixing different methods for different operations.
    """

    @abstractmethod
    def compute_bounds(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        config: StrategyConfig,
    ) -> AbstractBounds:
        """
        Compute output bounds for a node.

        Args:
            node: The operation node to compute bounds for
            input_bounds: Bounds for each input to the node (in order)
            config: Configuration for this computation

        Returns:
            Bounds for the node's output

        Raises:
            ValueError: If the operation is not supported by this strategy
            RuntimeError: If bound computation fails
        """
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """
        Get the name of this bounding method.

        Returns:
            Method name (e.g., "ibp", "forward", "backward", "lbp")
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(method={self.method_name})"
