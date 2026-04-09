"""
Abstract bounding strategy interface.

Defines the contract for all bounding strategies (IBP, forward, backward, LBP, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from ..bounds import AbstractBounds
    from ..ir import Node


class BoundingStrategy(ABC):
    @property
    @abstractmethod
    def method_name(self) -> str:
        """
        Get the name of this bounding method.

        Returns:
            Method name (e.g., "ibp", "forward", "backward")
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(method={self.method_name})"


T = TypeVar("T", bound=AbstractBounds)


class ForwardBoundingStrategy(BoundingStrategy, ABC, Generic[T]):
    """
    Abstract base class for bounding strategies.

    A bounding strategy computes bounds for a single operation node in the graph.
    Different strategies implement different propagation methods:
    - IBP (Interval Bound Propagation): Forward interval arithmetic
    - Forward: Forward linear bound propagation with alpha-beta parameterization
    - Backward: Backward linear bound propagation (LBP-style)

    Each strategy is responsible for computing the output bounds of one node
    given the input bounds of its input nodes.

    The strategy pattern allows easy extension with new propagation methods
    and mixing different methods for different operations.
    """

    @abstractmethod
    def propagate_forwards(
        self,
        node: Node,
        input_bounds: list[T],
    ) -> T:
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




class BackwardBoundingStrategy(BoundingStrategy, ABC, Generic[T]):
    """
    Abstract base class for backward bound propagation strategies.

    Backward propagation works differently from forward propagation:
    - Forward: Given bounds on inputs, compute bounds on output
    - Backward: Given linear bounds on output, propagate backward to compute
                contribution to linear bounds on inputs

    For an operation z = f(x, y, ...):
    - We have linear bounds A_z, Ā_z representing how the final output depends on z
    - We have concrete bounds for x, y (from forward pass)
    - We compute how A_z, Ā_z translate to dependencies on each input

    Example: z = x + y
        If A_z represents "output_lower = A_z @ z + b_z"
        Then A_x gets the same contribution: A_x += A_z
        And A_y gets the same contribution: A_y += A_z

    Example: z = relu(x)
        Use concrete bounds on x to compute linear relaxation [α, β]
        Propagate: A_x += α * A_z
    """

    @abstractmethod
    def propagate_backwards(
        self,
        node: Node,
        output_bounds: T,
    ) -> list[T]:
        """
        Propagate linear bounds backward through this operation to a specific input.

        Args:
            node: The operation node (z = f(x, y, ...))
            input_idx: Index of the input we're propagating to (0 for x, 1 for y, etc.)
            output_bounds: Linear bounds for the operation output (A_z, Ā_z)
                          Represents how the final output depends on this operation's output
            concrete_input_bounds: Concrete bounds for all inputs (from forward pass)
                                   Used for computing relaxations of non-linear operations
            config: Strategy configuration

        Returns:
            Linear bounds representing the contribution to the specified input
            This contribution will be accumulated: A_input += returned_bounds

        Raises:
            ValueError: If the operation is not supported or input_idx is invalid
            RuntimeError: If propagation fails
        """
        pass
