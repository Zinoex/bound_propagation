"""
Base classes for method-specific propagators.

Each propagation method (IBP, Forward LBP, Backward LBP) has its own
propagator class that orchestrates bound propagation through the graph.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ...bounds import AbstractBounds
from ...ir import Graph
from ...regions import SimpleRegion


class BoundPropagator(ABC):
    """
    Abstract base class for method-specific bound propagators.

    A BoundPropagator implements a specific bound propagation algorithm
    (e.g., IBP, Forward LBP, Backward LBP) by traversing the computation
    graph and computing bounds at each node.

    Subclasses implement the propagate() method with their specific logic.

    Attributes:
        graph: The computation graph being propagated through.
    """

    def __init__(self, graph: Graph):
        self._graph = graph

    @property
    def graph(self) -> Graph:
        """Access the computation graph being propagated through."""
        return self._graph

    @abstractmethod
    def propagate(
        self,
        input_regions: list[SimpleRegion],
    ) -> Sequence[AbstractBounds]:
        """
        Propagate bounds through the computation graph.

        Args:
            input_regions: List of input regions (e.g., HyperRectangles) defining bounds on inputs.

        Returns:
            List of computed bounds, one for each output node.
        """
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of this propagation method."""
        pass
