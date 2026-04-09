"""
Base classes for method-specific propagators.

Each propagation method (IBP, Forward LBP, Backward LBP) has its own
propagator class that orchestrates bound propagation through the graph.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from bound_propagation.bounds.abstract_bounds import AbstractBounds
from bound_propagation.cache.bound_cache import BoundCache
from bound_propagation.ir.graph import Graph
from bound_propagation.regions.abstract import AbstractRegion


class MethodPropagator(ABC):
    """
    Abstract base class for method-specific bound propagators.
    
    A MethodPropagator implements a specific bound propagation algorithm
    (e.g., IBP, Forward LBP, Backward LBP) by traversing the computation
    graph and computing bounds at each node.
    
    Subclasses implement the propagate() method with their specific logic.
    
    Attributes:
        cache: Optional cache for storing computed bounds
    """
    
    def __init__(self, *, cache: Optional[BoundCache] = None) -> None:
        """
        Initialize propagator.
        
        Args:
            cache: Optional cache for storing/retrieving bounds. If None,
                  no caching is performed.
        """
        self.cache = cache
    
    @abstractmethod
    def propagate(
        self,
        graph: Graph,
        region: AbstractRegion,
        start_node: Optional[int] = None,
    ) -> Dict[int, AbstractBounds]:
        """
        Propagate bounds through the computation graph.
        
        Args:
            graph: The computation graph to propagate through.
            region: Input region (e.g., HyperRectangle) defining bounds on inputs.
            start_node: Optional node ID to start propagation from. If None,
                       propagates to all output nodes.
        
        Returns:
            Dictionary mapping node IDs to their computed bounds.
            At minimum, includes bounds for all output nodes.
        """
        pass
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of this propagation method."""
        pass
