"""
Base classes for method-specific propagators.

Each propagation method (IBP, Forward LBP, Backward LBP) has its own
propagator class that orchestrates bound propagation through the graph.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from itertools import product

import torch

from ...bounds import AbstractBounds, IntervalBounds, LinearBounds
from ...ir import Graph
from ...regions import SimpleRegion


class InputBoundKind(StrEnum):
    """Coarse input classification used for strategy dispatch."""

    CONSTANT = "constant"
    ABSTRACT = "abstract"


def classify_input_bound(bound: AbstractBounds) -> InputBoundKind:
    """Classify a bound as constant or abstract for dispatch purposes."""
    if isinstance(bound, IntervalBounds):
        if torch.allclose(bound.lower, bound.upper):
            return InputBoundKind.CONSTANT
        return InputBoundKind.ABSTRACT

    if isinstance(bound, LinearBounds):
        if bound.linear_lower is None and bound.linear_upper is None:
            return InputBoundKind.CONSTANT
        return InputBoundKind.ABSTRACT

    return InputBoundKind.ABSTRACT


def classify_input_signature(
    bounds: list[AbstractBounds],
) -> tuple[InputBoundKind, ...]:
    """Build a dispatch signature for a list of input bounds."""
    return tuple(classify_input_bound(bound) for bound in bounds)


def enumerate_input_signatures(arity: int) -> list[tuple[InputBoundKind, ...]]:
    """Enumerate all constant/abstract signatures for a given arity."""
    if arity < 0:
        raise ValueError(f"arity must be non-negative, got {arity}")

    return [
        tuple(signature)
        for signature in product(
            (InputBoundKind.CONSTANT, InputBoundKind.ABSTRACT),
            repeat=arity,
        )
    ]


class MethodPropagator(ABC):
    """
    Abstract base class for method-specific bound propagators.

    A MethodPropagator implements a specific bound propagation algorithm
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
