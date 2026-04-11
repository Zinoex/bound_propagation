"""
Interval Bound Propagation (IBP) method.

Simple forward propagation using only interval bounds (no linear relaxations).
Faster but less precise than LBP methods.
"""

from collections.abc import Callable

import torch

from ...bounds import AbstractBounds, IntervalBounds
from ...ir import Graph, Node, NodeType
from ...regions import AbstractRegion, MultiInputRegion
from .base import (
    InputBoundKind,
    MethodPropagator,
    classify_input_signature,
    enumerate_input_signatures,
)


class IBPPropagator(MethodPropagator):
    """
    Interval Bound Propagation (IBP).

    Propagates simple interval bounds forward through the computation graph.
    Uses interval arithmetic rules for all operations. Faster than LBP but
    less precise because it doesn't track linear dependencies.
    """

    @property
    def method_name(self) -> str:
        """Return the name of this propagation method."""
        return "ibp"

    def propagate(
        self,
        graph: Graph,
        region: AbstractRegion,
    ) -> list[AbstractBounds]:
        """
        Propagate interval bounds forward through the graph.

        Args:
            graph: The computation graph to propagate through.
            region: Input region defining the domain.

        Returns:
            List of computed interval bounds, one for each output.
        """
        # Get nodes in topological order
        nodes = graph.topological_order()

        # Initialize bounds dictionary
        bounds: dict[int, IntervalBounds] = {}

        # Propagate through each node
        for node in nodes:
            # Compute bounds for this node
            if node.is_input:
                # Input nodes get bounds from the region
                bounds[node.id] = self._create_input_bounds(node, region)
            elif node.node_type == NodeType.CONSTANT:
                # Constants have point bounds
                bounds[node.id] = self._create_constant_bounds(node)
            else:
                # Operation node - compute bounds from inputs
                input_bounds = [bounds[inp.id] for inp in node.inputs]
                bounds[node.id] = self._compute_operation_bounds(node, input_bounds)

        return bounds

    def _create_input_bounds(
        self,
        node: Node,
        region: AbstractRegion,
    ) -> IntervalBounds:
        """
        Create interval bounds for an input node from the region.

        Args:
            node: The input node
            region: Input region. Can be HyperRectangle (single input)
                   or MultiInputRegion (multiple inputs).

        Returns:
            IntervalBounds from the region
        """
        # Handle multi-input regions
        if isinstance(region, MultiInputRegion):
            if node.id not in region:
                raise ValueError(
                    f"Input node {node.id} not found in MultiInputRegion. "
                    f"Available inputs: {list(region.keys())}"
                )
            node_region = region[node.id]
            return IntervalBounds(
                lower=node_region.lower.clone(),
                upper=node_region.upper.clone(),
            )
        else:
            # Single input region
            return IntervalBounds(
                lower=region.lower.clone(),
                upper=region.upper.clone(),
            )
