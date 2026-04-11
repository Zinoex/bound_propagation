from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from ...bounds import AbstractBounds, IntervalBounds
from ...ir import Graph, Node, NodeType
from ...regions import SimpleRegion
from ..ibp import ForwardIBPStrategy, ForwardIBPStrategyRegistry
from .base import (
    MethodPropagator,
)


class IBPPropagator(MethodPropagator):
    """
    Interval Bound Propagation (IBP).

    Propagates simple interval bounds forward through the computation graph.
    Uses interval arithmetic rules for all operations. Faster than LBP but
    less precise because it doesn't track linear dependencies.
    """
    def __init__(self, graph: Graph, registry: ForwardIBPStrategyRegistry | None = None):
        super().__init__(graph)
        self._registry = registry or ForwardIBPStrategyRegistry.default_registry()

        self._bound_strategies = self._build_strategy_cache()

    def _build_strategy_cache(self) -> dict[int, ForwardIBPStrategy]:
        """Build a cache of strategies for each node in the graph."""
        strategy_cache: dict[int, ForwardIBPStrategy] = {}
        for node in self.graph.nodes:
            if node.node_type == NodeType.OPERATION:
                strategy_cache[node.id] = self._registry.get_strategy(node.op_type)
        return strategy_cache

    @property
    def method_name(self) -> str:
        """Return the name of this propagation method."""
        return "ibp"

    def propagate(
        self,
        input_regions: list[SimpleRegion],
    ) -> Sequence[AbstractBounds]:
        """
        Propagate interval bounds forward through the graph.

        Args:
            graph: The computation graph to propagate through.
            input_regions: List of input regions defining the domain.

        Returns:
            List of computed interval bounds, one for each output.
        """
        # Get nodes in topological order
        nodes = self.graph.topological_order()

        # Initialize bounds dictionary
        bounds: dict[int, IntervalBounds | torch.Tensor] = {}

        # Initialize input bounds from input regions
        for node, region in zip(self.graph.input_nodes, input_regions, strict=True):
            bounds[node.id] = self._create_input_bounds(node, region)

        # Propagate through each node
        for node in nodes:
            match node.node_type:
                case NodeType.INPUT:
                    # Input bounds already initialized
                    continue
                case NodeType.CONSTANT | NodeType.PARAMETER:
                    bounds[node.id] = node.value
                case NodeType.OPERATION | NodeType.OUTPUT:
                    input_bounds = [bounds[inp.id] for inp in node.inputs]
                    strategy = self._bound_strategies[node.id]
                    bounds[node.id] = strategy.propagate_forwards(node, input_bounds)
                case _:
                    raise ValueError(f"Unsupported node type: {node.node_type}")

        # Build output bounds list
        outputs = [bounds[node.id] for node in self.graph.output_nodes]

        for output in outputs:
            if not isinstance(output, IntervalBounds):
                raise TypeError(f"Expected output bounds to be IntervalBounds, got {type(output)}")

        outputs = cast(list[IntervalBounds], outputs)

        return outputs

    def _create_input_bounds(
        self,
        node: Node,
        region: SimpleRegion,
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
        lower, upper = region.aabb()

        # Single input region
        return IntervalBounds(lower, upper)
