from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from ...bounds import AbstractBounds, LinearBounds
from ...ir import Graph, NodeType
from ...regions import SimpleRegion
from ..forward_lbp import ForwardLBPStrategy, ForwardLBPStrategyRegistry
from .base import (
    BoundPropagator,
)


class ForwardLBPPropagator(BoundPropagator):
    """
    Forward Linear Bound Propagation (LBP).

    Propagates linear bounds forward through the computation graph.
    Uses interval arithmetic rules for all operations. Faster than LBP but
    less precise because it doesn't track linear dependencies.
    """

    def __init__(self, graph: Graph, registry: ForwardLBPStrategyRegistry | None = None):
        super().__init__(graph)
        self._registry = registry or ForwardLBPStrategyRegistry.default_registry()
        self._ensure_node_kind_annotations()

        self._bound_strategies = self._build_strategies()

    def _ensure_node_kind_annotations(self) -> None:
        """Ensure node-level input/output kind annotations exist for dispatch."""
        missing_operation_annotation = any(
            node.is_operation and node.input_signature is None for node in self.graph.nodes
        )
        if missing_operation_annotation:
            raise ValueError("All OPERATION and OUTPUT nodes must have input_signature annotations for LBP dispatch")

    def _build_strategies(self) -> dict[int, ForwardLBPStrategy]:
        """Build a cache of strategies for each node in the graph."""
        strategy_cache: dict[int, ForwardLBPStrategy] = {}
        for node in self.graph.nodes:
            if node.is_operation:
                signature = node.input_signature
                if signature is None:
                    raise ValueError(f"Node {node.id} is missing input_signature annotation required for LBP dispatch")
                strategy_cache[node.id] = self._registry.get_strategy(
                    node.op_type,
                    signature,
                )

        return strategy_cache

    @property
    def method_name(self) -> str:
        """Return the name of this propagation method."""
        return "lbp"

    def propagate(
        self,
        input_regions: Sequence[SimpleRegion],
    ) -> Sequence[AbstractBounds]:
        """
        Propagate interval bounds forward through the graph.

        Args:
            graph: The computation graph to propagate through.
            input_regions: List of input regions defining the domain.

        Returns:
            List of computed interval bounds, one for each output.
        """
        if len(input_regions) != len(self.graph.input_nodes):
            raise ValueError(f"Expected {len(self.graph.input_nodes)} input regions, got {len(input_regions)}")

        # Get nodes in topological order
        nodes = self.graph.topological_order()

        # Initialize bounds dictionary
        bounds: dict[int, LinearBounds | torch.Tensor] = {}

        # Initialize input bounds from input regions
        for node, region in zip(self.graph.input_nodes, input_regions, strict=True):
            bounds[node.id] = self._create_input_bounds(region)

        # Propagate through each node
        for node in nodes:
            match node.node_type:
                case NodeType.INPUT:
                    # Input bounds already initialized
                    continue
                case NodeType.CONSTANT | NodeType.PARAMETER:
                    bounds[node.id] = node.value
                case NodeType.OPERATION | NodeType.OUTPUT:
                    input_bounds: list[LinearBounds | torch.Tensor | torch.types.Number] = [
                        bounds[inp.id] for inp in node.inputs
                    ]
                    strategy = self._bound_strategies[node.id]
                    bounds[node.id] = strategy.propagate_forwards(node, input_bounds)
                case _:
                    raise ValueError(f"Unsupported node type: {node.node_type}")

        # Build output bounds list
        outputs = [bounds[node.id] for node in self.graph.output_nodes]

        for output in outputs:
            if not isinstance(output, LinearBounds):
                raise TypeError(f"Expected output bounds to be LinearBounds, got {type(output)}")

        outputs = cast(list[LinearBounds], outputs)

        return outputs

    def _create_input_bounds(
        self,
        region: SimpleRegion,
    ) -> LinearBounds:
        """
        Create linear bounds for an input node from the region.

        Args:
            region: Input region. Can be HyperRectangle (single input)
                   or MultiInputRegion (multiple inputs).

        Returns:
            LinearBounds from the region
        """
        # Handle multi-input regions
        lower, upper = region.aabb()

        # Single input region
        return LinearBounds(region, None, lower, None, upper)
