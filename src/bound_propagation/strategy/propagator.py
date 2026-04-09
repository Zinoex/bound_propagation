"""
Bound propagation orchestrator.

Coordinates bound computation across a computation graph using registered strategies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..bounds import AbstractBounds, IntervalBounds, LinearBounds
from ..regions import HyperRectangle
from .config import StrategyConfig
from .registry import get_global_registry

if TYPE_CHECKING:
    from ..ir import Graph, Node
    from .registry import StrategyRegistry


class BoundPropagator:
    """
    Orchestrates bound propagation through a computation graph.

    The propagator:
    1. Walks the graph in topological order
    2. For each node, looks up the appropriate strategy from the registry
    3. Computes bounds using that strategy
    4. Caches results for efficient repeated queries

    Example:
        propagator = BoundPropagator(graph, method="ibp")
        bounds = propagator.compute_bounds(
            input_region=HyperRectangle(lower, upper)
        )
    """

    def __init__(
        self,
        graph: Graph,
        method: str = "ibp",
        registry: StrategyRegistry | None = None,
        config: StrategyConfig | None = None,
    ):
        """
        Initialize bound propagator.

        Args:
            graph: The computation graph to propagate bounds through
            method: The bounding method to use (e.g., "ibp", "forward", "backward")
            registry: Strategy registry (uses global registry if None)
            config: Default strategy configuration (creates default if None)
        """
        self.graph = graph
        self.method = method
        self.registry = registry if registry is not None else get_global_registry()
        self.config = config if config is not None else StrategyConfig()

        # Cache for computed bounds: node_id -> bounds
        self._bounds_cache: dict[int, AbstractBounds] = {}

    def compute_bounds(
        self,
        input_region: HyperRectangle,
        node_configs: dict[int, StrategyConfig] | None = None,
    ) -> dict[int, AbstractBounds]:
        """
        Compute bounds for all nodes in the graph.

        Args:
            input_region: The input region specification (e.g., HyperRectangle)
            node_configs: Optional per-node strategy configurations

        Returns:
            Dictionary mapping node ID to computed bounds

        Raises:
            ValueError: If no strategy is available for a node
            RuntimeError: If bound computation fails
        """
        # Clear cache
        self._bounds_cache.clear()

        node_configs = node_configs or {}

        # Get topological order
        nodes_in_order = self.graph.topological_order()

        # Process each node
        for node in nodes_in_order:
            # Get bounds for this node
            bounds = self._compute_node_bounds(node, input_region, node_configs)

            # Cache the result
            self._bounds_cache[node.id] = bounds

        return self._bounds_cache

    def get_bounds(self, node_id: int) -> AbstractBounds | None:
        """
        Get cached bounds for a node.

        Args:
            node_id: The node ID

        Returns:
            Cached bounds, or None if not computed yet
        """
        return self._bounds_cache.get(node_id)

    def get_output_bounds(self) -> list[AbstractBounds]:
        """
        Get bounds for all output nodes.

        Returns:
            List of bounds for output nodes

        Raises:
            RuntimeError: If bounds haven't been computed yet
        """
        output_bounds = []
        for node in self.graph.output_nodes:
            bounds = self.get_bounds(node.id)
            if bounds is None:
                raise RuntimeError(
                    f"Bounds not computed for output node {node.id}. "
                    "Call compute_bounds() first."
                )
            output_bounds.append(bounds)
        return output_bounds

    def _compute_node_bounds(
        self,
        node: Node,
        input_region: HyperRectangle,
        node_configs: dict[int, StrategyConfig],
    ) -> AbstractBounds:
        """
        Compute bounds for a single node.

        Args:
            node: The node to compute bounds for
            input_region: The input region specification
            node_configs: Per-node configurations

        Returns:
            Computed bounds for the node

        Raises:
            ValueError: If no strategy is available
        """
        # Handle input nodes: create bounds from input region
        if node.is_input:
            return self._create_input_bounds(node, input_region)

        # Handle constant and parameter nodes: create point bounds
        if node.is_constant or node.is_parameter:
            return self._create_constant_bounds(node)

        # Get input bounds for this node
        input_bounds = self._get_input_bounds(node)

        # Get strategy for this operation
        strategy = self.registry.get(node.op_type, self.method)
        if strategy is None:
            raise ValueError(
                f"No strategy registered for operation {node.op_type} "
                f"with method {self.method}"
            )

        # Get config for this node (use per-node config if available, else default)
        config = node_configs.get(node.id, self.config)

        # Compute bounds using strategy
        try:
            bounds = strategy.compute_bounds(node, input_bounds, config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute bounds for node {node.id} ({node.op_type}): {e}"
            ) from e

        return bounds

    def _create_input_bounds(
        self,
        node: Node,
        input_region: HyperRectangle,
    ) -> AbstractBounds:
        """
        Create bounds for an input node from the input region.

        For IBP, this creates IntervalBounds from the hyperrectangle.
        For LBP, this creates LinearBounds with identity mapping.

        Args:
            node: The input node
            input_region: The input region specification

        Returns:
            Bounds for the input
        """
        if self.method == "ibp":
            # IBP: create interval bounds from region
            return IntervalBounds(
                region=input_region,
                lower=input_region.lower,
                upper=input_region.upper,
            )
        elif self.method in ("lbp", "forward"):
            # LBP/Forward: create linear bounds with identity mapping
            # This represents: lower = I @ x + 0, upper = I @ x + 0
            input_flat = input_region.lower.flatten()
            input_size = input_flat.numel()

            identity = torch.eye(input_size, dtype=input_region.dtype, device=input_region.device)
            bias = torch.zeros(input_size, dtype=input_region.dtype, device=input_region.device)

            return LinearBounds(
                region=input_region,
                linear_lower=identity,
                bias_lower=bias,
                linear_upper=identity,
                bias_upper=bias,
            )
        else:
            # For other methods, default to IntervalBounds
            return IntervalBounds(
                region=input_region,
                lower=input_region.lower,
                upper=input_region.upper,
            )

    def _create_constant_bounds(self, node: Node) -> AbstractBounds:
        """
        Create bounds for a constant node.

        Constants have zero-width bounds (point bounds).
        For LBP, creates constant LinearBounds (no linear dependency).

        Args:
            node: The constant node

        Returns:
            Point bounds for the constant
        """
        # Get the constant value from node attributes
        value = node.attributes.get("value")
        if value is None:
            raise ValueError(f"Constant node {node.id} missing 'value' attribute")

        # Create a point region and point bounds
        if isinstance(value, torch.Tensor):
            tensor_value = value
        else:
            tensor_value = torch.tensor(value)

        region = HyperRectangle(tensor_value, tensor_value)

        if self.method in ("lbp", "forward"):
            # LBP/Forward: create constant LinearBounds (no linear dependency on input)
            # This represents: lower = 0 @ x + value, upper = 0 @ x + value
            return LinearBounds(
                region=region,
                linear_lower=None,  # No linear dependency
                bias_lower=tensor_value,
                linear_upper=None,  # No linear dependency
                bias_upper=tensor_value,
            )
        else:
            # IBP and other methods: use IntervalBounds
            return IntervalBounds(region, tensor_value, tensor_value)

    def _get_input_bounds(self, node: Node) -> list[AbstractBounds]:
        """
        Get the bounds for all inputs to a node.

        Args:
            node: The node

        Returns:
            List of bounds for each input

        Raises:
            RuntimeError: If input bounds haven't been computed yet
        """
        input_bounds = []
        for input_node in node.inputs:
            bounds = self.get_bounds(input_node.id)
            if bounds is None:
                raise RuntimeError(
                    f"Bounds not computed for input node {input_node.id} "
                    f"(required by node {node.id})"
                )
            input_bounds.append(bounds)
        return input_bounds

    def clear_cache(self) -> None:
        """Clear the bounds cache."""
        self._bounds_cache.clear()

    def __repr__(self) -> str:
        """String representation."""
        cached = len(self._bounds_cache)
        total = len(self.graph.nodes)
        return f"BoundPropagator(method={self.method}, cached={cached}/{total})"
