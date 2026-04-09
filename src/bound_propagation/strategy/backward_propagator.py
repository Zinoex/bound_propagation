"""
Backward-mode Linear Bound Propagation (LBP) propagator.

Implements Algorithm 2 from the auto_LiRPA paper: backward-mode bound propagation
on a general computational graph.

Key differences from forward propagation:
1. Traverses graph in reverse topological order (output → input)
2. Accumulates linear bounds from all downstream consumers of each node
3. For non-linearities, obtains concrete bounds by recursively applying backward
   propagation from that node to the input
4. Final bounds at INPUT nodes represent how OUTPUT depends on inputs
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from ..bounds import AbstractBounds, LinearBounds
from ..regions import HyperRectangle
from .config import StrategyConfig
from .registry import get_global_registry

if TYPE_CHECKING:
    from ..ir import Graph, Node
    from .registry import StrategyRegistry


class BackwardBoundPropagator:
    """
    Backward-mode Linear Bound Propagation using reverse topological traversal.

    Algorithm overview:
    1. Initialize output nodes with identity linear bounds (A_o = I, Ā_o = I)
    2. Traverse nodes in reverse topological order (output → input)
    3. For each node i:
       - Collect linear bounds from all nodes j that depend on i (downstream)
       - For non-linear operations: recursively compute backward bounds from i to input
         to obtain concrete bounds for relaxation (not a separate forward pass)
       - Propagate these bounds backward through operation j to contribute to A_i, Ā_i
       - Accumulate contributions (A_i += ...)
    4. Final bounds at INPUT nodes represent how OUTPUT depends on inputs
    5. To get output bounds: concretize the input bounds with the input region

    Note: For hybrid approaches (IBP+backward, forward+backward), a full forward pass
          is done first to provide concrete bounds. Pure backward computes bounds
          on-demand via recursive backward propagation.

    Example:
        # Pure backward LBP
        propagator = BackwardLBPPropagator(graph)
        propagator.compute_bounds(input_region)

        # Get bounds on output by concretizing input dependencies
        input_bounds = propagator.get_input_bounds()[0]
        output_lower, output_upper = input_bounds.concretize()

        # Or for hybrid approach with pre-computed forward bounds:
        backward_prop = BackwardLBPPropagator(graph, forward_bounds=forward_bounds)
        backward_bounds = backward_prop.compute_bounds(input_region)
    """

    def __init__(
        self,
        graph: Graph,
        forward_bounds: dict[int, AbstractBounds] | None = None,
        registry: StrategyRegistry | None = None,
        config: StrategyConfig | None = None,
    ):
        """
        Initialize backward LBP propagator.

        Args:
            graph: The computation graph
            forward_bounds: Pre-computed concrete bounds from forward pass (optional)
                          If None, will compute IBP bounds automatically
            registry: Strategy registry (uses global registry if None)
            config: Default strategy configuration
        """
        self.graph = graph
        self.registry = registry if registry is not None else get_global_registry()
        self.config = config if config is not None else StrategyConfig()

        # Concrete bounds from forward pass (needed for relaxations)
        self._forward_bounds = forward_bounds or {}

        # Linear bounds being accumulated in backward pass
        self._backward_bounds: dict[int, AbstractBounds] = {}

        # Track which nodes haven't been processed yet
        self._unprocessed_count: dict[int, int] = {}

        # Map each node to the list of nodes that use it
        self._users: dict[int, list[Node]] = defaultdict(list)

    def compute_bounds(
        self,
        input_region: HyperRectangle,
        node_configs: dict[int, StrategyConfig] | None = None,
    ) -> dict[int, AbstractBounds]:
        """
        Compute backward LBP bounds for all nodes.

        Args:
            input_region: The input region specification
            node_configs: Optional per-node strategy configurations

        Returns:
            Dictionary mapping node ID to computed linear bounds

        Raises:
            ValueError: If no strategy is available for a node
            RuntimeError: If bound computation fails
        """
        node_configs = node_configs or {}

        # Step 1: Ensure we have concrete bounds
        # TODO: Update this to recursively build input bounds via backward propagation if not provided
        # if not self._forward_bounds:
        #     self._compute_forward_bounds(input_region)

        # Step 2: Build dependency mappings
        self._build_dependency_mapping()

        # Step 3: Initialize output nodes with identity bounds
        self._initialize_output_bounds(input_region)

        # Step 4: Traverse in reverse topological order and accumulate bounds
        self._backward_pass(input_region)

        return self._backward_bounds

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
            bounds = self._backward_bounds.get(node.id)
            if bounds is None:
                raise RuntimeError(
                    f"Bounds not computed for output node {node.id}. "
                    "Call compute_bounds() first."
                )
            output_bounds.append(bounds)
        return output_bounds

    def get_input_bounds(self) -> list[AbstractBounds]:
        """
        Get backward bounds for input nodes.

        These represent how the output depends on the inputs.

        Returns:
            List of linear bounds for input nodes
        """
        input_bounds = []
        for node in self.graph.input_nodes:
            bounds = self._backward_bounds.get(node.id)
            if bounds is None:
                raise RuntimeError(
                    f"Bounds not computed for input node {node.id}. "
                    "Call compute_bounds() first."
                )
            input_bounds.append(bounds)
        return input_bounds

    def _build_dependency_mapping(self) -> None:
        """
        Build mapping from each node to nodes that depend on it.

        Also initialize unprocessed count for each node.
        """
        self._users.clear()
        self._unprocessed_count.clear()

        # Build users mapping: for each node, which nodes use it as input
        for node in self.graph.nodes:
            # Initialize unprocessed count as number of output nodes this depends on
            # (will be updated during traversal)
            self._unprocessed_count[node.id] = 0

            # Register this node as a user of each of its inputs
            for input_node in node.inputs:
                self._users[input_node.id].append(node)

        # Initialize output nodes as having 0 unprocessed dependencies
        for node in self.graph.output_nodes:
            self._unprocessed_count[node.id] = 0

    def _initialize_output_bounds(self, input_region: HyperRectangle) -> None:
        """
        Initialize output nodes with identity linear bounds.

        A_o = I (identity matrix)
        Ā_o = I (identity matrix)

        This represents: output_lower = I @ output + 0, output_upper = I @ output + 0
        """
        for node in self.graph.output_nodes:
            # Get output shape from forward bounds
            forward_bound = self._forward_bounds[node.id]
            lower, upper = forward_bound.concretize()
            output_size = lower.numel()

            # Create identity linear bounds
            identity = torch.eye(
                output_size,
                dtype=input_region.dtype,
                device=input_region.device
            )
            bias = torch.zeros(
                output_size,
                dtype=input_region.dtype,
                device=input_region.device
            )

            self._backward_bounds[node.id] = LinearBounds(
                region=input_region,
                linear_lower=identity,
                bias_lower=bias,
                linear_upper=identity,
                bias_upper=bias,
            )

    def _backward_pass(
        self,
        input_region: HyperRectangle,
    ) -> None:
        """
        Main backward pass: traverse in reverse order and accumulate bounds.

        For each node i in reverse topological order:
        - For each node j that uses i:
          - Propagate A_j, Ā_j backward through operation j
          - Accumulate contribution to A_i, Ā_i
        """
        # Process nodes in reverse topological order
        for node in self.graph.reverse_topological_order():
            # Skip if this node has no users (unreachable from output)
            if node.id not in self._users and not node.is_output:
                continue

            # Process this node: propagate bounds from users back through this node
            self._process_node_backward(node, input_region)

    def _process_node_backward(
        self,
        node: Node,
        input_region: HyperRectangle,
    ) -> None:
        """
        Process a single node in backward pass.

        Collect linear bounds from all users (nodes that depend on this node)
        and propagate them backward.
        """
        # If this is an output node, it already has identity bounds initialized
        # Otherwise, we need to accumulate from users

        if not node.is_output:
            # For non-output nodes, collect bounds from all users
            users = self._users.get(node.id, [])

            for user_node in users:
                # Get the linear bounds for the user node
                user_bounds = self._backward_bounds.get(user_node.id)
                if user_bounds is None:
                    # User hasn't been processed yet, skip
                    continue

                # Propagate user_bounds backward through user_node's operation
                # to get contribution to this node
                contribution = self._propagate_backward_through_operation(
                    user_node,
                    node,
                    user_bounds,
                    input_region,
                )

                # Accumulate contribution
                if node.id in self._backward_bounds:
                    self._backward_bounds[node.id] = self._add_linear_bounds(
                        self._backward_bounds[node.id],
                        contribution
                    )
                else:
                    self._backward_bounds[node.id] = contribution

    def _propagate_backward_through_operation(
        self,
        operation_node: Node,
        input_node: Node,
        output_bounds: AbstractBounds,
        input_region: HyperRectangle,
    ) -> AbstractBounds:
        """
        Propagate linear bounds backward through an operation.

        Given:
        - operation_node: the node computing z = f(x, y, ...)
        - input_node: one of the inputs to operation_node (e.g., x)
        - output_bounds: linear bounds A_z, Ā_z for the operation output

        Compute:
        - Contribution to A_x, Ā_x representing how changes to x affect the final output

        Args:
            operation_node: The operation node
            input_node: The specific input we're propagating to
            output_bounds: Linear bounds for the operation output
            input_region: Input region specification
            config: Strategy configuration

        Returns:
            Linear bounds representing the contribution to the input node
        """
        # Get the strategy for this operation
        strategy = self.registry.get(operation_node.op_type, "backward")
        if strategy is None:
            raise ValueError(
                f"No backward strategy registered for operation {operation_node.op_type}"
            )

        # Get concrete bounds for relaxations
        concrete_bounds = [
            self._forward_bounds[inp.id] for inp in operation_node.inputs
        ]

        # Call strategy to propagate backward
        # The strategy should compute: how does output_bounds relate to this specific input
        contribution = strategy.propagate_backward(
            operation_node,
            output_bounds,
            concrete_bounds,
        )

        return contribution

    # Wrong: we need to accumulate contributions from multiple users, not just add two bounds together
    #        and the handling of biases is wrong. 
    # def _add_linear_bounds(
    #     self,
    #     bounds1: LinearBounds,
    #     bounds2: LinearBounds
    # ) -> LinearBounds:
    #     """
    #     Add two LinearBounds by summing their linear coefficients and biases.

    #     This implements the accumulation: A_i += contribution
    #     """
    #     # Add linear coefficients
    #     if bounds1.linear_lower is not None and bounds2.linear_lower is not None:
    #         linear_lower = bounds1.linear_lower + bounds2.linear_lower
    #     elif bounds1.linear_lower is not None:
    #         linear_lower = bounds1.linear_lower
    #     elif bounds2.linear_lower is not None:
    #         linear_lower = bounds2.linear_lower
    #     else:
    #         linear_lower = None

    #     if bounds1.linear_upper is not None and bounds2.linear_upper is not None:
    #         linear_upper = bounds1.linear_upper + bounds2.linear_upper
    #     elif bounds1.linear_upper is not None:
    #         linear_upper = bounds1.linear_upper
    #     elif bounds2.linear_upper is not None:
    #         linear_upper = bounds2.linear_upper
    #     else:
    #         linear_upper = None

    #     # Add biases
    #     bias_lower = bounds1.bias_lower + bounds2.bias_lower
    #     bias_upper = bounds1.bias_upper + bounds2.bias_upper

    #     return LinearBounds(
    #         region=bounds1.region,
    #         linear_lower=linear_lower,
    #         bias_lower=bias_lower,
    #         linear_upper=linear_upper,
    #         bias_upper=bias_upper,
    #     )
