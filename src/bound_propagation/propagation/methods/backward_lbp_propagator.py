"""
Backward Linear Bound Propagation (LBP) method.

Propagates bounds backward through the graph from outputs to inputs,
computing relaxations for non-linear operations.
"""

from collections.abc import Callable

import torch

from ...bounds import AbstractBounds, IntervalBounds
from ...ir import Graph, Node, NodeType, OperationType
from ...regions import AbstractRegion
from ..relaxations import RelaxationRegistry, RelaxationStrategy
from .base import (
    InputBoundKind,
    MethodPropagator,
    enumerate_input_signatures,
)


class BackwardLBPPropagator(MethodPropagator):
    """
    Backward Linear Bound Propagation.

    Propagates bounds backward through the computation graph from outputs
    to inputs. Useful for computing input bounds that satisfy output
    constraints.

    For linear operations, uses exact backward propagation.
    For non-linear operations, uses relaxations from RelaxationRegistry.
    """

    def __init__(self, *, cache=None) -> None:
        super().__init__(cache=cache)
        self._relaxation_strategies = self._load_relaxation_strategies()
        self._linear_operation_strategies: dict[
            tuple[OperationType, tuple[InputBoundKind, ...]],
            Callable[
                [Node, AbstractBounds, list[AbstractBounds], int, AbstractRegion],
                AbstractBounds,
            ],
        ] = {}
        self._register_linear_strategy_for_all_signatures(OperationType.ADD, 2, self._compute_backward_add)
        self._register_linear_strategy_for_all_signatures(OperationType.SUB, 2, self._compute_backward_sub)
        self._register_linear_strategy_for_all_signatures(OperationType.LINEAR, 1, self._compute_backward_linear_op)
        self._register_linear_strategy(
            OperationType.MATMUL,
            (InputBoundKind.ABSTRACT, InputBoundKind.CONSTANT),
            self._compute_backward_matmul,
        )
        self._register_linear_strategy(
            OperationType.MATMUL,
            (InputBoundKind.CONSTANT, InputBoundKind.CONSTANT),
            self._compute_backward_matmul,
        )

    def _register_linear_strategy(
        self,
        op_type: OperationType,
        signature: tuple[InputBoundKind, ...],
        strategy: Callable[
            [Node, AbstractBounds, list[AbstractBounds], int, AbstractRegion],
            AbstractBounds,
        ],
    ) -> None:
        self._linear_operation_strategies[(op_type, signature)] = strategy

    def _register_linear_strategy_for_all_signatures(
        self,
        op_type: OperationType,
        arity: int,
        strategy: Callable[
            [Node, AbstractBounds, list[AbstractBounds], int, AbstractRegion],
            AbstractBounds,
        ],
    ) -> None:
        for signature in enumerate_input_signatures(arity):
            self._register_linear_strategy(op_type, signature, strategy)

    def _input_signature_from_nodes(
        self,
        node: Node,
    ) -> tuple[InputBoundKind, ...]:
        signature: list[InputBoundKind] = []
        for input_node in node.inputs:
            if input_node.node_type in {NodeType.CONSTANT, NodeType.PARAMETER}:
                signature.append(InputBoundKind.CONSTANT)
            else:
                signature.append(InputBoundKind.ABSTRACT)
        return tuple(signature)

    @property
    def method_name(self) -> str:
        """Return the name of this propagation method."""
        return "backward_lbp"

    def _load_relaxation_strategies(
        self,
    ) -> dict[OperationType, RelaxationStrategy]:
        strategies: dict[OperationType, RelaxationStrategy] = {}
        for op_type in RelaxationRegistry.list_registered_ops():
            strategy = RelaxationRegistry.get(op_type)
            if strategy is None:
                raise ValueError(f"Relaxation registry reported {op_type} as registered but returned None")
            strategies[op_type] = strategy
        return strategies

    def propagate(
        self,
        graph: Graph,
        region: AbstractRegion,
        output_bounds: dict[int, AbstractBounds] | None = None,
    ) -> dict[int, AbstractBounds]:
        """
        Propagate bounds backward through the graph.

        Args:
            graph: The computation graph to propagate through.
            region: Input region for concretization.
            output_bounds: Optional output bounds. If None, uses identity bounds
                          for outputs (useful for sensitivity analysis).

        Returns:
            Dictionary mapping node IDs to their computed bounds.
            Input nodes will have bounds showing their contribution to outputs.

        Note:
            Caching for backward propagation is complex because it depends on
            output_bounds. Currently, caching is disabled for backward propagation
            with custom output_bounds. When output_bounds is None, caching works normally.
        """
        # Disable caching if custom output bounds are provided
        # (caching with output_bounds requires more complex logic)
        use_cache = self.cache is not None and output_bounds is None

        # Get nodes in reverse topological order (outputs → inputs)
        nodes = list(reversed(graph.topological_order()))

        # Initialize bounds dictionary
        bounds: dict[int, AbstractBounds] = {}

        # Initialize output bounds
        if output_bounds is None:
            # Default: use identity bounds for outputs
            for node in graph.output_nodes:
                bounds[node.id] = self._create_identity_bounds(node, region)
        else:
            bounds.update(output_bounds)

        # Propagate backward through the graph
        for node in nodes:
            # Try cache first (only if using default output bounds)
            if use_cache:
                cached_bounds = self.cache.get(node.id, self.method_name, region)
                if cached_bounds is not None:
                    bounds[node.id] = cached_bounds
                    continue

            if node.id in bounds:
                # Already computed (output node)
                continue

            if node.node_type == NodeType.CONSTANT:
                # Constants don't need backward bounds
                continue

            # Find all nodes that use this node as input
            downstream_nodes = self._find_downstream_nodes(graph, node)

            if not downstream_nodes:
                # No downstream consumers - initialize with zero bounds
                bounds[node.id] = self._create_zero_bounds(node, region)
                continue

            # Accumulate contributions from all downstream consumers
            accumulated_bounds = None

            for consumer_node in downstream_nodes:
                if consumer_node.id not in bounds:
                    continue  # Consumer not yet processed

                consumer_bounds = bounds[consumer_node.id]

                # Get input bounds for the consumer
                input_bounds = [bounds.get(inp.id, self._create_zero_bounds(inp, region)) for inp in consumer_node.inputs]

                # Find which input index we are
                input_idx = self._find_input_index(consumer_node, node)

                # Compute backward contribution
                contribution = self._compute_backward_contribution(
                    consumer_node,
                    consumer_bounds,
                    input_bounds,
                    input_idx,
                    region,
                )

                # Accumulate
                if accumulated_bounds is None:
                    accumulated_bounds = contribution
                else:
                    accumulated_bounds = self._add_bounds(accumulated_bounds, contribution)

            if accumulated_bounds is not None:
                bounds[node.id] = accumulated_bounds

                # Store in cache (only if using default output bounds)
                if use_cache:
                    self.cache.store(node.id, self.method_name, region, bounds[node.id])

        return bounds

    def _create_identity_bounds(
        self,
        node: Node,
        region: AbstractRegion,
    ) -> IntervalBounds:
        """
        Create identity bounds for a node.

        For backward propagation, output nodes start with identity bounds
        to track sensitivity. We use IntervalBounds with identity values.
        """
        shape = node.output_metadata.shape
        numel = 1
        for dim in shape:
            if dim > 0:
                numel *= dim

        device = region.device
        # Create identity interval bounds - each element has range [1, 1]
        identity_tensor = torch.ones(numel, device=device, dtype=torch.float32)

        return IntervalBounds(
            lower=identity_tensor,
            upper=identity_tensor,
        )

    def _create_zero_bounds(
        self,
        node: Node,
        region: AbstractRegion,
    ) -> IntervalBounds:
        """Create zero interval bounds for a node."""
        shape = node.output_metadata.shape
        numel = 1
        for dim in shape:
            if dim > 0:
                numel *= dim

        device = region.device
        zeros = torch.zeros(numel, device=device, dtype=torch.float32)

        return IntervalBounds(
            lower=zeros,
            upper=zeros,
        )

    def _find_downstream_nodes(self, graph: Graph, node: Node) -> list[Node]:
        """Find all nodes that use this node as input."""
        downstream = []
        for potential_consumer in graph.nodes:
            if node in potential_consumer.inputs:
                downstream.append(potential_consumer)
        return downstream

    def _find_input_index(self, consumer: Node, input_node: Node) -> int:
        """Find the index of input_node in consumer's inputs."""
        for idx, inp in enumerate(consumer.inputs):
            if inp.id == input_node.id:
                return idx
        raise ValueError(f"Node {input_node.id} not found in inputs of {consumer.id}")

    def _compute_backward_contribution(
        self,
        node: Node,
        node_bounds: AbstractBounds,
        input_bounds: list[AbstractBounds],
        input_idx: int,
        region: AbstractRegion,
    ) -> AbstractBounds:
        """
        Compute backward contribution from a node to one of its inputs.

        For linear operations, uses exact backward propagation.
        For non-linear operations, uses transposed relaxations.
        """
        op_type = node.op_type

        # Check if we have a relaxation for this operation
        if op_type in self._relaxation_strategies:
            return self._compute_backward_with_relaxation(
                node,
                node_bounds,
                input_bounds,
                input_idx,
                region,
                self._relaxation_strategies[op_type],
            )

        signature = self._input_signature_from_nodes(node)
        strategy = self._linear_operation_strategies.get((op_type, signature))
        if strategy is None:
            raise NotImplementedError(f"Backward propagation for {op_type} with input signature {signature} not yet implemented")

        return strategy(node, node_bounds, input_bounds, input_idx, region)

    def _compute_backward_with_relaxation(
        self,
        node: Node,
        node_bounds: AbstractBounds,
        input_bounds: list[AbstractBounds],
        input_idx: int,
        region: AbstractRegion,
        strategy: RelaxationStrategy,
    ) -> AbstractBounds:
        """
        Compute backward bounds using relaxation.

        For backward propagation with relaxations, we use the transpose
        of the forward relaxation to propagate bounds backward.
        """
        # Concretize input bounds to intervals
        interval_inputs = []
        for bound in input_bounds:
            if isinstance(bound, IntervalBounds):
                interval_inputs.append(bound)
            else:
                lower, upper = bound.concretize()
                interval_inputs.append(IntervalBounds(lower=lower, upper=upper))

        # Get relaxation strategy
        relaxation = strategy.relax(node, interval_inputs)

        # Get coefficients for this input
        coeff_l, coeff_u = relaxation.get_input_coeff(input_idx)

        # For backward propagation, we multiply the output bounds by the transpose
        # of the relaxation coefficients
        # If node_bounds is LinearBounds, we compose; if IntervalBounds, we use directly

        if isinstance(node_bounds, IntervalBounds):
            # Simple case: interval × coefficients
            lower_contrib = torch.where(coeff_l >= 0, coeff_l * node_bounds.lower, coeff_l * node_bounds.upper)
            upper_contrib = torch.where(coeff_u >= 0, coeff_u * node_bounds.upper, coeff_u * node_bounds.lower)
            return IntervalBounds(lower=lower_contrib, upper=upper_contrib)
        else:
            # LinearBounds case - would need composition
            # For now, concretize and use interval arithmetic
            lower, upper = node_bounds.concretize()
            interval_bound = IntervalBounds(lower=lower, upper=upper)

            lower_contrib = torch.where(coeff_l >= 0, coeff_l * interval_bound.lower, coeff_l * interval_bound.upper)
            upper_contrib = torch.where(coeff_u >= 0, coeff_u * interval_bound.upper, coeff_u * interval_bound.lower)
            return IntervalBounds(lower=lower_contrib, upper=upper_contrib)

    def _compute_backward_add(
        self,
        node: Node,
        node_bounds: AbstractBounds,
        input_bounds: list[AbstractBounds],
        input_idx: int,
        region: AbstractRegion,
    ) -> AbstractBounds:
        """Exact backward propagation for ADD."""
        return node_bounds

    def _compute_backward_sub(
        self,
        node: Node,
        node_bounds: AbstractBounds,
        input_bounds: list[AbstractBounds],
        input_idx: int,
        region: AbstractRegion,
    ) -> AbstractBounds:
        """Exact backward propagation for SUB."""
        if input_idx == 0:
            return node_bounds

        if isinstance(node_bounds, IntervalBounds):
            return IntervalBounds(
                lower=-node_bounds.upper,
                upper=-node_bounds.lower,
            )

        raise NotImplementedError("Negation of LinearBounds not yet implemented")

    def _compute_backward_matmul(
        self,
        node: Node,
        node_bounds: AbstractBounds,
        input_bounds: list[AbstractBounds],
        input_idx: int,
        region: AbstractRegion,
    ) -> AbstractBounds:
        """Exact backward propagation for MATMUL."""
        if input_idx != 0:
            raise NotImplementedError("Backward through weight matrix not yet implemented")

        weight_node = node.inputs[1]
        if weight_node.node_type != NodeType.CONSTANT:
            raise NotImplementedError("Backward MATMUL with non-constant weight not yet implemented")

        weight = weight_node.attributes.get("value")
        if weight is None:
            raise ValueError("Weight node missing value attribute")

        return self._apply_transposed_linear_map(node_bounds, weight.T, "MATMUL")

    def _compute_backward_linear_op(
        self,
        node: Node,
        node_bounds: AbstractBounds,
        input_bounds: list[AbstractBounds],
        input_idx: int,
        region: AbstractRegion,
    ) -> AbstractBounds:
        """Exact backward propagation for LINEAR."""
        if input_idx > 0:
            raise NotImplementedError("Backward through LINEAR weight/bias not yet implemented")

        weight = node.attributes.get("weight")
        if weight is None:
            raise ValueError("LINEAR operation missing weight attribute")

        return self._apply_transposed_linear_map(node_bounds, weight.T, "LINEAR")

    def _apply_transposed_linear_map(
        self,
        node_bounds: AbstractBounds,
        weight_transpose: torch.Tensor,
        operation_name: str,
    ) -> AbstractBounds:
        if not isinstance(node_bounds, IntervalBounds):
            raise NotImplementedError(f"Backward {operation_name} with LinearBounds not yet implemented")

        pos_weight = torch.clamp(weight_transpose, min=0)
        neg_weight = torch.clamp(weight_transpose, max=0)

        lower = node_bounds.lower @ pos_weight + node_bounds.upper @ neg_weight
        upper = node_bounds.upper @ pos_weight + node_bounds.lower @ neg_weight

        return IntervalBounds(lower=lower, upper=upper)

    def _add_bounds(
        self,
        bounds1: AbstractBounds,
        bounds2: AbstractBounds,
    ) -> AbstractBounds:
        """Add two bounds together (for accumulating contributions)."""
        if isinstance(bounds1, IntervalBounds) and isinstance(bounds2, IntervalBounds):
            return IntervalBounds(
                lower=bounds1.lower + bounds2.lower,
                upper=bounds1.upper + bounds2.upper,
            )
        else:
            # Would need LinearBounds addition - for now, concretize
            if not isinstance(bounds1, IntervalBounds):
                lower1, upper1 = bounds1.concretize()
                bounds1 = IntervalBounds(lower=lower1, upper=upper1)

            if not isinstance(bounds2, IntervalBounds):
                lower2, upper2 = bounds2.concretize()
                bounds2 = IntervalBounds(lower=lower2, upper=upper2)

            return IntervalBounds(
                lower=bounds1.lower + bounds2.lower,
                upper=bounds1.upper + bounds2.upper,
            )
