"""
Forward Linear Bound Propagation (LBP) method.

Propagates linear bounds forward through the graph in topological order,
computing relaxations for non-linear operations.
"""

from collections.abc import Callable

import torch

from ...bounds import AbstractBounds, IntervalBounds, LinearBounds
from ...ir import Graph, Node, NodeType, OperationType
from ...regions import AbstractRegion, MultiInputRegion
from ..relaxations import RelaxationRegistry, RelaxationStrategy
from .base import (
    InputBoundKind,
    MethodPropagator,
    classify_input_bound,
    classify_input_signature,
    enumerate_input_signatures,
)


class ForwardLBPPropagator(MethodPropagator):
    """
    Forward Linear Bound Propagation.

    Propagates linear bounds (affine functions of inputs) forward through
    the computation graph. For non-linear operations, computes relaxations
    using the RelaxationRegistry.

    The key insight is that relaxations are computed once and reused,
    while propagation strategies handle composition of linear bounds.
    """

    def __init__(self, *, cache=None) -> None:
        super().__init__(cache=cache)
        self._exact_operation_strategies: dict[
            tuple[OperationType, tuple[InputBoundKind, ...]],
            Callable[[Node, list[AbstractBounds]], AbstractBounds],
        ] = {}
        self._register_exact_strategy_for_all_signatures(OperationType.ADD, 2, self._propagate_add_exact_strategy)
        self._register_exact_strategy_for_all_signatures(OperationType.SUB, 2, self._propagate_sub_exact_strategy)
        self._register_exact_strategy_for_all_signatures(OperationType.LINEAR, 1, self._propagate_linear_exact_strategy)
        self._register_exact_strategy(
            OperationType.MATMUL,
            (InputBoundKind.CONSTANT, InputBoundKind.CONSTANT),
            self._propagate_matmul_exact_strategy,
        )
        self._register_exact_strategy(
            OperationType.MATMUL,
            (InputBoundKind.CONSTANT, InputBoundKind.ABSTRACT),
            self._propagate_matmul_exact_strategy,
        )
        self._register_exact_strategy(
            OperationType.MATMUL,
            (InputBoundKind.ABSTRACT, InputBoundKind.CONSTANT),
            self._propagate_matmul_exact_strategy,
        )
        self._register_exact_strategy(
            OperationType.MUL,
            (InputBoundKind.CONSTANT, InputBoundKind.ABSTRACT),
            self._propagate_mul_by_constant_strategy,
        )
        self._register_exact_strategy(
            OperationType.MUL,
            (InputBoundKind.ABSTRACT, InputBoundKind.CONSTANT),
            self._propagate_mul_by_constant_strategy,
        )
        self._register_exact_strategy(
            OperationType.MUL,
            (InputBoundKind.CONSTANT, InputBoundKind.CONSTANT),
            self._propagate_mul_by_constant_strategy,
        )
        self._register_exact_strategy(
            OperationType.DIV,
            (InputBoundKind.CONSTANT, InputBoundKind.CONSTANT),
            self._propagate_div_by_constant_strategy,
        )
        self._register_exact_strategy(
            OperationType.DIV,
            (InputBoundKind.ABSTRACT, InputBoundKind.CONSTANT),
            self._propagate_div_by_constant_strategy,
        )
        self._relaxation_strategies = self._load_relaxation_strategies()

    def _register_exact_strategy(
        self,
        op_type: OperationType,
        signature: tuple[InputBoundKind, ...],
        strategy: Callable[[Node, list[AbstractBounds]], AbstractBounds],
    ) -> None:
        self._exact_operation_strategies[(op_type, signature)] = strategy

    def _register_exact_strategy_for_all_signatures(
        self,
        op_type: OperationType,
        arity: int,
        strategy: Callable[[Node, list[AbstractBounds]], AbstractBounds],
    ) -> None:
        for signature in enumerate_input_signatures(arity):
            self._register_exact_strategy(op_type, signature, strategy)

    @property
    def method_name(self) -> str:
        return "forward_lbp"

    def propagate(
        self,
        graph: Graph,
        region: AbstractRegion,
        start_node: int | None = None,
    ) -> dict[int, AbstractBounds]:
        """
        Propagate linear bounds forward through the graph.

        Args:
            graph: The computation graph.
            region: Input region (e.g., HyperRectangle) defining input bounds.
            start_node: Optional node ID to compute bounds for (not yet implemented).

        Returns:
            Dictionary mapping node IDs to their linear bounds.
        """
        # Dictionary to store bounds for each node
        bounds: dict[int, AbstractBounds] = {}

        # Process nodes in topological order
        for node in graph.topological_order():
            # Try to get from cache first
            if self.cache is not None:
                cached_bounds = self.cache.get(node.id, self.method_name, region)
                if cached_bounds is not None:
                    bounds[node.id] = cached_bounds
                    continue

            # Compute bounds for this node
            if node.node_type == NodeType.INPUT:
                # Create identity linear bounds for inputs
                bounds[node.id] = self._create_input_bounds(node, region)
            elif node.node_type == NodeType.CONSTANT:
                # Create constant bounds
                bounds[node.id] = self._create_constant_bounds(node)
            elif node.node_type == NodeType.PARAMETER:
                # Create constant bounds for parameters
                bounds[node.id] = self._create_constant_bounds(node)
            else:
                # Operation node: compute bounds
                input_bounds = [bounds[inp.id] for inp in node.inputs]
                bounds[node.id] = self._compute_operation_bounds(node, input_bounds, region)

            # Store in cache
            if self.cache is not None:
                self.cache.store(node.id, self.method_name, region, bounds[node.id])

        return bounds

    def _create_input_bounds(
        self,
        node: Node,
        region: AbstractRegion,
    ) -> IntervalBounds:
        """
        Create interval bounds for an input node from the region.

        For forward LBP, we start with interval bounds from the input region.
        These will be lifted to linear bounds as needed during propagation.

        Args:
            node: The input node.
            region: Input region containing the input. Can be HyperRectangle
                   (single input) or MultiInputRegion (multiple inputs).

        Returns:
            IntervalBounds from the input region.
        """
        # Handle multi-input regions by looking up the node's region
        if isinstance(region, MultiInputRegion):
            if node.id not in region:
                raise ValueError(f"Input node {node.id} not found in MultiInputRegion. Available inputs: {list(region.keys())}")
            node_region = region[node.id]
            return IntervalBounds(
                lower=node_region.lower.clone(),
                upper=node_region.upper.clone(),
            )
        else:
            # Single input region - use directly
            return IntervalBounds(
                lower=region.lower.clone(),
                upper=region.upper.clone(),
            )

    def _create_constant_bounds(self, node: Node) -> IntervalBounds:
        """
        Create point bounds for a constant node.

        Args:
            node: The constant or parameter node.

        Returns:
            IntervalBounds with zero width (point bound).
        """
        # Get the constant value from node attributes
        value = node.attributes.get("value")
        if value is None:
            # For parameters without explicit value, use a placeholder
            # This should be handled better in a full implementation
            shape = node.output_metadata.shape
            value = torch.zeros(shape)

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        # Point bounds: lower = upper = value
        return IntervalBounds(lower=value, upper=value)

    def _compute_operation_bounds(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        region: AbstractRegion,
    ) -> AbstractBounds:
        """
        Compute bounds for an operation node.

        For linear operations, propagation is exact.
        For non-linear operations, we:
        1. Concretize input bounds to intervals
        2. Compute relaxation using RelaxationRegistry
        3. Apply relaxation via composition (TODO)

        Args:
            node: The operation node.
            input_bounds: Bounds for the node's inputs.
            region: Input region for concretization.

        Returns:
            Computed bounds for this operation.
        """
        signature = classify_input_signature(input_bounds)
        exact_strategy = self._exact_operation_strategies.get((node.op_type, signature))
        if exact_strategy is not None:
            return exact_strategy(node, input_bounds)

        # Check if we need a relaxation for this operation.
        if node.op_type in self._relaxation_strategies:
            # Non-linear operation: compute relaxation
            return self._compute_with_relaxation(
                node,
                input_bounds,
                region,
                self._relaxation_strategies[node.op_type],
            )

        # Linear operation or not yet implemented.
        # For now, fall back to interval bounds via concretization.
        return self._compute_interval_fallback(node, input_bounds)

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

    def _compute_with_relaxation(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
        region: AbstractRegion,
        strategy: RelaxationStrategy,
    ) -> AbstractBounds:
        """
        Compute bounds using relaxation for non-linear operations.

        Args:
            node: The operation node.
            input_bounds: Bounds for inputs.
            region: Input region.

        Returns:
            Bounds after applying relaxation.
        """
        # Step 1: Concretize input bounds to intervals
        interval_inputs = []
        for bound in input_bounds:
            if isinstance(bound, IntervalBounds):
                interval_inputs.append(bound)
            elif isinstance(bound, LinearBounds):
                # Concretize linear bounds using the region
                lower, upper = bound.concretize()
                interval_inputs.append(IntervalBounds(lower=lower, upper=upper))
            else:
                # Fallback: try concretize method
                lower, upper = bound.concretize()
                interval_inputs.append(IntervalBounds(lower=lower, upper=upper))

        # Step 2: Get relaxation strategy and compute relaxation
        relaxation = strategy.relax(node, interval_inputs)

        # Step 3: Apply relaxation using proper interval arithmetic
        # For element-wise operations with diagonal relaxations, we use interval arithmetic
        # that correctly handles positive and negative coefficients

        output_lower_contrib = []
        output_upper_contrib = []

        for i, interval_input in enumerate(interval_inputs):
            coeff_l, coeff_u = relaxation.get_input_coeff(i)

            # For lower bound: minimize coeff_l * x
            # If coeff_l >= 0: minimum is coeff_l * lower
            # If coeff_l < 0: minimum is coeff_l * upper
            lower_contrib = torch.where(coeff_l >= 0, coeff_l * interval_input.lower, coeff_l * interval_input.upper)
            output_lower_contrib.append(lower_contrib)

            # For upper bound: maximize coeff_u * x
            # If coeff_u >= 0: maximum is coeff_u * upper
            # If coeff_u < 0: maximum is coeff_u * lower
            upper_contrib = torch.where(coeff_u >= 0, coeff_u * interval_input.upper, coeff_u * interval_input.lower)
            output_upper_contrib.append(upper_contrib)

        # Combine contributions
        output_lower = sum(output_lower_contrib) + relaxation.bias_lower
        output_upper = sum(output_upper_contrib) + relaxation.bias_upper

        return IntervalBounds(lower=output_lower, upper=output_upper)

    def _compute_interval_fallback(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
    ) -> IntervalBounds:
        """
        Exact propagation for linear operations using interval arithmetic.

        Args:
            node: The operation node.
            input_bounds: Bounds for inputs.

        Returns:
            Interval bounds computed via exact propagation.
        """
        # Concretize all input bounds to intervals
        intervals = []
        for bound in input_bounds:
            if isinstance(bound, IntervalBounds):
                intervals.append(bound)
            else:
                lower, upper = bound.concretize()
                intervals.append(IntervalBounds(lower=lower, upper=upper))

        signature = classify_input_signature(input_bounds)
        exact_strategy = self._exact_operation_strategies.get((node.op_type, signature))
        if exact_strategy is None:
            raise NotImplementedError(f"Exact propagation not implemented for {node.op_type} with input signature {signature}")

        result = exact_strategy(node, input_bounds)
        if not isinstance(result, IntervalBounds):
            lower, upper = result.concretize()
            return IntervalBounds(lower=lower, upper=upper)
        return result

    def _propagate_add_exact_strategy(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
    ) -> IntervalBounds:
        return self._propagate_add(self._concretize_to_intervals(input_bounds))

    def _propagate_sub_exact_strategy(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
    ) -> IntervalBounds:
        return self._propagate_sub(self._concretize_to_intervals(input_bounds))

    def _propagate_matmul_exact_strategy(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
    ) -> IntervalBounds:
        return self._propagate_matmul(node, self._concretize_to_intervals(input_bounds))

    def _propagate_linear_exact_strategy(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
    ) -> IntervalBounds:
        return self._propagate_linear(node, self._concretize_to_intervals(input_bounds))

    def _propagate_mul_by_constant_strategy(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
    ) -> IntervalBounds:
        intervals = self._concretize_to_intervals(input_bounds)
        left_kind = classify_input_bound(input_bounds[0])
        right_kind = classify_input_bound(input_bounds[1])

        if right_kind == InputBoundKind.CONSTANT:
            return self._scale_interval_by_constant(intervals[0], intervals[1].lower)

        if left_kind == InputBoundKind.CONSTANT:
            return self._scale_interval_by_constant(intervals[1], intervals[0].lower)

        raise ValueError("constant multiplication strategy requires at least one constant input")

    def _propagate_div_by_constant_strategy(
        self,
        node: Node,
        input_bounds: list[AbstractBounds],
    ) -> IntervalBounds:
        intervals = self._concretize_to_intervals(input_bounds)
        divisor = intervals[1].lower
        if not torch.allclose(intervals[1].lower, intervals[1].upper):
            raise ValueError("constant division strategy requires a constant divisor")
        return self._divide_interval_by_constant(intervals[0], divisor)

    def _concretize_to_intervals(
        self,
        input_bounds: list[AbstractBounds],
    ) -> list[IntervalBounds]:
        intervals: list[IntervalBounds] = []
        for bound in input_bounds:
            if isinstance(bound, IntervalBounds):
                intervals.append(bound)
            else:
                lower, upper = bound.concretize()
                intervals.append(IntervalBounds(lower=lower, upper=upper))
        return intervals

    def _scale_interval_by_constant(
        self,
        interval: IntervalBounds,
        constant: torch.Tensor,
    ) -> IntervalBounds:
        lower = torch.where(
            constant >= 0,
            interval.lower * constant,
            interval.upper * constant,
        )
        upper = torch.where(
            constant >= 0,
            interval.upper * constant,
            interval.lower * constant,
        )
        return IntervalBounds(lower=lower, upper=upper)

    def _divide_interval_by_constant(
        self,
        interval: IntervalBounds,
        divisor: torch.Tensor,
    ) -> IntervalBounds:
        if torch.any(divisor == 0):
            raise ValueError("division by zero in constant division strategy")

        lower = torch.where(
            divisor > 0,
            interval.lower / divisor,
            interval.upper / divisor,
        )
        upper = torch.where(
            divisor > 0,
            interval.upper / divisor,
            interval.lower / divisor,
        )
        return IntervalBounds(lower=lower, upper=upper)

    def _propagate_add(self, intervals: list[IntervalBounds]) -> IntervalBounds:
        """Exact interval propagation for ADD: [a,b] + [c,d] = [a+c, b+d]."""
        if len(intervals) != 2:
            raise ValueError(f"ADD expects 2 inputs, got {len(intervals)}")

        lower = intervals[0].lower + intervals[1].lower
        upper = intervals[0].upper + intervals[1].upper
        return IntervalBounds(lower=lower, upper=upper)

    def _propagate_sub(self, intervals: list[IntervalBounds]) -> IntervalBounds:
        """Exact interval propagation for SUB: [a,b] - [c,d] = [a-d, b-c]."""
        if len(intervals) != 2:
            raise ValueError(f"SUB expects 2 inputs, got {len(intervals)}")

        lower = intervals[0].lower - intervals[1].upper
        upper = intervals[0].upper - intervals[1].lower
        return IntervalBounds(lower=lower, upper=upper)

    def _propagate_matmul(self, node: Node, intervals: list[IntervalBounds]) -> IntervalBounds:
        """Exact interval propagation for MATMUL: y = x @ W."""
        if len(intervals) != 2:
            raise ValueError(f"MATMUL expects 2 inputs, got {len(intervals)}")

        x_interval = intervals[0]
        w_interval = intervals[1]

        # For matrix multiplication, we need to consider all combinations
        # y[i,k] = sum_j x[i,j] * w[j,k]
        # To bound this, we use: for each element, compute min/max over corners

        # Simplified: assume W is constant (common case)
        if torch.allclose(w_interval.lower, w_interval.upper):
            W = w_interval.lower

            # y = x @ W, where x in [x_l, x_u]
            # For each column k: y[:,k] = x @ W[:,k]
            # We compute bounds by considering positive/negative coefficients

            pos_W = torch.clamp(W, min=0)
            neg_W = torch.clamp(W, max=0)

            lower = x_interval.lower @ pos_W + x_interval.upper @ neg_W
            upper = x_interval.upper @ pos_W + x_interval.lower @ neg_W

            return IntervalBounds(lower=lower, upper=upper)
        else:
            # Both inputs are non-constant - use bilinear bounds
            # This is conservative but sound
            raise NotImplementedError("MATMUL with two non-constant inputs not yet implemented")

    def _propagate_linear(self, node: Node, intervals: list[IntervalBounds]) -> IntervalBounds:
        """Exact interval propagation for LINEAR: y = x @ W + b."""
        # LINEAR typically has x as learnable, W and b as constants
        if len(intervals) == 1:
            # W and b are in node attributes
            x_interval = intervals[0]
            W = node.attributes.get("weight")
            b = node.attributes.get("bias")

            if W is None:
                raise ValueError("LINEAR operation missing 'weight' attribute")

            # y = x @ W + b
            # First compute x @ W using matmul logic
            pos_W = torch.clamp(W, min=0)
            neg_W = torch.clamp(W, max=0)

            matmul_lower = x_interval.lower @ pos_W + x_interval.upper @ neg_W
            matmul_upper = x_interval.upper @ pos_W + x_interval.lower @ neg_W

            # Then add bias
            if b is not None:
                lower = matmul_lower + b
                upper = matmul_upper + b
            else:
                lower = matmul_lower
                upper = matmul_upper

            return IntervalBounds(lower=lower, upper=upper)
        else:
            # Multiple inputs - treat as matmul + add
            matmul_result = self._propagate_matmul(node, intervals[:2])
            if len(intervals) > 2:
                # Add bias
                return self._propagate_add([matmul_result, intervals[2]])
            return matmul_result
