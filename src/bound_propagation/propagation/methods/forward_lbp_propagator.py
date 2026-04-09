"""
Forward Linear Bound Propagation (LBP) method.

Propagates linear bounds forward through the graph in topological order,
computing relaxations for non-linear operations.
"""

from typing import Dict, List, Optional

import torch

from bound_propagation.bounds.abstract_bounds import AbstractBounds
from bound_propagation.bounds.interval_bounds import IntervalBounds
from bound_propagation.bounds.linear_bounds import LinearBounds
from bound_propagation.ir.graph import Graph
from bound_propagation.ir.node import Node, NodeType
from bound_propagation.ir.operations import OperationType
from bound_propagation.propagation.methods.base import MethodPropagator
from bound_propagation.regions.abstract import AbstractRegion
from bound_propagation.regions.hyperrectangle import HyperRectangle
from bound_propagation.regions.multi_input import MultiInputRegion
from bound_propagation.relaxations import RelaxationRegistry


class ForwardLBPPropagator(MethodPropagator):
    """
    Forward Linear Bound Propagation.
    
    Propagates linear bounds (affine functions of inputs) forward through
    the computation graph. For non-linear operations, computes relaxations
    using the RelaxationRegistry.
    
    The key insight is that relaxations are computed once and reused,
    while propagation strategies handle composition of linear bounds.
    """
    
    # No custom __init__ needed - uses base class __init__
    
    @property
    def method_name(self) -> str:
        return "forward_lbp"
    
    def propagate(
        self,
        graph: Graph,
        region: AbstractRegion,
        start_node: Optional[int] = None,
    ) -> Dict[int, AbstractBounds]:
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
        bounds: Dict[int, AbstractBounds] = {}
        
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
                bounds[node.id] = self._compute_operation_bounds(
                    node, input_bounds, region
                )
            
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
        input_bounds: List[AbstractBounds],
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
        op_type = node.op_type
        
        # Check if we need a relaxation for this operation
        if RelaxationRegistry.has_strategy(op_type):
            # Non-linear operation: compute relaxation
            return self._compute_with_relaxation(node, input_bounds, region)
        else:
            # Linear operation or not yet implemented
            # For now, fall back to interval bounds via concretization
            return self._compute_interval_fallback(node, input_bounds)
    
    def _compute_with_relaxation(
        self,
        node: Node,
        input_bounds: List[AbstractBounds],
        region: AbstractRegion,
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
        strategy = RelaxationRegistry.get(node.op_type)
        if strategy is None:
            raise ValueError(f"No relaxation strategy for {node.op_type}")
        
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
            lower_contrib = torch.where(
                coeff_l >= 0,
                coeff_l * interval_input.lower,
                coeff_l * interval_input.upper
            )
            output_lower_contrib.append(lower_contrib)
            
            # For upper bound: maximize coeff_u * x
            # If coeff_u >= 0: maximum is coeff_u * upper
            # If coeff_u < 0: maximum is coeff_u * lower
            upper_contrib = torch.where(
                coeff_u >= 0,
                coeff_u * interval_input.upper,
                coeff_u * interval_input.lower
            )
            output_upper_contrib.append(upper_contrib)
        
        # Combine contributions
        output_lower = sum(output_lower_contrib) + relaxation.bias_lower
        output_upper = sum(output_upper_contrib) + relaxation.bias_upper
        
        return IntervalBounds(lower=output_lower, upper=output_upper)
    
    def _compute_interval_fallback(
        self,
        node: Node,
        input_bounds: List[AbstractBounds],
    ) -> IntervalBounds:
        """
        Exact propagation for linear operations using interval arithmetic.
        
        Args:
            node: The operation node.
            input_bounds: Bounds for inputs.
        
        Returns:
            Interval bounds computed via exact propagation.
        """
        op_type = node.op_type
        
        # Concretize all input bounds to intervals
        intervals = []
        for bound in input_bounds:
            if isinstance(bound, IntervalBounds):
                intervals.append(bound)
            else:
                lower, upper = bound.concretize()
                intervals.append(IntervalBounds(lower=lower, upper=upper))
        
        # Exact propagation for linear operations
        if op_type == OperationType.ADD:
            return self._propagate_add(intervals)
        elif op_type == OperationType.SUB:
            return self._propagate_sub(intervals)
        elif op_type == OperationType.MATMUL:
            return self._propagate_matmul(node, intervals)
        elif op_type == OperationType.LINEAR:
            return self._propagate_linear(node, intervals)
        else:
            raise NotImplementedError(
                f"Exact propagation not implemented for {op_type}"
            )
    
    def _propagate_add(self, intervals: List[IntervalBounds]) -> IntervalBounds:
        """Exact interval propagation for ADD: [a,b] + [c,d] = [a+c, b+d]."""
        if len(intervals) != 2:
            raise ValueError(f"ADD expects 2 inputs, got {len(intervals)}")
        
        lower = intervals[0].lower + intervals[1].lower
        upper = intervals[0].upper + intervals[1].upper
        return IntervalBounds(lower=lower, upper=upper)
    
    def _propagate_sub(self, intervals: List[IntervalBounds]) -> IntervalBounds:
        """Exact interval propagation for SUB: [a,b] - [c,d] = [a-d, b-c]."""
        if len(intervals) != 2:
            raise ValueError(f"SUB expects 2 inputs, got {len(intervals)}")
        
        lower = intervals[0].lower - intervals[1].upper
        upper = intervals[0].upper - intervals[1].lower
        return IntervalBounds(lower=lower, upper=upper)
    
    def _propagate_matmul(
        self, node: Node, intervals: List[IntervalBounds]
    ) -> IntervalBounds:
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
            raise NotImplementedError(
                "MATMUL with two non-constant inputs not yet implemented"
            )
    
    def _propagate_linear(
        self, node: Node, intervals: List[IntervalBounds]
    ) -> IntervalBounds:
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
