"""
Backward Linear Bound Propagation (LBP) method.

Propagates bounds backward through the graph from outputs to inputs,
computing relaxations for non-linear operations.
"""

from typing import Dict, List

import torch

from bound_propagation.bounds.abstract_bounds import AbstractBounds
from bound_propagation.bounds.interval_bounds import IntervalBounds
from bound_propagation.bounds.linear_bounds import LinearBounds
from bound_propagation.ir.graph import Graph
from bound_propagation.ir.node import Node, NodeType
from bound_propagation.ir.operations import OperationType
from bound_propagation.propagation.methods.base import MethodPropagator
from bound_propagation.regions.abstract import AbstractRegion
from bound_propagation.relaxations import RelaxationRegistry


class BackwardLBPPropagator(MethodPropagator):
    """
    Backward Linear Bound Propagation.
    
    Propagates bounds backward through the computation graph from outputs
    to inputs. Useful for computing input bounds that satisfy output
    constraints.
    
    For linear operations, uses exact backward propagation.
    For non-linear operations, uses relaxations from RelaxationRegistry.
    """
    
    @property
    def method_name(self) -> str:
        """Return the name of this propagation method."""
        return "backward_lbp"
    
    def propagate(
        self,
        graph: Graph,
        region: AbstractRegion,
        output_bounds: Dict[int, AbstractBounds] | None = None,
    ) -> Dict[int, AbstractBounds]:
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
        bounds: Dict[int, AbstractBounds] = {}
        
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
                input_bounds = [
                    bounds.get(inp.id, self._create_zero_bounds(inp, region))
                    for inp in consumer_node.inputs
                ]
                
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
    
    def _find_downstream_nodes(self, graph: Graph, node: Node) -> List[Node]:
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
        input_bounds: List[AbstractBounds],
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
        if RelaxationRegistry.has_strategy(op_type):
            return self._compute_backward_with_relaxation(
                node, node_bounds, input_bounds, input_idx, region
            )
        else:
            # Linear operation - exact backward propagation
            return self._compute_backward_linear(
                node, node_bounds, input_bounds, input_idx, region
            )
    
    def _compute_backward_with_relaxation(
        self,
        node: Node,
        node_bounds: AbstractBounds,
        input_bounds: List[AbstractBounds],
        input_idx: int,
        region: AbstractRegion,
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
        strategy = RelaxationRegistry.get(node.op_type)
        if strategy is None:
            raise ValueError(f"No relaxation strategy for {node.op_type}")
        
        relaxation = strategy.relax(node, interval_inputs)
        
        # Get coefficients for this input
        coeff_l, coeff_u = relaxation.get_input_coeff(input_idx)
        
        # For backward propagation, we multiply the output bounds by the transpose
        # of the relaxation coefficients
        # If node_bounds is LinearBounds, we compose; if IntervalBounds, we use directly
        
        if isinstance(node_bounds, IntervalBounds):
            # Simple case: interval × coefficients
            lower_contrib = torch.where(
                coeff_l >= 0,
                coeff_l * node_bounds.lower,
                coeff_l * node_bounds.upper
            )
            upper_contrib = torch.where(
                coeff_u >= 0,
                coeff_u * node_bounds.upper,
                coeff_u * node_bounds.lower
            )
            return IntervalBounds(lower=lower_contrib, upper=upper_contrib)
        else:
            # LinearBounds case - would need composition
            # For now, concretize and use interval arithmetic
            lower, upper = node_bounds.concretize()
            interval_bound = IntervalBounds(lower=lower, upper=upper)
            
            lower_contrib = torch.where(
                coeff_l >= 0,
                coeff_l * interval_bound.lower,
                coeff_l * interval_bound.upper
            )
            upper_contrib = torch.where(
                coeff_u >= 0,
                coeff_u * interval_bound.upper,
                coeff_u * interval_bound.lower
            )
            return IntervalBounds(lower=lower_contrib, upper=upper_contrib)
    
    def _compute_backward_linear(
        self,
        node: Node,
        node_bounds: AbstractBounds,
        input_bounds: List[AbstractBounds],
        input_idx: int,
        region: AbstractRegion,
    ) -> AbstractBounds:
        """Exact backward propagation for linear operations."""
        op_type = node.op_type
        
        # For linear operations, backward propagation is straightforward
        if op_type == OperationType.ADD:
            # For ADD: y = x1 + x2, so dy/dx1 = 1, dy/dx2 = 1
            # Backward: dx = dy (identity)
            return node_bounds
        
        elif op_type == OperationType.SUB:
            # For SUB: y = x1 - x2
            # dy/dx1 = 1, dy/dx2 = -1
            if input_idx == 0:
                return node_bounds
            else:
                # Negate for second input
                if isinstance(node_bounds, IntervalBounds):
                    return IntervalBounds(
                        lower=-node_bounds.upper,
                        upper=-node_bounds.lower,
                    )
                else:
                    raise NotImplementedError("Negation of LinearBounds not yet implemented")
        
        elif op_type == OperationType.MATMUL:
            # For MATMUL: y = x @ W
            # dy/dx = dy @ W^T, dy/dW = x^T @ dy
            # We need to know which input we're computing for
            
            # Get the other input (weight matrix)
            if input_idx == 0:
                # Computing gradient w.r.t. x
                # Need W to compute dy @ W^T
                weight_idx = 1
            else:
                # Computing gradient w.r.t. W - not typically used
                raise NotImplementedError("Backward through weight matrix not yet implemented")
            
            weight_node = node.inputs[weight_idx]
            if weight_node.node_type == NodeType.CONSTANT:
                W = weight_node.attributes.get("value")
                if W is None:
                    raise ValueError("Weight node missing value attribute")
                
                # Compute dy @ W^T
                if isinstance(node_bounds, IntervalBounds):
                    W_T = W.T
                    pos_W_T = torch.clamp(W_T, min=0)
                    neg_W_T = torch.clamp(W_T, max=0)
                    
                    lower = node_bounds.lower @ pos_W_T + node_bounds.upper @ neg_W_T
                    upper = node_bounds.upper @ pos_W_T + node_bounds.lower @ neg_W_T
                    
                    return IntervalBounds(lower=lower, upper=upper)
                else:
                    raise NotImplementedError("Backward MATMUL with LinearBounds not yet implemented")
            else:
                raise NotImplementedError("Backward MATMUL with non-constant weight not yet implemented")
        
        elif op_type == OperationType.LINEAR:
            # Similar to MATMUL but need to handle bias
            if input_idx > 0:
                raise NotImplementedError("Backward through LINEAR weight/bias not yet implemented")
            
            W = node.attributes.get("weight")
            if W is None:
                raise ValueError("LINEAR operation missing weight attribute")
            
            # Compute dy @ W^T
            if isinstance(node_bounds, IntervalBounds):
                W_T = W.T
                pos_W_T = torch.clamp(W_T, min=0)
                neg_W_T = torch.clamp(W_T, max=0)
                
                lower = node_bounds.lower @ pos_W_T + node_bounds.upper @ neg_W_T
                upper = node_bounds.upper @ pos_W_T + node_bounds.lower @ neg_W_T
                
                return IntervalBounds(lower=lower, upper=upper)
            else:
                raise NotImplementedError("Backward LINEAR with LinearBounds not yet implemented")
        
        else:
            raise NotImplementedError(f"Backward propagation for {op_type} not yet implemented")
    
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
