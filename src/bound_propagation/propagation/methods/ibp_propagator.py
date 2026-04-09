"""
Interval Bound Propagation (IBP) method.

Simple forward propagation using only interval bounds (no linear relaxations).
Faster but less precise than LBP methods.
"""

from typing import Dict, List

import torch

from bound_propagation.bounds.abstract_bounds import AbstractBounds
from bound_propagation.bounds.interval_bounds import IntervalBounds
from bound_propagation.ir.graph import Graph
from bound_propagation.ir.node import Node, NodeType
from bound_propagation.ir.operations import OperationType
from bound_propagation.propagation.methods.base import MethodPropagator
from bound_propagation.regions.abstract import AbstractRegion
from bound_propagation.regions.multi_input import MultiInputRegion


class IBPPropagator(MethodPropagator):
    """
    Interval Bound Propagation (IBP).
    
    Propagates simple interval bounds forward through the computation graph.
    Uses interval arithmetic rules for all operations. Faster than LBP but
    less precise because it doesn't track linear dependencies.
    
    For non-linear operations, uses sound interval extensions:
    - ReLU: [a,b] → [max(0,a), max(0,b)]
    - Sigmoid: [a,b] → [sigmoid(a), sigmoid(b)]
    - etc.
    """
    
    @property
    def method_name(self) -> str:
        """Return the name of this propagation method."""
        return "ibp"
    
    def propagate(
        self,
        graph: Graph,
        region: AbstractRegion,
    ) -> Dict[int, AbstractBounds]:
        """
        Propagate interval bounds forward through the graph.
        
        Args:
            graph: The computation graph to propagate through.
            region: Input region defining the domain.
        
        Returns:
            Dictionary mapping node IDs to their computed interval bounds.
        """
        # Get nodes in topological order
        nodes = graph.topological_order()
        
        # Initialize bounds dictionary
        bounds: Dict[int, IntervalBounds] = {}
        
        # Propagate through each node
        for node in nodes:
            # Try to get from cache first
            if self.cache is not None:
                cached_bounds = self.cache.get(node.id, self.method_name, region)
                if cached_bounds is not None:
                    bounds[node.id] = cached_bounds
                    continue
            
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
    
    def _create_constant_bounds(self, node: Node) -> IntervalBounds:
        """Create point bounds for a constant node."""
        value = node.attributes.get("value")
        if value is None:
            shape = node.output_metadata.shape
            value = torch.zeros(shape)
        
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        
        return IntervalBounds(lower=value.clone(), upper=value.clone())
    
    def _compute_operation_bounds(
        self,
        node: Node,
        input_bounds: List[IntervalBounds],
    ) -> IntervalBounds:
        """Compute interval bounds for an operation."""
        op_type = node.op_type
        
        # Linear operations
        if op_type == OperationType.ADD:
            return self._propagate_add(input_bounds)
        elif op_type == OperationType.SUB:
            return self._propagate_sub(input_bounds)
        elif op_type == OperationType.MATMUL:
            return self._propagate_matmul(node, input_bounds)
        elif op_type == OperationType.LINEAR:
            return self._propagate_linear(node, input_bounds)
        
        # Element-wise non-linear operations
        elif op_type == OperationType.RELU:
            return self._propagate_relu(input_bounds[0])
        elif op_type == OperationType.SIGMOID:
            return self._propagate_sigmoid(input_bounds[0])
        elif op_type == OperationType.TANH:
            return self._propagate_tanh(input_bounds[0])
        elif op_type == OperationType.EXP:
            return self._propagate_exp(input_bounds[0])
        elif op_type == OperationType.LOG:
            return self._propagate_log(input_bounds[0])
        
        # Bilinear operations
        elif op_type == OperationType.MUL:
            return self._propagate_mul(input_bounds)
        elif op_type == OperationType.DIV:
            return self._propagate_div(input_bounds)
        
        else:
            raise NotImplementedError(f"IBP not implemented for {op_type}")
    
    # Linear operations (exact)
    
    def _propagate_add(self, input_bounds: List[IntervalBounds]) -> IntervalBounds:
        """ADD: [a,b] + [c,d] = [a+c, b+d]."""
        if len(input_bounds) != 2:
            raise ValueError(f"ADD expects 2 inputs, got {len(input_bounds)}")
        
        lower = input_bounds[0].lower + input_bounds[1].lower
        upper = input_bounds[0].upper + input_bounds[1].upper
        return IntervalBounds(lower=lower, upper=upper)
    
    def _propagate_sub(self, input_bounds: List[IntervalBounds]) -> IntervalBounds:
        """SUB: [a,b] - [c,d] = [a-d, b-c]."""
        if len(input_bounds) != 2:
            raise ValueError(f"SUB expects 2 inputs, got {len(input_bounds)}")
        
        lower = input_bounds[0].lower - input_bounds[1].upper
        upper = input_bounds[0].upper - input_bounds[1].lower
        return IntervalBounds(lower=lower, upper=upper)
    
    def _propagate_matmul(
        self, node: Node, input_bounds: List[IntervalBounds]
    ) -> IntervalBounds:
        """MATMUL: y = x @ W."""
        if len(input_bounds) != 2:
            raise ValueError(f"MATMUL expects 2 inputs, got {len(input_bounds)}")
        
        x_bounds = input_bounds[0]
        w_bounds = input_bounds[1]
        
        # Assume W is constant (common case)
        if torch.allclose(w_bounds.lower, w_bounds.upper):
            W = w_bounds.lower
            
            pos_W = torch.clamp(W, min=0)
            neg_W = torch.clamp(W, max=0)
            
            lower = x_bounds.lower @ pos_W + x_bounds.upper @ neg_W
            upper = x_bounds.upper @ pos_W + x_bounds.lower @ neg_W
            
            return IntervalBounds(lower=lower, upper=upper)
        else:
            # Both non-constant - use bilinear bounds
            raise NotImplementedError("MATMUL with two non-constant inputs not yet implemented")
    
    def _propagate_linear(
        self, node: Node, input_bounds: List[IntervalBounds]
    ) -> IntervalBounds:
        """LINEAR: y = x @ W + b."""
        if len(input_bounds) == 1:
            x_bounds = input_bounds[0]
            W = node.attributes.get("weight")
            b = node.attributes.get("bias")
            
            if W is None:
                raise ValueError("LINEAR operation missing 'weight' attribute")
            
            pos_W = torch.clamp(W, min=0)
            neg_W = torch.clamp(W, max=0)
            
            matmul_lower = x_bounds.lower @ pos_W + x_bounds.upper @ neg_W
            matmul_upper = x_bounds.upper @ pos_W + x_bounds.lower @ neg_W
            
            if b is not None:
                lower = matmul_lower + b
                upper = matmul_upper + b
            else:
                lower = matmul_lower
                upper = matmul_upper
            
            return IntervalBounds(lower=lower, upper=upper)
        else:
            raise NotImplementedError("LINEAR with multiple inputs not yet implemented")
    
    # Element-wise non-linear operations
    
    def _propagate_relu(self, input_bounds: IntervalBounds) -> IntervalBounds:
        """ReLU: [a,b] → [max(0,a), max(0,b)]."""
        lower = torch.clamp(input_bounds.lower, min=0)
        upper = torch.clamp(input_bounds.upper, min=0)
        return IntervalBounds(lower=lower, upper=upper)
    
    def _propagate_sigmoid(self, input_bounds: IntervalBounds) -> IntervalBounds:
        """Sigmoid: [a,b] → [sigmoid(a), sigmoid(b)] (monotone increasing)."""
        lower = torch.sigmoid(input_bounds.lower)
        upper = torch.sigmoid(input_bounds.upper)
        return IntervalBounds(lower=lower, upper=upper)
    
    def _propagate_tanh(self, input_bounds: IntervalBounds) -> IntervalBounds:
        """Tanh: [a,b] → [tanh(a), tanh(b)] (monotone increasing)."""
        lower = torch.tanh(input_bounds.lower)
        upper = torch.tanh(input_bounds.upper)
        return IntervalBounds(lower=lower, upper=upper)
    
    def _propagate_exp(self, input_bounds: IntervalBounds) -> IntervalBounds:
        """Exp: [a,b] → [exp(a), exp(b)] (monotone increasing)."""
        lower = torch.exp(input_bounds.lower)
        upper = torch.exp(input_bounds.upper)
        return IntervalBounds(lower=lower, upper=upper)
    
    def _propagate_log(self, input_bounds: IntervalBounds) -> IntervalBounds:
        """Log: [a,b] → [log(a), log(b)] (monotone increasing, requires a > 0)."""
        # Clamp to avoid log(0) or log(negative)
        lower = torch.log(torch.clamp(input_bounds.lower, min=1e-8))
        upper = torch.log(torch.clamp(input_bounds.upper, min=1e-8))
        return IntervalBounds(lower=lower, upper=upper)
    
    # Bilinear operations
    
    def _propagate_mul(self, input_bounds: List[IntervalBounds]) -> IntervalBounds:
        """MUL: [a,b] * [c,d] using 4-corner method."""
        if len(input_bounds) != 2:
            raise ValueError(f"MUL expects 2 inputs, got {len(input_bounds)}")
        
        x = input_bounds[0]
        y = input_bounds[1]
        
        # Compute all 4 corners
        corner1 = x.lower * y.lower
        corner2 = x.lower * y.upper
        corner3 = x.upper * y.lower
        corner4 = x.upper * y.upper
        
        # Stack and compute min/max
        corners = torch.stack([corner1, corner2, corner3, corner4])
        lower = torch.min(corners, dim=0)[0]
        upper = torch.max(corners, dim=0)[0]
        
        return IntervalBounds(lower=lower, upper=upper)
    
    def _propagate_div(self, input_bounds: List[IntervalBounds]) -> IntervalBounds:
        """DIV: [a,b] / [c,d] using 4-corner method (requires c,d > 0 or < 0)."""
        if len(input_bounds) != 2:
            raise ValueError(f"DIV expects 2 inputs, got {len(input_bounds)}")
        
        x = input_bounds[0]
        y = input_bounds[1]
        
        # Check for division by zero
        if torch.any(y.lower <= 0) and torch.any(y.upper >= 0):
            raise ValueError("Division by interval containing zero")
        
        # Compute all 4 corners
        corner1 = x.lower / y.lower
        corner2 = x.lower / y.upper
        corner3 = x.upper / y.lower
        corner4 = x.upper / y.upper
        
        # Stack and compute min/max
        corners = torch.stack([corner1, corner2, corner3, corner4])
        lower = torch.min(corners, dim=0)[0]
        upper = torch.max(corners, dim=0)[0]
        
        return IntervalBounds(lower=lower, upper=upper)
