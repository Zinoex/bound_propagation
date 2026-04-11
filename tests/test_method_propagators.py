"""
Tests for method propagators.
"""

# Import relaxations to trigger auto-registration
import bound_propagation.relaxations  # noqa: F401
import pytest
import torch

from bound_propagation.ir.graph import Graph
from bound_propagation.ir.metadata import TensorMetadata
from bound_propagation.ir.node import Node, NodeType
from bound_propagation.ir.operations import OperationType
from bound_propagation.propagation.methods import ForwardLBPPropagator
from bound_propagation.regions.hyperrectangle import HyperRectangle


class TestForwardLBPPropagator:
    """Test the ForwardLBP propagator."""
    
    def test_propagator_creation(self):
        """Test creating a ForwardLBP propagator."""
        propagator = ForwardLBPPropagator()
        assert propagator.method_name == "forward_lbp"
    
    def test_input_node_bounds(self):
        """Test bounds creation for input nodes."""
        # Create a simple graph with just an input node
        graph = Graph()
        metadata = TensorMetadata(shape=(2, 3), dtype="float32")
        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_node)
        
        # Create input region
        lower = torch.zeros(2, 3)
        upper = torch.ones(2, 3)
        region = HyperRectangle(lower=lower, upper=upper)
        
        # Propagate
        propagator = ForwardLBPPropagator()
        bounds = propagator.propagate(graph, region)
        
        # Check that bounds were computed for input
        assert input_node.id in bounds
        # Input should have interval bounds from the region
        from bound_propagation.bounds.interval_bounds import IntervalBounds
        assert isinstance(bounds[input_node.id], IntervalBounds)
    
    def test_relu_with_relaxation(self):
        """Test ReLU operation using relaxation."""
        # Create graph: input -> relu -> output
        graph = Graph()
        metadata = TensorMetadata(shape=(3,), dtype="float32")
        
        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_node)
        
        relu_node = Node(
            id=1,
            op_type=OperationType.RELU,
            inputs=[input_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(relu_node)
        
        # Mark as output
        graph.mark_outputs([relu_node])
        
        # Create input region: [-1, 1] so ReLU is in crossing regime
        lower = torch.tensor([-1.0, -1.0, -1.0])
        upper = torch.tensor([1.0, 1.0, 1.0])
        region = HyperRectangle(lower=lower, upper=upper)
        
        # Propagate
        propagator = ForwardLBPPropagator()
        bounds = propagator.propagate(graph, region)
        
        # Check that bounds were computed
        assert input_node.id in bounds
        assert relu_node.id in bounds
        
        # ReLU output should have bounds
        relu_bounds = bounds[relu_node.id]
        assert relu_bounds is not None

    def test_relaxation_strategies_are_reused_during_propagation(self, monkeypatch):
        """Forward LBP should resolve relaxation strategies once in the propagator."""
        graph = Graph()
        metadata = TensorMetadata(shape=(2,), dtype="float32")

        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_node)

        relu_node = Node(
            id=1,
            op_type=OperationType.RELU,
            inputs=[input_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(relu_node)
        graph.mark_outputs([relu_node])

        propagator = ForwardLBPPropagator()
        cached_strategy = propagator._relaxation_strategies[OperationType.RELU]

        monkeypatch.setattr(
            "bound_propagation.relaxations.base.RelaxationRegistry.get",
            lambda op_type: (_ for _ in ()).throw(AssertionError("registry lookup during propagate")),
        )
        monkeypatch.setattr(
            "bound_propagation.relaxations.base.RelaxationRegistry.has_strategy",
            lambda op_type: (_ for _ in ()).throw(AssertionError("registry lookup during propagate")),
        )

        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        bounds = propagator.propagate(graph, region)

        assert bounds[relu_node.id] is not None
        assert propagator._relaxation_strategies[OperationType.RELU] is cached_strategy
    
    def test_constant_node(self):
        """Test bounds for constant nodes."""
        # Create graph with constant
        graph = Graph()
        metadata = TensorMetadata(shape=(2,), dtype="float32")
        
        const_value = torch.tensor([3.14, 2.71])
        const_node = Node(
            id=0,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=metadata,
            attributes={"value": const_value},
            node_type=NodeType.CONSTANT,
        )
        graph.add_node(const_node)
        
        # Create dummy region
        region = HyperRectangle(
            lower=torch.zeros(1),
            upper=torch.ones(1),
        )
        
        # Propagate
        propagator = ForwardLBPPropagator()
        bounds = propagator.propagate(graph, region)
        
        # Check constant bounds
        assert const_node.id in bounds
        from bound_propagation.bounds.interval_bounds import IntervalBounds
        const_bounds = bounds[const_node.id]
        assert isinstance(const_bounds, IntervalBounds)
        
        # Should be point bounds
        assert torch.allclose(const_bounds.lower, const_value)
        assert torch.allclose(const_bounds.upper, const_value)
