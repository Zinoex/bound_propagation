"""
Tests for BackwardLBP and IBP propagators.
"""

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.ir import Graph, Node, NodeType, OperationType, TensorMetadata
from bound_propagation.propagation import (
    IBPPropagator,
)
from bound_propagation.regions import HyperRectangle


class TestIBPPropagator:
    """Test the IBP propagator."""

    def test_propagator_creation(self):
        """Test creating an IBP propagator."""
        propagator = IBPPropagator()
        assert propagator.method_name == "ibp"

    def test_operation_strategies_are_reused(self):
        """IBP should create operation strategies once and reuse them."""
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

        const_node = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=metadata,
            attributes={"value": torch.tensor([1.0, 2.0])},
            node_type=NodeType.CONSTANT,
        )
        graph.add_node(const_node)

        add_node = Node(
            id=2,
            op_type=OperationType.ADD,
            inputs=[input_node, const_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(add_node)
        graph.mark_outputs([add_node])

        propagator = IBPPropagator()
        dispatch_key = (
            OperationType.ADD,
            (InputBoundKind.ABSTRACT, InputBoundKind.CONSTANT),
        )
        cached_strategy = propagator._operation_strategies[dispatch_key]

        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        first_bounds = propagator.propagate(graph, region)
        second_bounds = propagator.propagate(graph, region)

        assert torch.allclose(first_bounds[add_node.id].lower, second_bounds[add_node.id].lower)
        assert torch.allclose(first_bounds[add_node.id].upper, second_bounds[add_node.id].upper)
        assert propagator._operation_strategies[dispatch_key] is cached_strategy

    def test_input_node_bounds(self):
        """Test bounds creation for input nodes."""
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

        lower = torch.tensor([0.0, 0.0, 0.0])
        upper = torch.tensor([1.0, 1.0, 1.0])
        region = HyperRectangle(lower=lower, upper=upper)

        propagator = IBPPropagator()
        bounds = propagator.propagate(graph, region)

        assert input_node.id in bounds
        assert isinstance(bounds[input_node.id], IntervalBounds)
        assert torch.allclose(bounds[input_node.id].lower, lower)
        assert torch.allclose(bounds[input_node.id].upper, upper)

    def test_relu_operation(self):
        """Test ReLU with IBP: [a,b] → [max(0,a), max(0,b)]."""
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
        graph.mark_outputs([relu_node])

        # Input in [-1, 2]
        lower = torch.tensor([-1.0, -1.0, -1.0])
        upper = torch.tensor([2.0, 2.0, 2.0])
        region = HyperRectangle(lower=lower, upper=upper)

        propagator = IBPPropagator()
        bounds = propagator.propagate(graph, region)

        # Expected: [max(0,-1), max(0,2)] = [0, 2]
        expected_lower = torch.tensor([0.0, 0.0, 0.0])
        expected_upper = torch.tensor([2.0, 2.0, 2.0])

        assert torch.allclose(bounds[relu_node.id].lower, expected_lower)
        assert torch.allclose(bounds[relu_node.id].upper, expected_upper)

    def test_add_operation(self):
        """Test ADD with IBP."""
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

        const_value = torch.tensor([2.0, 3.0])
        const_node = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=metadata,
            attributes={"value": const_value},
            node_type=NodeType.CONSTANT,
        )
        graph.add_node(const_node)

        add_node = Node(
            id=2,
            op_type=OperationType.ADD,
            inputs=[input_node, const_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(add_node)
        graph.mark_outputs([add_node])

        # Input in [0, 1]
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 1.0])
        region = HyperRectangle(lower=lower, upper=upper)

        propagator = IBPPropagator()
        bounds = propagator.propagate(graph, region)

        # Expected: [0,1] + [2,3] = [2,4]
        expected_lower = torch.tensor([2.0, 3.0])
        expected_upper = torch.tensor([3.0, 4.0])

        assert torch.allclose(bounds[add_node.id].lower, expected_lower)
        assert torch.allclose(bounds[add_node.id].upper, expected_upper)

    def test_sigmoid_operation(self):
        """Test Sigmoid with IBP (monotone increasing)."""
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

        sigmoid_node = Node(
            id=1,
            op_type=OperationType.SIGMOID,
            inputs=[input_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(sigmoid_node)
        graph.mark_outputs([sigmoid_node])

        # Input in [-1, 1]
        lower = torch.tensor([-1.0, -1.0])
        upper = torch.tensor([1.0, 1.0])
        region = HyperRectangle(lower=lower, upper=upper)

        propagator = IBPPropagator()
        bounds = propagator.propagate(graph, region)

        # Expected: [sigmoid(-1), sigmoid(1)]
        expected_lower = torch.sigmoid(torch.tensor([-1.0, -1.0]))
        expected_upper = torch.sigmoid(torch.tensor([1.0, 1.0]))

        assert torch.allclose(bounds[sigmoid_node.id].lower, expected_lower)
        assert torch.allclose(bounds[sigmoid_node.id].upper, expected_upper)

    def test_matmul_with_constant_weight(self):
        """Test MATMUL with IBP."""
        graph = Graph()
        input_metadata = TensorMetadata(shape=(2,), dtype="float32")
        output_metadata = TensorMetadata(shape=(3,), dtype="float32")

        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=input_metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input_node)

        W = torch.tensor([[1.0, -0.5, 2.0], [0.5, 1.0, -1.0]])
        weight_node = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=TensorMetadata(shape=(2, 3), dtype="float32"),
            attributes={"value": W},
            node_type=NodeType.CONSTANT,
        )
        graph.add_node(weight_node)

        matmul_node = Node(
            id=2,
            op_type=OperationType.MATMUL,
            inputs=[input_node, weight_node],
            output_metadata=output_metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(matmul_node)
        graph.mark_outputs([matmul_node])

        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 1.0])
        region = HyperRectangle(lower=lower, upper=upper)

        propagator = IBPPropagator()
        bounds = propagator.propagate(graph, region)

        # Same expected bounds as ForwardLBP for linear ops
        expected_lower = torch.tensor([0.0, -0.5, -1.0])
        expected_upper = torch.tensor([1.5, 1.0, 2.0])

        assert torch.allclose(bounds[matmul_node.id].lower, expected_lower, atol=1e-6)
        assert torch.allclose(bounds[matmul_node.id].upper, expected_upper, atol=1e-6)
