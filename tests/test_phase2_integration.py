"""
Phase 2 Integration Tests: Bounding Strategy Decoupling

DEPRECATED: This test file tests the old architecture that has been replaced
with the new method-based propagators in Phase 3.

Tests the complete Phase 2 system end-to-end:
- Bounds representation (IntervalBounds, LinearBounds, regions)
- Strategy framework (BoundingStrategy, registry, config)
- IBP strategies for all operations
- Bound propagation orchestrator

These tests verify that we can compute verified bounds for complete networks.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Old Phase 2 architecture deprecated - replaced with method-based propagators")

import torch

# Old imports - no longer available
# import bound_propagation.strategy.ibp  # noqa: F401
from bound_propagation.bounds import IntervalBounds
from bound_propagation.ir import Graph, Node, NodeType, OperationType, TensorMetadata
from bound_propagation.regions import HyperRectangle

# from bound_propagation.strategy import (
#     BoundPropagator,
#     get_global_registry,
# )


class TestPhase2Integration:
    """Integration tests for Phase 2 functionality."""

    def setup_method(self):
        """Setup registry with IBP strategies."""
        # Use the global registry which has IBP strategies auto-registered
        self.registry = get_global_registry()

    def test_simple_linear_network(self):
        """Test bound propagation through a simple linear network."""
        # Create graph: x -> linear -> relu
        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=TensorMetadata(shape=(3,), dtype=torch.float32, device=torch.device("cpu")),
            node_type=NodeType.INPUT,
            name="x",
        )

        weight = torch.tensor([[1.0, -1.0, 0.5], [2.0, 0.0, -2.0]])
        bias = torch.tensor([0.5, -0.5])

        linear_node = Node(
            id=1,
            op_type=OperationType.LINEAR,
            inputs=[input_node],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            attributes={"weight": weight, "bias": bias},
            name="linear",
        )

        relu_node = Node(
            id=2,
            op_type=OperationType.RELU,
            inputs=[linear_node],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            name="relu",
        )

        graph = Graph(nodes=[input_node, linear_node, relu_node])
        graph.mark_outputs([relu_node])

        # Create bound propagator
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        # Input region: [-1, 1] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )

        # Compute bounds
        all_bounds = propagator.compute_bounds(input_region)

        # Verify input bounds
        input_bounds = all_bounds[0]
        assert torch.allclose(input_bounds.lower, input_region.lower)
        assert torch.allclose(input_bounds.upper, input_region.upper)

        # Verify linear bounds
        linear_bounds = all_bounds[1]
        assert linear_bounds.shape == (2,)
        # Row 0: [1, -1, 0.5] @ [-1,1], [-1,1], [-1,1] = [-2.5, 2.5] + 0.5 = [-2.0, 3.0]
        # Row 1: [2, 0, -2] @ [-1,1], [-1,1], [-1,1] = [-4, 4] - 0.5 = [-4.5, 3.5]
        assert linear_bounds.lower[0] >= -2.1 and linear_bounds.lower[0] <= -1.9
        assert linear_bounds.upper[0] >= 2.9 and linear_bounds.upper[0] <= 3.1

        # Verify ReLU bounds
        relu_bounds = all_bounds[2]
        # ReLU should clamp negative lower bounds to 0
        assert torch.all(relu_bounds.lower >= 0.0)
        assert torch.allclose(relu_bounds.upper, torch.clamp(linear_bounds.upper, min=0.0))

    def test_network_with_multiple_operations(self):
        """Test bound propagation through network with multiple operation types."""
        # Create graph: x -> sigmoid -> mul(sigmoid, 2.0) -> add(mul, 1.0)
        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            node_type=NodeType.INPUT,
            name="x",
        )

        sigmoid_node = Node(
            id=1,
            op_type=OperationType.SIGMOID,
            inputs=[input_node],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            name="sigmoid",
        )

        const_2 = Node(
            id=2,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            node_type=NodeType.CONSTANT,
            attributes={"value": torch.tensor([2.0, 2.0])},
            name="const_2",
        )

        mul_node = Node(
            id=3,
            op_type=OperationType.MUL,
            inputs=[sigmoid_node, const_2],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            name="mul",
        )

        const_1 = Node(
            id=4,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            node_type=NodeType.CONSTANT,
            attributes={"value": torch.tensor([1.0, 1.0])},
            name="const_1",
        )

        add_node = Node(
            id=5,
            op_type=OperationType.ADD,
            inputs=[mul_node, const_1],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            name="add",
        )

        graph = Graph(nodes=[input_node, sigmoid_node, const_2, mul_node, const_1, add_node])
        graph.mark_outputs([add_node])

        # Create propagator
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        # Input region
        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0]),
            upper=torch.tensor([2.0, 2.0]),
        )

        # Compute bounds
        all_bounds = propagator.compute_bounds(input_region)

        # Sigmoid output is in [0, 1]
        sigmoid_bounds = all_bounds[1]
        assert torch.all(sigmoid_bounds.lower >= 0.0)
        assert torch.all(sigmoid_bounds.upper <= 1.0)

        # Mul by 2.0: [0, 2]
        mul_bounds = all_bounds[3]
        assert torch.all(mul_bounds.lower >= 0.0)
        assert torch.all(mul_bounds.upper <= 2.0)

        # Add 1.0: [1, 3]
        add_bounds = all_bounds[5]
        assert torch.all(add_bounds.lower >= 1.0)
        assert torch.all(add_bounds.upper <= 3.0)

    def test_get_output_bounds(self):
        """Test getting bounds for output nodes."""
        # Simple graph
        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            node_type=NodeType.INPUT,
            name="x",
        )

        tanh_node = Node(
            id=1,
            op_type=OperationType.TANH,
            inputs=[input_node],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            name="tanh",
        )

        graph = Graph(nodes=[input_node, tanh_node])
        graph.mark_outputs([tanh_node])

        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        propagator.compute_bounds(input_region)

        # Get output bounds
        output_bounds = propagator.get_output_bounds()

        assert len(output_bounds) == 1
        assert isinstance(output_bounds[0], IntervalBounds)

        # Tanh is bounded by [-1, 1] but input is [0, 1] so output should be [tanh(0), tanh(1)]
        expected_lower = torch.tanh(torch.tensor([0.0, 0.0]))
        expected_upper = torch.tanh(torch.tensor([1.0, 1.0]))

        assert torch.allclose(output_bounds[0].lower, expected_lower)
        assert torch.allclose(output_bounds[0].upper, expected_upper)

    def test_multiple_inputs_network(self):
        """Test network with operations that take multiple inputs."""
        # Create graph: x1 -> relu, x2 -> sigmoid, add(relu, sigmoid)
        input1 = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            node_type=NodeType.INPUT,
            name="x1",
        )

        input2 = Node(
            id=1,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            node_type=NodeType.INPUT,
            name="x2",
        )

        relu_node = Node(
            id=2,
            op_type=OperationType.RELU,
            inputs=[input1],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            name="relu",
        )

        sigmoid_node = Node(
            id=3,
            op_type=OperationType.SIGMOID,
            inputs=[input2],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            name="sigmoid",
        )

        add_node = Node(
            id=4,
            op_type=OperationType.ADD,
            inputs=[relu_node, sigmoid_node],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            name="add",
        )

        graph = Graph(nodes=[input1, input2, relu_node, sigmoid_node, add_node])
        graph.mark_outputs([add_node])

        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        # For multi-input graphs, we need a region that covers all inputs
        # For now, we'll use the same region for both inputs
        # In practice, you'd extend this to handle multiple input regions
        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        all_bounds = propagator.compute_bounds(input_region)

        # ReLU bounds: relu([-1, 1]) = [0, 1]
        relu_bounds = all_bounds[2]
        assert torch.allclose(relu_bounds.lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(relu_bounds.upper, torch.tensor([1.0, 1.0]))

        # Sigmoid bounds: sigmoid([-1, 1]) = [sigmoid(-1), sigmoid(1)]
        sigmoid_bounds = all_bounds[3]
        expected_lower = torch.sigmoid(torch.tensor([-1.0, -1.0]))
        expected_upper = torch.sigmoid(torch.tensor([1.0, 1.0]))
        assert torch.allclose(sigmoid_bounds.lower, expected_lower)
        assert torch.allclose(sigmoid_bounds.upper, expected_upper)

        # Add bounds: [0, 1] + [sigmoid(-1), sigmoid(1)]
        add_bounds = all_bounds[4]
        expected_add_lower = relu_bounds.lower + sigmoid_bounds.lower
        expected_add_upper = relu_bounds.upper + sigmoid_bounds.upper
        assert torch.allclose(add_bounds.lower, expected_add_lower)
        assert torch.allclose(add_bounds.upper, expected_add_upper)


class TestPhase2Summary:
    """Summary test verifying all Phase 2 components work together."""

    def test_phase2_complete(self):
        """Verify all Phase 2 components are functional."""
        # 1. Bounds representation
        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        bounds = IntervalBounds(region, torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        assert bounds.shape == (2,)

        # 2. Strategy framework
        registry = get_global_registry()
        assert registry.has_strategy(OperationType.ADD, "ibp")
        assert registry.has_strategy(OperationType.RELU, "ibp")

        # 3. IBP strategies - verify all are registered
        add_strategy = registry.get(OperationType.ADD, "ibp")
        assert add_strategy.method_name == "ibp"

        # 4. Bound propagator
        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
            node_type=NodeType.INPUT,
        )

        relu_node = Node(
            id=1,
            op_type=OperationType.RELU,
            inputs=[input_node],
            output_metadata=TensorMetadata(shape=(2,), dtype=torch.float32, device=torch.device("cpu")),
        )

        graph = Graph(nodes=[input_node, relu_node])
        graph.mark_outputs([relu_node])

        propagator = BoundPropagator(graph, method="ibp", registry=registry)
        all_bounds = propagator.compute_bounds(region)

        assert len(all_bounds) == 2
        output_bounds = propagator.get_output_bounds()
        assert len(output_bounds) == 1
