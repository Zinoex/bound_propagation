"""
Tests for bound propagation orchestrator.

DEPRECATED: Old strategy architecture replaced with method-based propagators.
See tests/test_method_propagators.py for new tests.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Old strategy architecture deprecated")

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.ir import Graph, Node, NodeType, OperationType, TensorMetadata
from bound_propagation.regions import HyperRectangle

# from bound_propagation.strategy import (
#     BoundPropagator,
#     StrategyRegistry,
#     get_global_registry,
# )


def create_simple_graph() -> Graph:
    """
    Create a simple test graph: x -> relu -> output

    Input: x (shape: [2])
    Operations: relu(x)
    Output: relu(x)
    """
    # Input node
    input_node = Node(
        id=0,
        op_type=OperationType.INPUT,
        inputs=[],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        node_type=NodeType.INPUT,
        name="x",
    )

    # ReLU node
    relu_node = Node(
        id=1,
        op_type=OperationType.RELU,
        inputs=[input_node],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        name="relu",
    )

    graph = Graph(nodes=[input_node, relu_node])
    graph.mark_outputs([relu_node])

    return graph


def create_linear_graph() -> Graph:
    """
    Create a linear layer graph: x -> linear -> output

    Input: x (shape: [2])
    Operations: linear(x) with weight (3, 2)
    Output: (3,)
    """
    # Input node
    input_node = Node(
        id=0,
        op_type=OperationType.INPUT,
        inputs=[],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        node_type=NodeType.INPUT,
        name="x",
    )

    # Linear node
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])
    bias = torch.tensor([0.5, 1.0, 2.0])

    linear_node = Node(
        id=1,
        op_type=OperationType.LINEAR,
        inputs=[input_node],
        output_metadata=TensorMetadata(
            shape=(3,), dtype=torch.float32, device=torch.device("cpu")
        ),
        attributes={"weight": weight, "bias": bias},
        name="linear",
    )

    graph = Graph(nodes=[input_node, linear_node])
    graph.mark_outputs([linear_node])

    return graph


def create_chained_graph() -> Graph:
    """
    Create a chained graph: x -> relu -> add(relu, const) -> output

    Input: x (shape: [2])
    Operations: relu(x), add(relu, 1.0)
    Output: (2,)
    """
    # Input node
    input_node = Node(
        id=0,
        op_type=OperationType.INPUT,
        inputs=[],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        node_type=NodeType.INPUT,
        name="x",
    )

    # ReLU node
    relu_node = Node(
        id=1,
        op_type=OperationType.RELU,
        inputs=[input_node],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        name="relu",
    )

    # Constant node
    const_node = Node(
        id=2,
        op_type=OperationType.CONSTANT,
        inputs=[],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        node_type=NodeType.CONSTANT,
        attributes={"value": torch.tensor([1.0, 1.0])},
        name="const",
    )

    # Add node
    add_node = Node(
        id=3,
        op_type=OperationType.ADD,
        inputs=[relu_node, const_node],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        name="add",
    )

    graph = Graph(nodes=[input_node, relu_node, const_node, add_node])
    graph.mark_outputs([add_node])

    return graph


def create_dag_graph() -> Graph:
    """
    Create a DAG (not a tree): x -> relu(x) + x

    This is a DAG because the input node x is used by both:
    - The ReLU operation
    - The add operation (directly)

    Input: x (shape: [2])
    Operations: relu(x), add(relu(x), x)
    Output: (2,)
    """
    # Input node
    input_node = Node(
        id=0,
        op_type=OperationType.INPUT,
        inputs=[],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        node_type=NodeType.INPUT,
        name="x",
    )

    # ReLU node
    relu_node = Node(
        id=1,
        op_type=OperationType.RELU,
        inputs=[input_node],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        name="relu",
    )

    # Add node: relu(x) + x
    # Note: input_node is used twice!
    add_node = Node(
        id=2,
        op_type=OperationType.ADD,
        inputs=[relu_node, input_node],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        name="add",
    )

    graph = Graph(nodes=[input_node, relu_node, add_node])
    graph.mark_outputs([add_node])

    return graph


def create_multi_use_dag() -> Graph:
    """
    Create a more complex DAG: y = (x + 1) * (x + 1)

    The intermediate result (x + 1) is used twice in the multiply operation.

    Input: x (shape: [2])
    Operations: temp = add(x, 1), mul(temp, temp)
    Output: (2,)
    """
    # Input node
    input_node = Node(
        id=0,
        op_type=OperationType.INPUT,
        inputs=[],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        node_type=NodeType.INPUT,
        name="x",
    )

    # Constant node
    const_node = Node(
        id=1,
        op_type=OperationType.CONSTANT,
        inputs=[],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        node_type=NodeType.CONSTANT,
        attributes={"value": torch.tensor([1.0, 1.0])},
        name="const",
    )

    # Add node: x + 1
    add_node = Node(
        id=2,
        op_type=OperationType.ADD,
        inputs=[input_node, const_node],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        name="add",
    )

    # Multiply node: (x + 1) * (x + 1)
    # Note: add_node is used twice!
    mul_node = Node(
        id=3,
        op_type=OperationType.MUL,
        inputs=[add_node, add_node],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        name="mul",
    )

    graph = Graph(nodes=[input_node, const_node, add_node, mul_node])
    graph.mark_outputs([mul_node])

    return graph


class TestBoundPropagator:
    """Tests for BoundPropagator."""

    def setup_method(self):
        """Setup registry with IBP strategies before each test."""
        # Use the global registry which has IBP strategies auto-registered
        self.registry = get_global_registry()

    def test_create_propagator(self):
        """Test creating a bound propagator."""
        graph = create_simple_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        assert propagator.graph is graph
        assert propagator.method == "ibp"
        assert propagator.registry is self.registry

    def test_simple_propagation(self):
        """Test bound propagation through simple graph."""
        graph = create_simple_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        # Input region: [-1, 1] x [-2, 2]
        input_region = HyperRectangle(
            torch.tensor([-1.0, -2.0]),
            torch.tensor([1.0, 2.0]),
        )

        # Compute bounds
        all_bounds = propagator.compute_bounds(input_region)

        # Check that all nodes have bounds
        assert len(all_bounds) == 2  # Input + ReLU

        # Input bounds should match input region
        input_bounds = all_bounds[0]
        assert torch.allclose(input_bounds.lower, torch.tensor([-1.0, -2.0]))
        assert torch.allclose(input_bounds.upper, torch.tensor([1.0, 2.0]))

        # ReLU bounds: relu([-1, 1], [-2, 2]) = [0, 1], [0, 2]
        relu_bounds = all_bounds[1]
        assert torch.allclose(relu_bounds.lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(relu_bounds.upper, torch.tensor([1.0, 2.0]))

    def test_linear_propagation(self):
        """Test bound propagation through linear layer."""
        graph = create_linear_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        # Input region: [0, 1] x [0, 1]
        input_region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Compute bounds
        propagator.compute_bounds(input_region)

        # Get output bounds
        output_bounds = propagator.get_output_bounds()
        assert len(output_bounds) == 1

        linear_bounds = output_bounds[0]

        # For linear with weight [[1,2], [3,4], [-1,-2]] and bias [0.5, 1.0, 2.0]:
        # Row 0: [1,2] @ [0,1], [0,1] = [0, 3] + 0.5 = [0.5, 3.5]
        # Row 1: [3,4] @ [0,1], [0,1] = [0, 7] + 1.0 = [1.0, 8.0]
        # Row 2: [-1,-2] @ [0,1], [0,1] = [-3, 0] + 2.0 = [-1.0, 2.0]

        assert linear_bounds.shape == (3,)
        assert torch.allclose(linear_bounds.lower, torch.tensor([0.5, 1.0, -1.0]))
        assert torch.allclose(linear_bounds.upper, torch.tensor([3.5, 8.0, 2.0]))

    def test_chained_propagation(self):
        """Test bound propagation through chained operations."""
        graph = create_chained_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        # Input region: [-1, 1] x [-1, 1]
        input_region = HyperRectangle(
            torch.tensor([-1.0, -1.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Compute bounds
        all_bounds = propagator.compute_bounds(input_region)

        # Should have 4 nodes: input, relu, const, add
        assert len(all_bounds) == 4

        # ReLU bounds: relu([-1, 1]) = [0, 1]
        relu_bounds = all_bounds[1]
        assert torch.allclose(relu_bounds.lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(relu_bounds.upper, torch.tensor([1.0, 1.0]))

        # Constant bounds: [1, 1]
        const_bounds = all_bounds[2]
        assert torch.allclose(const_bounds.lower, torch.tensor([1.0, 1.0]))
        assert torch.allclose(const_bounds.upper, torch.tensor([1.0, 1.0]))

        # Add bounds: [0, 1] + [1, 1] = [1, 2]
        add_bounds = all_bounds[3]
        assert torch.allclose(add_bounds.lower, torch.tensor([1.0, 1.0]))
        assert torch.allclose(add_bounds.upper, torch.tensor([2.0, 2.0]))

    def test_get_bounds(self):
        """Test getting cached bounds for a node."""
        graph = create_simple_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        input_region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Before compute, should return None
        assert propagator.get_bounds(0) is None

        # Compute bounds
        propagator.compute_bounds(input_region)

        # After compute, should return bounds
        bounds = propagator.get_bounds(0)
        assert bounds is not None
        assert isinstance(bounds, IntervalBounds)

    def test_get_output_bounds(self):
        """Test getting output bounds."""
        graph = create_simple_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        input_region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Before compute, should raise
        with pytest.raises(RuntimeError, match="not computed"):
            propagator.get_output_bounds()

        # Compute bounds
        propagator.compute_bounds(input_region)

        # After compute, should return output bounds
        output_bounds = propagator.get_output_bounds()
        assert len(output_bounds) == 1
        assert isinstance(output_bounds[0], IntervalBounds)

    def test_clear_cache(self):
        """Test clearing the bounds cache."""
        graph = create_simple_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        input_region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Compute bounds
        propagator.compute_bounds(input_region)
        assert propagator.get_bounds(0) is not None

        # Clear cache
        propagator.clear_cache()
        assert propagator.get_bounds(0) is None

    def test_repr(self):
        """Test string representation."""
        graph = create_simple_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        repr_str = repr(propagator)
        assert "BoundPropagator" in repr_str
        assert "method=ibp" in repr_str
        assert "cached=0/2" in repr_str

    def test_dag_propagation(self):
        """Test bound propagation through a DAG (not a tree): relu(x) + x."""
        graph = create_dag_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        # Input region: [-1, 1] x [-1, 1]
        input_region = HyperRectangle(
            torch.tensor([-1.0, -1.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Compute bounds
        all_bounds = propagator.compute_bounds(input_region)

        # Should have 3 nodes: input, relu, add
        assert len(all_bounds) == 3

        # Input bounds: [-1, 1] x [-1, 1]
        input_bounds = all_bounds[0]
        assert torch.allclose(input_bounds.lower, torch.tensor([-1.0, -1.0]))
        assert torch.allclose(input_bounds.upper, torch.tensor([1.0, 1.0]))

        # ReLU bounds: relu([-1, 1]) = [0, 1]
        relu_bounds = all_bounds[1]
        assert torch.allclose(relu_bounds.lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(relu_bounds.upper, torch.tensor([1.0, 1.0]))

        # Add bounds: relu(x) + x = [0, 1] + [-1, 1] = [-1, 2]
        add_bounds = all_bounds[2]
        assert torch.allclose(add_bounds.lower, torch.tensor([-1.0, -1.0]))
        assert torch.allclose(add_bounds.upper, torch.tensor([2.0, 2.0]))

    def test_multi_use_dag_propagation(self):
        """Test bound propagation when intermediate node is used multiple times: (x + 1) * (x + 1)."""
        graph = create_multi_use_dag()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        # Input region: [0, 2] x [1, 3]
        input_region = HyperRectangle(
            torch.tensor([0.0, 1.0]),
            torch.tensor([2.0, 3.0]),
        )

        # Compute bounds
        all_bounds = propagator.compute_bounds(input_region)

        # Should have 4 nodes: input, const, add, mul
        assert len(all_bounds) == 4

        # Add bounds: x + 1 = [0, 2] + 1 = [1, 3], [1, 3] + 1 = [2, 4]
        add_bounds = all_bounds[2]
        assert torch.allclose(add_bounds.lower, torch.tensor([1.0, 2.0]))
        assert torch.allclose(add_bounds.upper, torch.tensor([3.0, 4.0]))

        # Multiply bounds: (x + 1) * (x + 1)
        # [1, 3] * [1, 3] = [1, 9]
        # [2, 4] * [2, 4] = [4, 16]
        mul_bounds = all_bounds[3]
        assert torch.allclose(mul_bounds.lower, torch.tensor([1.0, 4.0]))
        assert torch.allclose(mul_bounds.upper, torch.tensor([9.0, 16.0]))

    def test_dag_with_negative_input(self):
        """Test DAG propagation with negative inputs: relu(x) + x where x is negative."""
        graph = create_dag_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        # Input region: all negative
        input_region = HyperRectangle(
            torch.tensor([-3.0, -2.0]),
            torch.tensor([-1.0, -1.0]),
        )

        # Compute bounds
        all_bounds = propagator.compute_bounds(input_region)

        # ReLU bounds: relu([-3, -1]) = [0, 0]
        relu_bounds = all_bounds[1]
        assert torch.allclose(relu_bounds.lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(relu_bounds.upper, torch.tensor([0.0, 0.0]))

        # Add bounds: 0 + [-3, -1] = [-3, -1], 0 + [-2, -1] = [-2, -1]
        add_bounds = all_bounds[2]
        assert torch.allclose(add_bounds.lower, torch.tensor([-3.0, -2.0]))
        assert torch.allclose(add_bounds.upper, torch.tensor([-1.0, -1.0]))

    def test_dag_with_positive_input(self):
        """Test DAG propagation with positive inputs: relu(x) + x where x is positive."""
        graph = create_dag_graph()
        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        # Input region: all positive
        input_region = HyperRectangle(
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
        )

        # Compute bounds
        all_bounds = propagator.compute_bounds(input_region)

        # ReLU bounds: relu([1, 3]) = [1, 3] (identity for positive)
        relu_bounds = all_bounds[1]
        assert torch.allclose(relu_bounds.lower, torch.tensor([1.0, 2.0]))
        assert torch.allclose(relu_bounds.upper, torch.tensor([3.0, 4.0]))

        # Add bounds: [1, 3] + [1, 3] = [2, 6], [2, 4] + [2, 4] = [4, 8]
        add_bounds = all_bounds[2]
        assert torch.allclose(add_bounds.lower, torch.tensor([2.0, 4.0]))
        assert torch.allclose(add_bounds.upper, torch.tensor([6.0, 8.0]))


class TestBoundPropagatorError:
    """Tests for error handling in BoundPropagator."""

    def setup_method(self):
        """Setup registry."""
        # Use the global registry which has IBP strategies auto-registered
        self.registry = get_global_registry()

    def test_missing_strategy_raises(self):
        """Test that missing strategy raises error."""
        graph = create_simple_graph()

        # Use empty registry (no strategies)
        empty_registry = StrategyRegistry()
        propagator = BoundPropagator(graph, method="ibp", registry=empty_registry)

        input_region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        with pytest.raises(ValueError, match="No strategy registered"):
            propagator.compute_bounds(input_region)

    def test_missing_constant_value_raises(self):
        """Test that constant node without value raises error."""
        # Create graph with constant missing value
        input_node = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=TensorMetadata(
                shape=(2,), dtype=torch.float32, device=torch.device("cpu")
            ),
            node_type=NodeType.INPUT,
            name="x",
        )

        # Constant without value
        const_node = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=TensorMetadata(
                shape=(2,), dtype=torch.float32, device=torch.device("cpu")
            ),
            node_type=NodeType.CONSTANT,
            attributes={},  # Missing 'value'
            name="const",
        )

        graph = Graph(nodes=[input_node, const_node])
        graph.mark_outputs([const_node])

        propagator = BoundPropagator(graph, method="ibp", registry=self.registry)

        input_region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        with pytest.raises(ValueError, match="missing 'value' attribute"):
            propagator.compute_bounds(input_region)
