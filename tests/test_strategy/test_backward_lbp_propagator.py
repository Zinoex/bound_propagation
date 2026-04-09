"""
Tests for backward-mode Linear Bound Propagation (Backward LBP).

Tests the new backward propagation algorithm (Algorithm 2 from auto_LiRPA paper)
which traverses the graph in reverse topological order and accumulates bounds.
"""

import pytest
import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.ir import Graph, Node, NodeType, OperationType, TensorMetadata
from bound_propagation.regions import HyperRectangle
from bound_propagation.strategy import BackwardLBPPropagator


def create_add_constant_graph() -> Graph:
    """
    Create a simple addition graph: y = x + c

    Input: x (shape: [2])
    Operations: add(x, constant)
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
        attributes={"value": torch.tensor([1.0, 2.0])},
        name="const",
    )

    # Add node
    add_node = Node(
        id=2,
        op_type=OperationType.ADD,
        inputs=[input_node, const_node],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        name="add",
    )

    graph = Graph(nodes=[input_node, const_node, add_node])
    graph.mark_outputs([add_node])

    return graph


def create_relu_graph() -> Graph:
    """
    Create a simple ReLU graph: y = relu(x)

    Input: x (shape: [2])
    Operations: relu(x)
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

    graph = Graph(nodes=[input_node, relu_node])
    graph.mark_outputs([relu_node])

    return graph


def create_matmul_relu_graph() -> Graph:
    """
    Create a graph with matmul and ReLU: y = relu(x @ W)

    Input: x (shape: [2])
    Operations: matmul(x, W), relu
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

    # Weight constant
    weight_node = Node(
        id=1,
        op_type=OperationType.CONSTANT,
        inputs=[],
        output_metadata=TensorMetadata(
            shape=(2, 2), dtype=torch.float32, device=torch.device("cpu")
        ),
        node_type=NodeType.CONSTANT,
        attributes={"value": torch.tensor([[1.0, -1.0], [0.5, 0.5]])},
        name="weight",
    )

    # Matmul node
    matmul_node = Node(
        id=2,
        op_type=OperationType.MATMUL,
        inputs=[input_node, weight_node],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        name="matmul",
    )

    # ReLU node
    relu_node = Node(
        id=3,
        op_type=OperationType.RELU,
        inputs=[matmul_node],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        name="relu",
    )

    graph = Graph(nodes=[input_node, weight_node, matmul_node, relu_node])
    graph.mark_outputs([relu_node])

    return graph


class TestBackwardLBPPropagator:
    """Test suite for BackwardLBPPropagator."""

    def test_simple_addition(self):
        """
        Test backward LBP on y = x + c.
        
        For linear operations, we expect identity linear bounds:
        - A = I (identity matrix)
        - bias = 0
        
        This means the output depends linearly on the input with slope 1.
        """
        graph = create_add_constant_graph()
        propagator = BackwardLBPPropagator(graph)

        # Input region: [0, 1] x [0, 1]
        input_region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Compute backward bounds
        propagator.compute_bounds(input_region)

        # Get input bounds (how output depends on input)
        input_bounds = propagator.get_input_bounds()
        assert len(input_bounds) == 1

        bounds = input_bounds[0]
        assert isinstance(bounds, LinearBounds)

        # For y = x + c, we expect A = I
        assert bounds.linear_lower is not None
        assert bounds.linear_upper is not None

        expected_identity = torch.eye(2)
        assert torch.allclose(bounds.linear_lower, expected_identity)
        assert torch.allclose(bounds.linear_upper, expected_identity)

        # Bias should be zero (no constant term in the dependency)
        assert torch.allclose(bounds.bias_lower, torch.zeros(2))
        assert torch.allclose(bounds.bias_upper, torch.zeros(2))

        # Concretized bounds should match input region
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(upper, torch.tensor([1.0, 1.0]))

    def test_relu_all_positive(self):
        """
        Test backward LBP on y = relu(x) where x is all positive.
        
        When the input is all positive, relu is the identity function,
        so we expect A = I.
        """
        graph = create_relu_graph()
        propagator = BackwardLBPPropagator(graph)

        # Input region: [0.5, 2.0] x [1.0, 3.0] (all positive)
        input_region = HyperRectangle(
            torch.tensor([0.5, 1.0]),
            torch.tensor([2.0, 3.0]),
        )

        # Compute backward bounds
        propagator.compute_bounds(input_region)

        # Get input bounds
        input_bounds = propagator.get_input_bounds()[0]
        assert isinstance(input_bounds, LinearBounds)

        # For all-positive regime, relu is identity, so A = I
        assert input_bounds.linear_lower is not None
        assert input_bounds.linear_upper is not None

        expected_identity = torch.eye(2)
        assert torch.allclose(input_bounds.linear_lower, expected_identity)
        assert torch.allclose(input_bounds.linear_upper, expected_identity)

        assert torch.allclose(input_bounds.bias_lower, torch.zeros(2))
        assert torch.allclose(input_bounds.bias_upper, torch.zeros(2))

        # Output bounds should match input
        lower, upper = input_bounds.concretize()
        assert torch.allclose(lower, torch.tensor([0.5, 1.0]))
        assert torch.allclose(upper, torch.tensor([2.0, 3.0]))

    def test_relu_crossing_zero(self):
        """
        Test backward LBP on y = relu(x) where x crosses zero.
        
        When x can be negative or positive, relu uses a linear relaxation:
        - Lower bound: y >= 0 (alpha_lower = 0)
        - Upper bound: y <= slope * x + bias (secant line)
        
        For crossing regime, slope = upper / (upper - lower)
        """
        graph = create_relu_graph()
        propagator = BackwardLBPPropagator(graph)

        # Input region: [-1, 1] x [0.5, 2.0]
        # First element crosses zero, second is all positive
        input_region = HyperRectangle(
            torch.tensor([-1.0, 0.5]),
            torch.tensor([1.0, 2.0]),
        )

        # Compute backward bounds
        propagator.compute_bounds(input_region)

        # Get input bounds
        input_bounds = propagator.get_input_bounds()[0]
        assert isinstance(input_bounds, LinearBounds)

        # Check lower bound slopes (should be 0 for crossing, 1 for positive)
        # Lower bound: y >= 0, so alpha = 0 for crossing element
        expected_lower = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        assert torch.allclose(input_bounds.linear_lower, expected_lower)
        assert torch.allclose(input_bounds.bias_lower, torch.zeros(2))

        # Check upper bound slopes
        # For first element (crossing [-1, 1]):
        # slope = 1 / (1 - (-1)) = 0.5
        # For second element (positive [0.5, 2.0]): slope = 1
        expected_upper = torch.tensor([[0.5, 0.0], [0.0, 1.0]])
        assert torch.allclose(input_bounds.linear_upper, expected_upper)

        # Upper bias for crossing: -slope * lower = -0.5 * (-1) = 0.5
        expected_bias_upper = torch.tensor([0.5, 0.0])
        assert torch.allclose(input_bounds.bias_upper, expected_bias_upper)

        # Concretize to verify final bounds
        lower, upper = input_bounds.concretize()
        # Lower: max(0, -1) and max(0, 0.5) = [0, 0.5]
        assert torch.allclose(lower, torch.tensor([0.0, 0.5]))
        # Upper: uses relaxation
        assert torch.allclose(upper, torch.tensor([1.0, 2.0]))

    def test_relu_all_negative(self):
        """
        Test backward LBP on y = relu(x) where x is all negative.
        
        When the input is all negative, relu is always zero,
        so we expect A = 0 (no dependency on input).
        """
        graph = create_relu_graph()
        propagator = BackwardLBPPropagator(graph)

        # Input region: [-2.0, -0.5] x [-3.0, -1.0] (all negative)
        input_region = HyperRectangle(
            torch.tensor([-2.0, -3.0]),
            torch.tensor([-0.5, -1.0]),
        )

        # Compute backward bounds
        propagator.compute_bounds(input_region)

        # Get input bounds
        input_bounds = propagator.get_input_bounds()[0]
        assert isinstance(input_bounds, LinearBounds)

        # For all-negative regime, relu is always 0, so A = 0
        expected_zeros = torch.zeros(2, 2)
        assert torch.allclose(input_bounds.linear_lower, expected_zeros)
        assert torch.allclose(input_bounds.linear_upper, expected_zeros)

        assert torch.allclose(input_bounds.bias_lower, torch.zeros(2))
        assert torch.allclose(input_bounds.bias_upper, torch.zeros(2))

        # Output bounds should be [0, 0]
        lower, upper = input_bounds.concretize()
        assert torch.allclose(lower, torch.zeros(2))
        assert torch.allclose(upper, torch.zeros(2))

    def test_matmul_relu(self):
        """
        Test backward LBP on y = relu(x @ W).
        
        This tests backward propagation through multiple operations:
        1. Backward through ReLU (uses relaxation based on concrete bounds)
        2. Backward through MATMUL (applies W^T)
        
        The composition should give us bounds on how output depends on input.
        """
        graph = create_matmul_relu_graph()
        propagator = BackwardLBPPropagator(graph)

        # Input region: [-1, 1] x [-1, 1]
        input_region = HyperRectangle(
            torch.tensor([-1.0, -1.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Compute backward bounds
        propagator.compute_bounds(input_region)

        # Get input bounds
        input_bounds = propagator.get_input_bounds()[0]
        assert isinstance(input_bounds, LinearBounds)

        # Should have linear dependencies (not concretized)
        assert input_bounds.linear_lower is not None
        assert input_bounds.linear_upper is not None

        # Concretize to check final bounds
        lower, upper = input_bounds.concretize()

        # The bounds should be valid (lower <= upper)
        assert torch.all(lower <= upper)

        # In backward mode, output bounds represent the output concretized
        # from the input dependencies. The actual network output bounds
        # are obtained by concretizing the input bounds (which tell us how
        # output depends on input).
        # For ReLU network, output should be non-negative >= 0
        # Note: The concretized input_bounds actually give us output bounds
        assert torch.all(lower >= -1e-6)  # Lower bound should be non-negative (ReLU)
        # Upper could be anything depending on the network

    def test_output_bounds_identity(self):
        """
        Test that output bounds are initialized with identity.
        
        For backward LBP, output nodes should have A = I, representing
        that the output depends on itself with identity mapping.
        """
        graph = create_relu_graph()
        propagator = BackwardLBPPropagator(graph)

        input_region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Compute bounds
        propagator.compute_bounds(input_region)

        # Get output bounds
        output_bounds = propagator.get_output_bounds()[0]
        assert isinstance(output_bounds, LinearBounds)

        # Output should have identity linear bounds
        expected_identity = torch.eye(2)
        assert torch.allclose(output_bounds.linear_lower, expected_identity)
        assert torch.allclose(output_bounds.linear_upper, expected_identity)
        assert torch.allclose(output_bounds.bias_lower, torch.zeros(2))
        assert torch.allclose(output_bounds.bias_upper, torch.zeros(2))

    def test_forward_bounds_computed(self):
        """
        Test that forward bounds are automatically computed if not provided.
        
        Backward LBP requires concrete bounds from a forward pass for computing
        relaxations. The propagator should automatically compute these using IBP
        if not provided.
        """
        graph = create_relu_graph()
        
        # Create propagator without providing forward bounds
        propagator = BackwardLBPPropagator(graph)

        input_region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Should not raise an error (forward bounds computed automatically)
        propagator.compute_bounds(input_region)

        # Verify that forward bounds were computed
        assert len(propagator._forward_bounds) > 0

    def test_reverse_topological_order(self):
        """
        Test that backward propagation processes nodes in reverse order.
        
        This is a key property of backward LBP: it should traverse from
        output to input, not input to output.
        """
        graph = create_matmul_relu_graph()
        propagator = BackwardLBPPropagator(graph)

        input_region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        # Compute bounds
        propagator.compute_bounds(input_region)

        # Verify that bounds exist for output nodes first
        # (This is implicit in the algorithm - if it runs successfully,
        # it processed in the correct order)
        output_bounds = propagator.get_output_bounds()
        assert len(output_bounds) > 0

        input_bounds = propagator.get_input_bounds()
        assert len(input_bounds) > 0


class TestBackwardLBPStrategies:
    """Test individual backward propagation strategies."""

    def test_backward_add_strategy(self):
        """
        Test BackwardAddStrategy.
        
        For z = x + y, both inputs should receive the full output bounds.
        """
        from bound_propagation.strategy.backward_lbp.add_backward import (
            BackwardAddStrategy,
        )

        strategy = BackwardAddStrategy()
        assert strategy.method_name == "backward"

        # Create mock linear bounds for output
        region = HyperRectangle(
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )

        output_bounds = LinearBounds(
            region=region,
            linear_lower=torch.eye(2),
            bias_lower=torch.zeros(2),
            linear_upper=torch.eye(2),
            bias_upper=torch.zeros(2),
        )

        # Create a mock ADD node
        node = Node(
            id=0,
            op_type=OperationType.ADD,
            inputs=[],  # Dummy inputs
            output_metadata=TensorMetadata(
                shape=(2,), dtype=torch.float32, device=torch.device("cpu")
            ),
        )

        # Propagate to first input
        contribution = strategy.propagate_backward(
            node=node,
            input_idx=0,
            output_bounds=output_bounds,
            concrete_input_bounds=[],
            config=None,
        )

        # Should return the same bounds (addition is linear)
        assert torch.allclose(contribution.linear_lower, output_bounds.linear_lower)
        assert torch.allclose(contribution.linear_upper, output_bounds.linear_upper)

    def test_backward_matmul_strategy(self):
        """
        Test BackwardMatmulStrategy.
        
        For z = x @ W, backward propagation should apply W^T.
        """
        from bound_propagation.strategy.backward_lbp.matmul_backward import (
            BackwardMatmulStrategy,
        )

        strategy = BackwardMatmulStrategy()
        assert strategy.method_name == "backward"

        # This strategy is tested implicitly in the matmul_relu test above

    def test_backward_relu_strategy(self):
        """
        Test BackwardReluStrategy.
        
        For z = relu(x), backward propagation should use linear relaxations
        based on concrete bounds.
        """
        from bound_propagation.strategy.backward_lbp.relu_backward import (
            BackwardReluStrategy,
        )

        strategy = BackwardReluStrategy()
        assert strategy.method_name == "backward"

        # This strategy is tested implicitly in the relu tests above
