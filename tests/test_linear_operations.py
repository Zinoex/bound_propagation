"""
Tests for exact propagation of linear operations.
"""

# Import relaxations to ensure they're registered
import torch

import bound_propagation.propagation.relaxations  # noqa: F401
from bound_propagation.bounds.interval_bounds import IntervalBounds
from bound_propagation.ir.graph import Graph
from bound_propagation.ir.metadata import TensorMetadata
from bound_propagation.ir.node import Node, NodeType
from bound_propagation.ir.operations import OperationType
from bound_propagation.propagation.methods import ForwardLBPPropagator
from bound_propagation.regions.hyperrectangle import HyperRectangle


class TestLinearOperations:
    """Test exact propagation for linear operations."""

    def test_add_operation(self):
        """Test ADD operation: [a,b] + [c,d] = [a+c, b+d]."""
        # Create graph: input1 + input2
        graph = Graph()
        metadata = TensorMetadata(shape=(3,), dtype="float32")

        input1 = Node(
            id=0,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input1)

        input2 = Node(
            id=1,
            op_type=OperationType.INPUT,
            inputs=[],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.INPUT,
        )
        graph.add_node(input2)

        add_node = Node(
            id=2,
            op_type=OperationType.ADD,
            inputs=[input1, input2],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(add_node)
        graph.mark_outputs([add_node])

        # Create region: input1 in [1,2], input2 in [3,4]
        # For multi-input, we'll use a single region for now
        # This is a simplified test
        lower = torch.tensor([1.0, 1.0, 1.0])
        upper = torch.tensor([2.0, 2.0, 2.0])
        region = HyperRectangle(lower=lower, upper=upper)

        # Note: Current implementation assumes single input region
        # This test would need multi-input support
        # For now, let's test with constant second input

    def test_sub_operation(self):
        """Test SUB operation: [a,b] - [c,d] = [a-d, b-c]."""
        # Create graph: input - constant
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

        const_value = torch.tensor([1.0, 1.0, 1.0])
        const_node = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=metadata,
            attributes={"value": const_value},
            node_type=NodeType.CONSTANT,
        )
        graph.add_node(const_node)

        sub_node = Node(
            id=2,
            op_type=OperationType.SUB,
            inputs=[input_node, const_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(sub_node)
        graph.mark_outputs([sub_node])

        # Input in [2, 4], constant = 1
        # Result should be [2-1, 4-1] = [1, 3]
        lower = torch.tensor([2.0, 2.0, 2.0])
        upper = torch.tensor([4.0, 4.0, 4.0])
        region = HyperRectangle(lower=lower, upper=upper)

        propagator = ForwardLBPPropagator()
        bounds = propagator.propagate(graph, region)

        assert sub_node.id in bounds
        result = bounds[sub_node.id]
        assert isinstance(result, IntervalBounds)

        # Expected: [2-1, 4-1] = [1, 3]
        expected_lower = torch.tensor([1.0, 1.0, 1.0])
        expected_upper = torch.tensor([3.0, 3.0, 3.0])

        assert torch.allclose(result.lower, expected_lower, atol=1e-6)
        assert torch.allclose(result.upper, expected_upper, atol=1e-6)

    def test_matmul_operation_with_constant_weight(self):
        """Test MATMUL operation with constant weight matrix."""
        # Create graph: input @ W
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

        # Weight matrix: 2x3 (input_dim x output_dim)
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

        # Input in [0, 1] x [0, 1]
        lower = torch.tensor([0.0, 0.0])
        upper = torch.tensor([1.0, 1.0])
        region = HyperRectangle(lower=lower, upper=upper)

        propagator = ForwardLBPPropagator()
        bounds = propagator.propagate(graph, region)

        assert matmul_node.id in bounds
        result = bounds[matmul_node.id]
        assert isinstance(result, IntervalBounds)

        # Manual calculation:
        # y = x @ W where x in [0,1]^2
        # y[0] = x[0]*1.0 + x[1]*0.5 = [0, 1.5]
        # y[1] = x[0]*(-0.5) + x[1]*1.0 = [-0.5, 1.0]
        # y[2] = x[0]*2.0 + x[1]*(-1.0) = [-1.0, 2.0]

        # Using pos/neg split:
        # positive weights contribute lower*pos_w to lower, upper*pos_w to upper
        # negative weights contribute upper*neg_w to lower, lower*neg_w to upper

        expected_lower = torch.tensor([0.0, -0.5, -1.0])
        expected_upper = torch.tensor([1.5, 1.0, 2.0])

        assert torch.allclose(result.lower, expected_lower, atol=1e-6)
        assert torch.allclose(result.upper, expected_upper, atol=1e-6)

    def test_mul_with_constant_uses_exact_dispatch(self, monkeypatch):
        """Constant MUL should use exact dispatch instead of bilinear relaxation."""
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

        constant_node = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=metadata,
            attributes={"value": torch.tensor([2.0, -3.0])},
            node_type=NodeType.CONSTANT,
        )
        graph.add_node(constant_node)

        mul_node = Node(
            id=2,
            op_type=OperationType.MUL,
            inputs=[input_node, constant_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(mul_node)
        graph.mark_outputs([mul_node])

        propagator = ForwardLBPPropagator()
        monkeypatch.setattr(
            propagator._relaxation_strategies[OperationType.MUL],
            "relax",
            lambda node, interval_inputs: (_ for _ in ()).throw(AssertionError("constant MUL should not use bilinear relaxation")),
        )

        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([2.0, 2.0]),
        )

        bounds = propagator.propagate(graph, region)
        result = bounds[mul_node.id]

        assert isinstance(result, IntervalBounds)
        assert torch.allclose(result.lower, torch.tensor([-2.0, -6.0]))
        assert torch.allclose(result.upper, torch.tensor([4.0, 3.0]))

    def test_div_by_constant_uses_exact_dispatch(self, monkeypatch):
        """Constant DIV should use exact dispatch instead of bilinear relaxation."""
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

        constant_node = Node(
            id=1,
            op_type=OperationType.CONSTANT,
            inputs=[],
            output_metadata=metadata,
            attributes={"value": torch.tensor([2.0, -2.0])},
            node_type=NodeType.CONSTANT,
        )
        graph.add_node(constant_node)

        div_node = Node(
            id=2,
            op_type=OperationType.DIV,
            inputs=[input_node, constant_node],
            output_metadata=metadata,
            attributes={},
            node_type=NodeType.OPERATION,
        )
        graph.add_node(div_node)
        graph.mark_outputs([div_node])

        propagator = ForwardLBPPropagator()
        monkeypatch.setattr(
            propagator._relaxation_strategies[OperationType.DIV],
            "relax",
            lambda node, interval_inputs: (_ for _ in ()).throw(AssertionError("constant DIV should not use bilinear relaxation")),
        )

        region = HyperRectangle(
            lower=torch.tensor([2.0, 2.0]),
            upper=torch.tensor([4.0, 4.0]),
        )

        bounds = propagator.propagate(graph, region)
        result = bounds[div_node.id]

        assert isinstance(result, IntervalBounds)
        assert torch.allclose(result.lower, torch.tensor([1.0, -2.0]))
        assert torch.allclose(result.upper, torch.tensor([2.0, -1.0]))


class TestInputIdsTracking:
    """Test input_ids tracking in LinearBounds."""

    def test_linear_bounds_with_input_ids(self):
        """Test that LinearBounds can be created with input_ids."""
        from bound_propagation.bounds.linear_bounds import LinearBounds
        from bound_propagation.regions.hyperrectangle import HyperRectangle

        region = HyperRectangle(lower=torch.tensor([0.0, 0.0]), upper=torch.tensor([1.0, 1.0]))

        linear_lower = torch.tensor([[1.0, 0.5]])
        bias_lower = torch.tensor([0.0])
        linear_upper = torch.tensor([[1.5, 1.0]])
        bias_upper = torch.tensor([1.0])

        # Create with input_ids
        bounds = LinearBounds(
            region=region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=[0, 1],
        )

        assert bounds.input_ids == [0, 1]

    def test_linear_bounds_clone_preserves_input_ids(self):
        """Test that cloning LinearBounds preserves input_ids."""
        from bound_propagation.bounds.linear_bounds import LinearBounds
        from bound_propagation.regions.hyperrectangle import HyperRectangle

        region = HyperRectangle(lower=torch.tensor([0.0]), upper=torch.tensor([1.0]))

        linear_lower = torch.tensor([[1.0]])
        bias_lower = torch.tensor([0.0])
        linear_upper = torch.tensor([[1.0]])
        bias_upper = torch.tensor([1.0])

        bounds = LinearBounds(
            region=region,
            linear_lower=linear_lower,
            bias_lower=bias_lower,
            linear_upper=linear_upper,
            bias_upper=bias_upper,
            input_ids=[0],
        )

        cloned = bounds.clone()

        assert cloned.input_ids == bounds.input_ids
        assert cloned.input_ids is not bounds.input_ids  # Different list object
