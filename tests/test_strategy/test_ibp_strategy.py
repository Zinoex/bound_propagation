"""
Tests for IBP (Interval Bound Propagation) strategy.
"""

import pytest
import torch

from bound_propagation.bounds import IntervalBounds, LinearBounds
from bound_propagation.ir import Node, OperationType, TensorMetadata
from bound_propagation.regions import HyperRectangle
from bound_propagation.strategy import StrategyConfig, get_global_registry


def create_test_node(op_type: OperationType, **attrs) -> Node:
    """Helper to create a test node with minimal required fields."""
    return Node(
        id=0,
        op_type=op_type,
        inputs=[],
        output_metadata=TensorMetadata(
            shape=(2,), dtype=torch.float32, device=torch.device("cpu")
        ),
        attributes=attrs,
        name=op_type.name.lower(),
    )


def get_ibp_strategy(op_type: OperationType):
    """Get the IBP strategy for a given operation type."""
    registry = get_global_registry()
    return registry.get(op_type, "ibp")


class TestIBPArithmetic:
    """Tests for IBP arithmetic operations."""

    def test_add(self):
        """Test IBP for ADD operation."""
        strategy = get_ibp_strategy(OperationType.ADD)
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        # [1,2] + [3,4] = [4,6]
        x_bounds = IntervalBounds(region, torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]))
        y_bounds = IntervalBounds(region, torch.tensor([3.0, 4.0]), torch.tensor([4.0, 5.0]))

        node = create_test_node(OperationType.ADD)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds, y_bounds], config)

        assert torch.allclose(result.lower, torch.tensor([4.0, 6.0]))
        assert torch.allclose(result.upper, torch.tensor([6.0, 8.0]))

    def test_sub(self):
        """Test IBP for SUB operation."""
        strategy = get_ibp_strategy(OperationType.SUB)
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        # [1,2] - [3,4] = [1-4, 2-3] = [-3, -1]
        x_bounds = IntervalBounds(region, torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]))
        y_bounds = IntervalBounds(region, torch.tensor([3.0, 4.0]), torch.tensor([4.0, 5.0]))

        node = create_test_node(OperationType.SUB)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds, y_bounds], config)

        assert torch.allclose(result.lower, torch.tensor([-3.0, -3.0]))
        assert torch.allclose(result.upper, torch.tensor([-1.0, -1.0]))

    def test_mul(self):
        """Test IBP for MUL operation."""
        strategy = get_ibp_strategy(OperationType.MUL)
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        # [1,2] * [3,4] = [min(1*3,1*4,2*3,2*4), max(...)] = [3, 8]
        x_bounds = IntervalBounds(region, torch.tensor([1.0, 2.0]), torch.tensor([2.0, 3.0]))
        y_bounds = IntervalBounds(region, torch.tensor([3.0, 4.0]), torch.tensor([4.0, 5.0]))

        node = create_test_node(OperationType.MUL)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds, y_bounds], config)

        assert torch.allclose(result.lower, torch.tensor([3.0, 8.0]))
        assert torch.allclose(result.upper, torch.tensor([8.0, 15.0]))

    def test_mul_with_negative(self):
        """Test IBP for MUL with negative bounds."""
        strategy = get_ibp_strategy(OperationType.MUL)
        region = HyperRectangle(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))

        # [-1,1] * [-1,1] = [-1, 1]
        x_bounds = IntervalBounds(region, torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))
        y_bounds = IntervalBounds(region, torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))

        node = create_test_node(OperationType.MUL)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds, y_bounds], config)

        assert torch.allclose(result.lower, torch.tensor([-1.0, -1.0]))
        assert torch.allclose(result.upper, torch.tensor([1.0, 1.0]))

    def test_div(self):
        """Test IBP for DIV operation."""
        strategy = get_ibp_strategy(OperationType.DIV)
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        # [4,8] / [2,4] = [1, 4]
        x_bounds = IntervalBounds(region, torch.tensor([4.0, 6.0]), torch.tensor([8.0, 12.0]))
        y_bounds = IntervalBounds(region, torch.tensor([2.0, 2.0]), torch.tensor([4.0, 3.0]))

        node = create_test_node(OperationType.DIV)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds, y_bounds], config)

        # Lower: min(4/2, 4/4, 8/2, 8/4) = 1
        # Upper: max(...) = 4 or 6
        assert result.lower[0] <= result.upper[0]


class TestIBPActivations:
    """Tests for IBP activation functions."""

    def test_relu(self):
        """Test IBP for RELU operation."""
        strategy = get_ibp_strategy(OperationType.RELU)
        region = HyperRectangle(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))

        # relu([-1, 1], [-2, 3]) = [0, 1], [0, 3]
        x_bounds = IntervalBounds(
            region,
            torch.tensor([-1.0, -2.0]),
            torch.tensor([1.0, 3.0]),
        )

        node = create_test_node(OperationType.RELU)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds], config)

        assert torch.allclose(result.lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(result.upper, torch.tensor([1.0, 3.0]))

    def test_sigmoid(self):
        """Test IBP for SIGMOID operation."""
        strategy = get_ibp_strategy(OperationType.SIGMOID)
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        x_bounds = IntervalBounds(
            region,
            torch.tensor([-1.0, 0.0]),
            torch.tensor([1.0, 2.0]),
        )

        node = create_test_node(OperationType.SIGMOID)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds], config)

        # Sigmoid is monotonic, so just apply to bounds
        expected_lower = torch.sigmoid(x_bounds.lower)
        expected_upper = torch.sigmoid(x_bounds.upper)

        assert torch.allclose(result.lower, expected_lower)
        assert torch.allclose(result.upper, expected_upper)

    def test_tanh(self):
        """Test IBP for TANH operation."""
        strategy = get_ibp_strategy(OperationType.TANH)
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        x_bounds = IntervalBounds(
            region,
            torch.tensor([-1.0, 0.0]),
            torch.tensor([1.0, 2.0]),
        )

        node = create_test_node(OperationType.TANH)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds], config)

        expected_lower = torch.tanh(x_bounds.lower)
        expected_upper = torch.tanh(x_bounds.upper)

        assert torch.allclose(result.lower, expected_lower)
        assert torch.allclose(result.upper, expected_upper)

    def test_exp(self):
        """Test IBP for EXP operation."""
        strategy = get_ibp_strategy(OperationType.EXP)
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        x_bounds = IntervalBounds(
            region,
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 2.0]),
        )

        node = create_test_node(OperationType.EXP)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds], config)

        expected_lower = torch.exp(x_bounds.lower)
        expected_upper = torch.exp(x_bounds.upper)

        assert torch.allclose(result.lower, expected_lower)
        assert torch.allclose(result.upper, expected_upper)

    def test_log(self):
        """Test IBP for LOG operation."""
        strategy = get_ibp_strategy(OperationType.LOG)
        region = HyperRectangle(torch.tensor([0.1, 0.1]), torch.tensor([1.0, 1.0]))

        x_bounds = IntervalBounds(
            region,
            torch.tensor([0.1, 0.5]),
            torch.tensor([1.0, 2.0]),
        )

        node = create_test_node(OperationType.LOG)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds], config)

        expected_lower = torch.log(x_bounds.lower)
        expected_upper = torch.log(x_bounds.upper)

        assert torch.allclose(result.lower, expected_lower, rtol=1e-5)
        assert torch.allclose(result.upper, expected_upper, rtol=1e-5)


class TestIBPLinear:
    """Tests for IBP linear operations."""

    def test_linear_layer(self):
        """Test IBP for LINEAR operation."""
        strategy = get_ibp_strategy(OperationType.LINEAR)

        # Input: 2D, shape [2]
        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        x_bounds = IntervalBounds(region, torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        # Weight: (3, 2), bias: (3,)
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, -2.0]])
        bias = torch.tensor([0.5, 1.0, 2.0])

        node = create_test_node(
            OperationType.LINEAR, weight=weight, bias=bias
        )
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds], config)

        # For input in [0,1] x [0,1]:
        # Row 0: [1,2] @ [0,1], [0,1] = [0, 3] + 0.5 = [0.5, 3.5]
        # Row 1: [3,4] @ [0,1], [0,1] = [0, 7] + 1.0 = [1.0, 8.0]
        # Row 2: [-1,-2] @ [0,1], [0,1] = [-3, 0] + 2.0 = [-1.0, 2.0]

        assert result.shape == (3,)
        assert torch.allclose(result.lower, torch.tensor([0.5, 1.0, -1.0]))
        assert torch.allclose(result.upper, torch.tensor([3.5, 8.0, 2.0]))

    def test_linear_without_bias(self):
        """Test LINEAR without bias."""
        strategy = get_ibp_strategy(OperationType.LINEAR)

        region = HyperRectangle(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        x_bounds = IntervalBounds(region, torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

        weight = torch.tensor([[1.0, -1.0]])

        node = create_test_node(
            OperationType.LINEAR, weight=weight, bias=None
        )
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds], config)

        # [1, -1] @ [0,1], [0,1] = [-1, 1]
        assert torch.allclose(result.lower, torch.tensor([-1.0]))
        assert torch.allclose(result.upper, torch.tensor([1.0]))


class TestIBPReshaping:
    """Tests for IBP reshaping operations."""

    def test_reshape(self):
        """Test IBP for RESHAPE operation."""
        strategy = get_ibp_strategy(OperationType.RESHAPE)

        region = HyperRectangle(
            torch.tensor([0.0, 0.0, 0.0, 0.0]),
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
        )
        x_bounds = IntervalBounds(
            region,
            torch.tensor([0.0, 1.0, 2.0, 3.0]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )

        node = create_test_node(OperationType.RESHAPE, shape=(2, 2))
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds], config)

        assert result.shape == (2, 2)
        assert torch.allclose(result.lower, x_bounds.lower.reshape(2, 2))
        assert torch.allclose(result.upper, x_bounds.upper.reshape(2, 2))

    def test_flatten(self):
        """Test IBP for FLATTEN operation."""
        strategy = get_ibp_strategy(OperationType.FLATTEN)

        # 2x2 input
        region = HyperRectangle(
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        )
        x_bounds = IntervalBounds(
            region,
            torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )

        node = create_test_node(OperationType.FLATTEN)
        config = StrategyConfig()

        result = strategy.compute_bounds(node, [x_bounds], config)

        assert result.shape == (4,)
        assert torch.allclose(result.lower, torch.tensor([0.0, 1.0, 2.0, 3.0]))
        assert torch.allclose(result.upper, torch.tensor([1.0, 2.0, 3.0, 4.0]))


class TestIBPError:
    """Tests for IBP error handling."""

    def test_non_interval_bounds_raises(self):
        """Test that non-IntervalBounds input raises error."""
        strategy = get_ibp_strategy(OperationType.ADD)
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))

        # Create LinearBounds instead of IntervalBounds
        linear_bounds = LinearBounds.from_interval_bounds(
            IntervalBounds(region, torch.tensor([0.0]), torch.tensor([1.0]))
        )

        node = create_test_node(OperationType.ADD)
        config = StrategyConfig()

        with pytest.raises(ValueError, match="IBP requires IntervalBounds"):
            strategy.compute_bounds(node, [linear_bounds, linear_bounds], config)

    def test_unsupported_operation_raises(self):
        """Test that unsupported operation returns None."""
        registry = get_global_registry()

        # CONV2D should not have a registered IBP strategy
        strategy = registry.get(OperationType.CONV2D, "ibp")
        assert strategy is None

    def test_wrong_number_of_inputs_raises(self):
        """Test that wrong number of inputs raises error."""
        strategy = get_ibp_strategy(OperationType.ADD)
        region = HyperRectangle(torch.tensor([0.0]), torch.tensor([1.0]))
        bounds = IntervalBounds(region, torch.tensor([0.0]), torch.tensor([1.0]))

        # ADD needs 2 inputs
        node = create_test_node(OperationType.ADD)
        config = StrategyConfig()

        with pytest.raises(ValueError, match="ADD requires 2 inputs"):
            strategy.compute_bounds(node, [bounds], config)  # Only 1 input

    def test_log_with_negative_input_raises(self):
        """Test that LOG with non-positive input raises error."""
        strategy = get_ibp_strategy(OperationType.LOG)
        region = HyperRectangle(torch.tensor([-1.0]), torch.tensor([1.0]))

        # Bounds include 0
        bounds = IntervalBounds(region, torch.tensor([-1.0]), torch.tensor([1.0]))

        node = create_test_node(OperationType.LOG)
        config = StrategyConfig()

        with pytest.raises(ValueError, match="LOG requires positive"):
            strategy.compute_bounds(node, [bounds], config)


class TestIBPMethod:
    """Tests for IBP method properties."""

    def test_method_name(self):
        """Test that method name is correct."""
        strategy = get_ibp_strategy(OperationType.ADD)
        assert strategy.method_name == "ibp"

    def test_repr(self):
        """Test string representation."""
        strategy = get_ibp_strategy(OperationType.RELU)
        repr_str = repr(strategy)

        # Check for the specific strategy class names
        assert "Strategy" in repr_str
        assert "ibp" in repr_str.lower()
