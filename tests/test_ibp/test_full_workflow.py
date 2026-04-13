"""
End-to-end IBP workflow tests.

Tests the complete IBP pipeline:
1. Tracing PyTorch functions/modules
2. Converting to IR Graph
3. Constructing IBP bounding strategies
4. Propagating bounds through the graph
5. Verifying correctness of output bounds

These tests ensure the full workflow works for various operations and network architectures.
"""

from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation import IBPPropagator
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer, GraphConverter


def _trace_and_convert(fn_or_module, example_inputs: tuple[torch.Tensor, ...]):
    """Helper to trace a function/module and convert to IR Graph."""
    tracer = BoundPropagationTracer()
    fx_graph = tracer.trace(fn_or_module)
    fx_module = torch.fx.GraphModule(tracer.root, fx_graph)

    converter = GraphConverter(fx_module)
    ir_graph = converter.convert(example_inputs=example_inputs)

    return ir_graph


class TestIBPWorkflowSimpleFunctions:
    """Test IBP workflow with simple mathematical functions."""

    def test_single_relu(self) -> None:
        """Test IBP on a simple ReLU activation."""

        def relu_fn(x):
            return torch.relu(x)

        # Trace and convert
        example_input = torch.randn(
            3,
        )
        graph = _trace_and_convert(relu_fn, (example_input,))

        # Create propagator
        propagator = IBPPropagator(graph)

        # Define input region: [-2, 3] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0, -2.0]),
            upper=torch.tensor([3.0, 3.0, 3.0]),
        )

        # Propagate bounds
        outputs = propagator.propagate([input_region])

        # Verify: ReLU([−2, 3]) = [0, 3]
        assert len(outputs) == 1
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(out.upper, torch.tensor([3.0, 3.0, 3.0]))

    def test_add_and_relu(self) -> None:
        """Test IBP on addition followed by ReLU."""

        def add_relu_fn(x):
            y = x + torch.tensor([1.0, -1.0, 0.5])
            return torch.relu(y)

        example_input = torch.randn(
            3,
        )
        graph = _trace_and_convert(add_relu_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [0, 1] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])

        # After add: [1, 2], [-1, 0], [0.5, 1.5]
        # After ReLU: [1, 2], [0, 0], [0.5, 1.5]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([1.0, 0.0, 0.5]))
        assert torch.allclose(out.upper, torch.tensor([2.0, 0.0, 1.5]))

    def test_mul_operation(self) -> None:
        """Test IBP on element-wise multiplication."""

        def mul_fn(x):
            return x * torch.tensor([2.0, -1.0, 3.0])

        example_input = torch.randn(
            3,
        )
        graph = _trace_and_convert(mul_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [-1, 2] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([2.0, 2.0, 2.0]),
        )

        outputs = propagator.propagate([input_region])

        # Element 0: [-1, 2] * 2 = [-2, 4]
        # Element 1: [-1, 2] * -1 = [-2, 1]
        # Element 2: [-1, 2] * 3 = [-3, 6]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([-2.0, -2.0, -3.0]))
        assert torch.allclose(out.upper, torch.tensor([4.0, 1.0, 6.0]))

    def test_exp_operation(self) -> None:
        """Test IBP on exponential function."""

        def exp_fn(x):
            return torch.exp(x)

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(exp_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [0, 1]
        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])

        # exp([0, 1]) = [exp(0), exp(1)] = [1, e]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([1.0, 1.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.exp(torch.tensor([1.0, 1.0])), atol=1e-5)

    def test_log_operation(self) -> None:
        """Test IBP on logarithm function."""

        def log_fn(x):
            return torch.log(x)

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(log_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [1, e]
        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 1.0]),
            upper=torch.tensor([2.718281828, 2.718281828]),
        )

        outputs = propagator.propagate([input_region])

        # log([1, e]) = [0, 1]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([0.0, 0.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([1.0, 1.0]), atol=1e-5)

    def test_sigmoid_operation(self) -> None:
        """Test IBP on sigmoid activation."""

        def sigmoid_fn(x):
            return torch.sigmoid(x)

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(sigmoid_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [-1, 1]
        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])

        # sigmoid([-1, 1]) = [sigmoid(-1), sigmoid(1)]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        expected_lower = torch.sigmoid(torch.tensor([-1.0, -1.0]))
        expected_upper = torch.sigmoid(torch.tensor([1.0, 1.0]))
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)

    def test_tanh_operation(self) -> None:
        """Test IBP on tanh activation."""

        def tanh_fn(x):
            return torch.tanh(x)

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(tanh_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [-2, 2]
        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0]),
            upper=torch.tensor([2.0, 2.0]),
        )

        outputs = propagator.propagate([input_region])

        # tanh([-2, 2]) = [tanh(-2), tanh(2)]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        expected_lower = torch.tanh(torch.tensor([-2.0, -2.0]))
        expected_upper = torch.tanh(torch.tensor([2.0, 2.0]))
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)


class TestIBPWorkflowNeuralNetworks:
    """Test IBP workflow with neural network-like operations."""

    def test_matmul_with_bias(self) -> None:
        """Test IBP on matrix multiplication with bias (simulating linear layer)."""

        def matmul_with_bias_fn(x):
            # Simulate a linear layer using matmul
            # Weight matrix (transposed): (2, 3)
            weight_T = torch.tensor([[1.0, -1.0, 0.5], [2.0, 0.0, -2.0]])
            bias = torch.tensor([0.5, -0.5])
            # y = xW^T + b where x is (3,), W^T is (2, 3)
            return x @ weight_T.T + bias

        # Trace and convert
        example_input = torch.randn(
            3,
        )
        graph = _trace_and_convert(matmul_with_bias_fn, (example_input,))

        # Create propagator
        propagator = IBPPropagator(graph)

        # Input: [-1, 1] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])

        # Output 0: [1, -1, 0.5] @ [-1,1], [-1,1], [-1,1] + 0.5
        # min: 1*(-1) + (-1)*1 + 0.5*(-1) + 0.5 = -2.0
        # max: 1*1 + (-1)*(-1) + 0.5*1 + 0.5 = 3.0
        # Output 1: [2, 0, -2] @ ... - 0.5
        # min: 2*(-1) + 0*(-1) + (-2)*1 - 0.5 = -4.5
        # max: 2*1 + 0*1 + (-2)*(-1) - 0.5 = 3.5
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([-2.0, -4.5]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([3.0, 3.5]), atol=1e-5)

    def test_two_layer_network(self) -> None:
        """Test IBP on a two-layer feedforward network."""

        def two_layer_fn(x):
            # First layer
            w1 = torch.eye(3, 4)
            x = x @ w1.T
            x = torch.relu(x)
            # Second layer
            w2 = torch.eye(2, 3)
            x = x @ w2.T
            return x

        example_input = torch.randn(
            4,
        )
        graph = _trace_and_convert(two_layer_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [0, 1] for each dimension
        input_region = HyperRectangle(
            lower=torch.zeros(4),
            upper=torch.ones(4),
        )

        outputs = propagator.propagate([input_region])

        # First layer output: identity-like so [0,1] for first 3 dims
        # After ReLU: [0,1] (no change as all positive)
        # Second layer output: identity-like so [0,1] for first 2 dims
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (2,)
        assert out.upper.shape == (2,)
        # With identity matrices, bounds should be exactly [0, 1]
        assert torch.allclose(out.lower, torch.zeros(2), atol=1e-5)
        assert torch.allclose(out.upper, torch.ones(2), atol=1e-5)

    def test_network_with_multiple_activations(self) -> None:
        """Test IBP on network with ReLU, Sigmoid, and Tanh."""

        def mixed_activation_fn(x):
            # First layer
            w1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            x = x @ w1.T
            x = torch.relu(x)
            # Second layer
            w2 = torch.eye(3)
            x = x @ w2.T
            x = torch.sigmoid(x)
            # Third layer
            w3 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            x = x @ w3.T
            x = torch.tanh(x)
            return x

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(mixed_activation_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [-1, 1] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])

        # Calculate exact bounds:
        # After first matmul: [[-1,1], [-1,1], [-2,2]]
        # After ReLU: [[0,1], [0,1], [0,2]]
        # After sigmoid: [[0.5,0.7311], [0.5,0.7311], [0.5,0.8808]]
        # After third matmul: [[0.5,0.7311], [0.5,0.7311]]
        # After tanh: [[tanh(0.5),tanh(0.7311)], [tanh(0.5),tanh(0.7311)]]
        sigmoid_1 = torch.sigmoid(torch.tensor(1.0)).item()
        tanh_05 = torch.tanh(torch.tensor(0.5)).item()
        tanh_sigmoid_1 = torch.tanh(torch.tensor(sigmoid_1)).item()
        expected_lower = torch.tensor([tanh_05, tanh_05])
        expected_upper = torch.tensor([tanh_sigmoid_1, tanh_sigmoid_1])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)


class TestIBPWorkflowComplexOperations:
    """Test IBP workflow with more complex operations."""

    def test_matmul_operation(self) -> None:
        """Test IBP on matrix multiplication."""

        def matmul_fn(x):
            # x: (2, 3), weight: (3, 4) -> output: (2, 4)
            weight = torch.tensor(
                [
                    [1.0, 0.0, -1.0, 2.0],
                    [0.0, 1.0, 0.0, -1.0],
                    [0.5, 0.5, 0.5, 0.5],
                ]
            )
            return x @ weight

        example_input = torch.randn(2, 3)
        graph = _trace_and_convert(matmul_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [0, 1] for all elements
        input_region = HyperRectangle(
            lower=torch.zeros(2, 3),
            upper=torch.ones(2, 3),
        )

        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (2, 4)
        assert out.upper.shape == (2, 4)
        # Calculate exact bounds for each output element:
        # Input: [0,1] for all elements (2,3)
        # Weight: [[1, 0, -1, 2], [0, 1, 0, -1], [0.5, 0.5, 0.5, 0.5]]
        # For each row in input:
        #   Col 0: [0,1]*1 + [0,1]*0 + [0,1]*0.5 = [0, 1.5]
        #   Col 1: [0,1]*0 + [0,1]*1 + [0,1]*0.5 = [0, 1.5]
        #   Col 2: [0,1]*(-1) + [0,1]*0 + [0,1]*0.5 = [-1, 0.5]
        #   Col 3: [0,1]*2 + [0,1]*(-1) + [0,1]*0.5 = [-1, 2.5]  (min: 0*2 + 1*(-1) + 0*0.5 = -1)
        expected_lower = torch.tensor([[0.0, 0.0, -1.0, -1.0], [0.0, 0.0, -1.0, -1.0]])
        expected_upper = torch.tensor([[1.5, 1.5, 0.5, 2.5], [1.5, 1.5, 0.5, 2.5]])
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)

    def test_div_operation(self) -> None:
        """Test IBP on division."""

        def div_fn(x):
            return x / torch.tensor([2.0, 4.0])

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(div_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [2, 8]
        input_region = HyperRectangle(
            lower=torch.tensor([2.0, 4.0]),
            upper=torch.tensor([8.0, 16.0]),
        )

        outputs = propagator.propagate([input_region])

        # Element 0: [2, 8] / 2 = [1, 4]
        # Element 1: [4, 16] / 4 = [1, 4]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([1.0, 1.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([4.0, 4.0]), atol=1e-5)

    def test_sub_operation(self) -> None:
        """Test IBP on subtraction with two abstract inputs."""

        def sub_fn(x, y):
            return x - y

        example_x = torch.randn(
            3,
        )
        example_y = torch.randn(
            3,
        )
        graph = _trace_and_convert(sub_fn, (example_x, example_y))

        propagator = IBPPropagator(graph)

        # Input 1: [5, 10]
        # Input 2: [1, 3]
        input_regions = [
            HyperRectangle(
                lower=torch.tensor([5.0, 5.0, 5.0]),
                upper=torch.tensor([10.0, 10.0, 10.0]),
            ),
            HyperRectangle(
                lower=torch.tensor([1.0, 2.0, 3.0]),
                upper=torch.tensor([1.0, 2.0, 3.0]),
            ),
        ]

        outputs = propagator.propagate(input_regions)

        # Output: [5,10] - [1,2,3] = [4,9], [3,8], [2,7]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([4.0, 3.0, 2.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([9.0, 8.0, 7.0]), atol=1e-5)

    def test_chained_operations(self) -> None:
        """Test IBP on a chain of various operations."""

        def complex_fn(x):
            x = x + torch.tensor([1.0, 2.0])  # Add with constant
            x = x * torch.tensor([2.0, 0.5])  # Mul with constant
            x = torch.relu(x)  # ReLU
            x = torch.sigmoid(x)  # Sigmoid
            return x

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(complex_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input: [0, 1]
        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])

        # Calculate exact bounds:
        # Input: [0, 0] to [1, 1]
        # After add: [1, 2] to [2, 3]
        # After mul: [2, 1] to [4, 1.5]
        # After ReLU: [2, 1] to [4, 1.5] (all positive, no change)
        # After sigmoid: [sigmoid(2), sigmoid(1)] to [sigmoid(4), sigmoid(1.5)]
        expected_lower = torch.sigmoid(torch.tensor([2.0, 1.0]))
        expected_upper = torch.sigmoid(torch.tensor([4.0, 1.5]))
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)


class TestIBPWorkflowBatchedInputs:
    """Test IBP workflow with batched inputs."""

    def test_batched_matmul(self) -> None:
        """Test IBP with batched inputs through matmul."""

        weight = torch.tensor([[1.0, 0.5, -0.5, 0.25], [0.0, 1.0, 0.0, -1.0], [-0.5, 0.5, 1.0, 0.5]])

        def matmul_fn(x):
            # x: (5, 4), weight: (3, 4) -> output: (5, 3)
            return x @ weight.T

        # Use batch dimension
        example_input = torch.randn(5, 4)  # Batch size 5
        graph = _trace_and_convert(matmul_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input region for batched data
        input_region = HyperRectangle(
            lower=torch.zeros(5, 4),
            upper=torch.ones(5, 4),
        )

        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (5, 3)
        assert out.upper.shape == (5, 3)
        # Calculate exact bounds for matmul with weight:
        # weight.T = [[1, 0, -0.5], [0.5, 1, 0.5], [-0.5, 0, 1], [0.25, -1, 0.5]]
        # For input [0,1]^4:
        #   Col 0: [0,1]*1 + [0,1]*0.5 + [0,1]*(-0.5) + [0,1]*0.25 = [-0.5, 1.75]
        #   Col 1: [0,1]*0 + [0,1]*1 + [0,1]*0 + [0,1]*(-1) = [-1, 1]
        #   Col 2: [0,1]*(-0.5) + [0,1]*0.5 + [0,1]*1 + [0,1]*0.5 = [-0.5, 2.5]
        # Min col 0: 0*1 + 0*0.5 + 1*(-0.5) + 0*0.25 = -0.5
        # Max col 0: 1*1 + 1*0.5 + 0*(-0.5) + 1*0.25 = 1.75
        # Min col 1: 0*0 + 0*1 + 0*0 + 1*(-1) = -1
        # Max col 1: 0*0 + 1*1 + 0*0 + 0*(-1) = 1
        # Min col 2: 1*(-0.5) + 0*0.5 + 0*1 + 0*0.5 = -0.5
        # Max col 2: 0*(-0.5) + 1*0.5 + 1*1 + 1*0.5 = 2.0
        expected_lower = torch.tensor(
            [[-0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, -1.0, -0.5], [-0.5, -1.0, -0.5]]
        )
        expected_upper = torch.tensor(
            [[1.75, 1.0, 2.0], [1.75, 1.0, 2.0], [1.75, 1.0, 2.0], [1.75, 1.0, 2.0], [1.75, 1.0, 2.0]]
        )
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)

    def test_batched_complex_network(self) -> None:
        """Test IBP with batched inputs through a complex network."""

        # Define fixed weights instead of random
        w1 = torch.tensor(
            [
                [1.0, 0.5, -0.5],
                [0.0, 1.0, 0.0],
                [-0.5, 0.5, 1.0],
                [0.5, 0.0, 0.5],
                [0.0, -0.5, 0.5],
            ]
        )
        w2 = torch.tensor([[1.0, 0.0, -1.0, 0.5, 0.0], [0.0, 1.0, 0.0, -0.5, 1.0]])

        def complex_fn(x):
            # x: (10, 3)
            x = x @ w1.T
            x = torch.relu(x)
            x = x @ w2.T
            x = torch.sigmoid(x)
            return x

        # Batched input
        example_input = torch.randn(10, 3)  # Batch size 10
        graph = _trace_and_convert(complex_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Input region
        input_region = HyperRectangle(
            lower=-torch.ones(10, 3),
            upper=torch.ones(10, 3),
        )

        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (10, 2)
        assert out.upper.shape == (10, 2)

        # Calculate exact bounds step by step:
        # Input: [-1, -1, -1] to [1, 1, 1] for each batch element
        # w1.T rows: [1, 0.5, -0.5], [0, 1, 0], [-0.5, 0.5, 1], [0.5, 0, 0.5], [0, -0.5, 0.5]
        # After w1.T matmul (x @ w1.T):
        #   Col 0: x[0]*1 + x[1]*0.5 + x[2]*(-0.5) → min: -1-0.5-0.5=-2, max: 1+0.5+0.5=2 → [-2, 2]
        #   Col 1: x[0]*0 + x[1]*1 + x[2]*0 = x[1] → [-1, 1]
        #   Col 2: x[0]*(-0.5) + x[1]*0.5 + x[2]*1 → min: -0.5-0.5-1=-2, max: 0.5+0.5+1=2 → [-2, 2]
        #   Col 3: x[0]*0.5 + x[1]*0 + x[2]*0.5 → min: -0.5-0.5=-1, max: 0.5+0.5=1 → [-1, 1]
        #   Col 4: x[0]*0 + x[1]*(-0.5) + x[2]*0.5 → min: -0.5-0.5=-1, max: 0.5+0.5=1 → [-1, 1]
        # After ReLU: [0, 2], [0, 1], [0, 2], [0, 1], [0, 1]
        # w2.T rows: [1, 0, -1, 0.5, 0], [0, 1, 0, -0.5, 1]
        # After w2.T matmul (y @ w2.T):
        #   Col 0: y[0]*1 + y[1]*0 + y[2]*(-1) + y[3]*0.5 + y[4]*0
        #     min: 0+0+2*(-1)+0+0 = -2, max: 2+0+0+1*0.5+0 = 2.5 → [-2, 2.5]
        #   Col 1: y[0]*0 + y[1]*1 + y[2]*0 + y[3]*(-0.5) + y[4]*1
        #     min: 0+0+0+1*(-0.5)+0 = -0.5, max: 0+1+0+0+1 = 2 → [-0.5, 2]
        # After sigmoid: [sigmoid(-2), sigmoid(2.5)], [sigmoid(-0.5), sigmoid(2)]
        expected_lower = torch.sigmoid(torch.tensor([-2.0, -0.5]))
        expected_upper = torch.sigmoid(torch.tensor([2.5, 2.0]))
        # Repeat for all 10 batch elements
        expected_lower = expected_lower.unsqueeze(0).repeat(10, 1)
        expected_upper = expected_upper.unsqueeze(0).repeat(10, 1)
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)


class TestIBPWorkflowEdgeCases:
    """Test IBP workflow with edge cases and special scenarios."""

    def test_constant_propagation(self) -> None:
        """Test IBP with constant values in the computation graph."""

        def const_fn(x):
            # Constants should be handled properly
            c = torch.tensor([5.0, 10.0])
            return x + c

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(const_fn, (example_input,))

        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])

        # [0, 1] + [5, 10] = [5, 11], [10, 11]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([5.0, 10.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([6.0, 11.0]), atol=1e-5)

    def test_zero_width_interval(self) -> None:
        """Test IBP with zero-width (point) intervals."""

        def simple_fn(x):
            # Use a constant tensor instead of scalar
            return x * torch.tensor([2.0, 2.0])

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(simple_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Point interval: [3, 5] (no width)
        point = torch.tensor([3.0, 5.0])
        input_region = HyperRectangle(lower=point, upper=point)

        outputs = propagator.propagate([input_region])

        # [3, 3] * 2 = [6, 6] and [5, 5] * 2 = [10, 10]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([6.0, 10.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([6.0, 10.0]), atol=1e-5)

    def test_negative_intervals(self) -> None:
        """Test IBP with entirely negative intervals."""

        def neg_fn(x):
            return torch.relu(x)

        example_input = torch.randn(
            3,
        )
        graph = _trace_and_convert(neg_fn, (example_input,))

        propagator = IBPPropagator(graph)

        # Entirely negative input
        input_region = HyperRectangle(
            lower=torch.tensor([-5.0, -3.0, -1.0]),
            upper=torch.tensor([-2.0, -1.0, -0.5]),
        )

        outputs = propagator.propagate([input_region])

        # ReLU of negative interval is [0, 0]
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.zeros(3), atol=1e-5)
        assert torch.allclose(out.upper, torch.zeros(3), atol=1e-5)


class TestIBPWorkflowVerification:
    """Test that IBP bounds are sound (contain all possible outputs)."""

    def test_bounds_soundness_simple(self) -> None:
        """Verify that computed bounds contain all actual outputs for simple function."""

        def simple_fn(x):
            # Use constant tensors instead of scalars
            return x * torch.tensor([2.0, 2.0]) + torch.tensor([1.0, 1.0])

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(simple_fn, (example_input,))

        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 1.0]),
            upper=torch.tensor([1.0, 2.0]),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)

        # Sample points in the input region and verify they're in output bounds
        test_points = [
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.5, 1.5]),
            torch.tensor([0.25, 1.75]),
        ]

        for point in test_points:
            actual_output = simple_fn(point)
            assert torch.all(actual_output >= out.lower - 1e-5)
            assert torch.all(actual_output <= out.upper + 1e-5)

    def test_bounds_soundness_relu_network(self) -> None:
        """Verify bounds soundness for a ReLU network."""

        def relu_network_fn(x):
            # x: (2,)
            w = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
            return torch.relu(x @ w.T)

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(relu_network_fn, (example_input,))

        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)

        # Test corner points
        corner_points = [
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 1.0]),
        ]

        for point in corner_points:
            actual_output = relu_network_fn(point)
            assert torch.all(actual_output >= out.lower - 1e-5)
            assert torch.all(actual_output <= out.upper + 1e-5)


class TestIBPAllOperations:
    """Comprehensive tests for all operations in the IBP registry."""

    def test_trigonometric_operations(self) -> None:
        """Test sin, cos in combination."""

        c1 = torch.tensor([2.0, 3.0, 1.5])

        def trig_fn(x):
            # Non-trivial combination of trig functions with constants
            x1 = torch.sin(x)  # sin([-1,1]) = [-0.84, 0.84]
            x2 = torch.cos(x)  # cos([-1,1]) = [0.54, 1.0]
            # Add them together then multiply by constant
            return (x1 + x2) * c1

        example_input = torch.randn(
            3,
        )
        graph = _trace_and_convert(trig_fn, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        # sin([-1,1]) = [-sin(1), sin(1)] = [-0.8414709848, 0.8414709848]
        # cos([-1,1]) = [cos(1), cos(0)] = [0.5403023059, 1.0]
        # sum: [-0.8414709848 + 0.5403023059, 0.8414709848 + 1.0]
        #    = [-0.3011686789, 1.8414709848]
        # Element 0: [-0.3011686789, 1.8414709848] * 2.0 = [-0.6023373578, 3.6829419696]
        # Element 1: [-0.3011686789, 1.8414709848] * 3.0 = [-0.9035060367, 5.5244129544]
        # Element 2: [-0.3011686789, 1.8414709848] * 1.5 = [-0.4517530184, 2.7622064772]
        sin_1 = torch.sin(torch.tensor(1.0)).item()
        cos_1 = torch.cos(torch.tensor(1.0)).item()
        sum_lower = -sin_1 + cos_1
        sum_upper = sin_1 + 1.0
        expected_lower = torch.tensor([sum_lower * 2.0, sum_lower * 3.0, sum_lower * 1.5])
        expected_upper = torch.tensor([sum_upper * 2.0, sum_upper * 3.0, sum_upper * 1.5])
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)

    def test_neg_abs_operations(self) -> None:
        """Test NEG and ABS operations."""

        def neg_abs_fn(x):
            # Negate, then take absolute value, should give |(-x)| = |x|
            x_neg = torch.neg(x)
            return torch.abs(x_neg)

        example_input = torch.randn(
            3,
        )
        graph = _trace_and_convert(neg_abs_fn, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.5]),
            upper=torch.tensor([1.0, 2.0, 3.0]),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        # For [-2, 1]: |neg([-2,1])| = |[-1,2]| = [0, 2]
        # For [-1, 2]: |neg([-1,2])| = |[-2,1]| = [0, 2]
        # For [0.5, 3]: |neg([0.5,3])| = |[-3,-0.5]| = [0.5, 3]
        assert torch.allclose(out.lower, torch.tensor([0.0, 0.0, 0.5]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([2.0, 2.0, 3.0]), atol=1e-5)

    def test_sqrt_reciprocal_operations(self) -> None:
        """Test SQRT and RECIPROCAL operations."""

        def sqrt_recip_fn(x):
            # sqrt then reciprocal: 1/sqrt(x) for positive x
            x_sqrt = torch.sqrt(x)
            return torch.reciprocal(x_sqrt)

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(sqrt_recip_fn, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 4.0]),
            upper=torch.tensor([4.0, 9.0]),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        # For [1, 4]: sqrt -> [1, 2], reciprocal -> [0.5, 1]
        # For [4, 9]: sqrt -> [2, 3], reciprocal -> [1/3, 0.5]
        assert torch.allclose(out.lower, torch.tensor([0.5, 1.0 / 3.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([1.0, 0.5]), atol=1e-5)

    def test_maximum_minimum_operations(self) -> None:
        """Test MAXIMUM and MINIMUM operations."""

        c1 = torch.tensor([0.0, 2.0])
        c2 = torch.tensor([1.0, -1.0])

        def max_min_fn(x):
            # Clamp value between min and max
            x_max = torch.maximum(x, c1)  # At least 0 or 2
            x_min = torch.minimum(x_max, c2)  # At most 1 or -1
            return x_min

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(max_min_fn, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0]),
            upper=torch.tensor([3.0, 3.0]),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        # Element 0: max([-2,3], 0) = [0,3], then min([0,3], 1) = [0,1]
        # Element 1: max([-2,3], 2) = [2,3], then min([2,3], -1) = [-1,-1]
        assert torch.allclose(out.lower, torch.tensor([0.0, -1.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([1.0, -1.0]), atol=1e-5)

    def test_clamp_operation(self) -> None:
        """Test CLAMP operation."""

        def clamp_fn(x):
            # Clamp between -0.5 and 1.5
            return torch.clamp(x, min=-0.5, max=1.5)

        example_input = torch.randn(
            3,
        )
        graph = _trace_and_convert(clamp_fn, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, 0.0, 1.0]),
            upper=torch.tensor([0.0, 1.0, 3.0]),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        # Element 0: clamp([-2, 0], -0.5, 1.5) = [-0.5, 0]
        # Element 1: clamp([0, 1], -0.5, 1.5) = [0, 1]
        # Element 2: clamp([1, 3], -0.5, 1.5) = [1, 1.5]
        assert torch.allclose(out.lower, torch.tensor([-0.5, 0.0, 1.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([0.0, 1.0, 1.5]), atol=1e-5)

    def test_reduction_operations(self) -> None:
        """Test SUM and MEAN reductions."""

        def reduction_fn(x):
            # x: (2, 3)
            x_sum = torch.sum(x, dim=1)  # -> (2,)
            x_mean = torch.mean(x, dim=1)  # -> (2,)
            # Concatenate them instead of stack to avoid issues
            return x_sum + x_mean  # -> (2,)

        example_input = torch.randn(2, 3)
        graph = _trace_and_convert(reduction_fn, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.zeros(2, 3),
            upper=torch.ones(2, 3),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (2,)
        assert out.upper.shape == (2,)
        # Sum of [0,1]x3 elements: [0, 3]
        # Mean of [0,1]x3 elements: [0, 1]
        # Sum + Mean: [0, 4]
        assert torch.allclose(out.lower, torch.tensor([0.0, 0.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([4.0, 4.0]), atol=1e-5)

    def test_reshaping_operations(self) -> None:
        """Test FLATTEN, UNSQUEEZE, SQUEEZE."""

        def reshape_fn(x):
            # x: (2, 3)
            x = torch.unsqueeze(x, dim=0)  # -> (1, 2, 3)
            x = torch.flatten(x, start_dim=1)  # -> (1, 6)
            x = torch.squeeze(x, dim=0)  # -> (6,)
            return x

        example_input = torch.randn(2, 3)
        graph = _trace_and_convert(reshape_fn, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.zeros(2, 3),
            upper=torch.ones(2, 3),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (6,)
        assert out.upper.shape == (6,)
        # Should preserve bounds through reshaping
        assert torch.allclose(out.lower, torch.zeros(6), atol=1e-5)
        assert torch.allclose(out.upper, torch.ones(6), atol=1e-5)

    def test_select_operation(self) -> None:
        """Test SELECT operation to extract slice."""

        def select_fn(x):
            # x: (3, 4)
            # Select specific dimension
            return torch.select(x, dim=0, index=1)  # -> (4,)

        example_input = torch.randn(3, 4)
        graph = _trace_and_convert(select_fn, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]),
            upper=torch.tensor([[0.5, 1.5, 2.5, 3.5], [4.5, 5.5, 6.5, 7.5], [8.5, 9.5, 10.5, 11.5]]),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (4,)
        assert out.upper.shape == (4,)
        # Select row 1 (index=1)
        expected_lower = torch.tensor([4.0, 5.0, 6.0, 7.0])
        expected_upper = torch.tensor([4.5, 5.5, 6.5, 7.5])
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)

    def test_reduction_amax_amin(self) -> None:
        """Test amax and amin operations (instead of max/min which return tuples)."""

        def reduction_fn(x):
            # x: (2, 3)
            # Use amax/amin for reduction without tuple returns
            x_max = torch.amax(x, dim=1)  # -> (2,)
            x_min = torch.amin(x, dim=1)  # -> (2,)
            return x_max + x_min  # -> (2,)

        example_input = torch.randn(2, 3)
        graph = _trace_and_convert(reduction_fn, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            upper=torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (2,)
        assert out.upper.shape == (2,)
        # Row 0: max([1-1.5, 2-2.5, 3-3.5]) = [3, 3.5], min = [1, 1.5], sum = [4, 5]
        # Row 1: max([4-4.5, 5-5.5, 6-6.5]) = [6, 6.5], min = [4, 4.5], sum = [10, 11]
        assert torch.allclose(out.lower, torch.tensor([4.0, 10.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([5.0, 11.0]), atol=1e-5)

    def test_unsqueeze_squeeze_chain(self) -> None:
        """Test UNSQUEEZE and SQUEEZE in combination."""

        def squeeze_fn(x):
            # x: (3, 1, 4)
            # Squeeze middle dimension
            x = torch.squeeze(x, dim=1)  # -> (3, 4)
            # Add dimension back at different position
            x = torch.unsqueeze(x, dim=0)  # -> (1, 3, 4)
            return x

        example_input = torch.randn(3, 1, 4)
        graph = _trace_and_convert(squeeze_fn, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.zeros(3, 1, 4),
            upper=torch.ones(3, 1, 4),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (1, 3, 4)
        assert out.upper.shape == (1, 3, 4)
        # Should preserve bounds through squeeze/unsqueeze
        assert torch.allclose(out.lower, torch.zeros(1, 3, 4), atol=1e-5)
        assert torch.allclose(out.upper, torch.ones(1, 3, 4), atol=1e-5)

    def test_complex_network_all_ops(self) -> None:
        """Test a complex network using many different operations."""

        w1 = torch.tensor([[1.0, 0.5, -0.5], [0.0, 1.0, 0.0], [-0.5, 0.5, 1.0]])
        c1 = torch.tensor([0.1, 0.2, 0.3])
        c2 = torch.tensor([2.0, 1.5, 1.0])
        c_add = torch.tensor(1e-6)

        def complex_network(x):
            # x: (3,)

            # Phase 1: Linear transformation and activations
            x = x @ w1.T  # (3,) - matmul
            x = torch.relu(x)  # relu
            x = torch.add(x, c1)  # add constant (use torch.add explicitly)
            x = torch.sigmoid(x)  # sigmoid

            # Phase 2: Element-wise operations with constants
            x = torch.add(x, c_add)  # add small constant for stability
            x = torch.sqrt(x)  # sqrt
            x = torch.mul(x, c2)  # mul with constant (use torch.mul explicitly)

            # Phase 3: Final transformation
            x = torch.tanh(x)  # tanh

            return x

        example_input = torch.randn(
            3,
        )
        graph = _trace_and_convert(complex_network, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.zeros(3),
            upper=torch.ones(3),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (3,)
        assert out.upper.shape == (3,)

        # Calculate exact bounds step by step:
        # Input: [0, 0, 0] to [1, 1, 1]
        # w1.T = [[1.0, 0.0, -0.5], [0.5, 1.0, 0.5], [-0.5, 0.0, 1.0]]
        # After matmul with w1.T:
        #   Element 0: x[0]*1 + x[1]*0.5 + x[2]*(-0.5)
        #     Min: 0*1 + 0*0.5 + 1*(-0.5) = -0.5, Max: 1*1 + 1*0.5 + 0*(-0.5) = 1.5
        #   Element 1: x[0]*0 + x[1]*1 + x[2]*0 = x[1]
        #     Min: 0, Max: 1
        #   Element 2: x[0]*(-0.5) + x[1]*0.5 + x[2]*1
        #     Min: 1*(-0.5) + 0*0.5 + 0*1 = -0.5, Max: 0*(-0.5) + 1*0.5 + 1*1 = 1.5
        # After ReLU: [0, 1.5], [0, 1], [0, 1.5]
        # After add c1: [0.1, 1.6], [0.2, 1.2], [0.3, 1.8]
        # After sigmoid, sqrt, mul, tanh...

        # Compute precisely:
        lower_after_matmul = torch.tensor([0.0, 0.0, 0.0])
        upper_after_matmul = torch.tensor([1.5, 1.0, 1.5])
        # After ReLU (no change, already positive)
        # After add c1
        lower_after_add = lower_after_matmul + c1  # [0.1, 0.2, 0.3]
        upper_after_add = upper_after_matmul + c1  # [1.6, 1.2, 1.8]
        # After sigmoid
        lower_after_sigmoid = torch.sigmoid(lower_after_add)
        upper_after_sigmoid = torch.sigmoid(upper_after_add)
        # After add 1e-6
        lower_after_add2 = lower_after_sigmoid + c_add
        upper_after_add2 = upper_after_sigmoid + c_add
        # After sqrt
        lower_after_sqrt = torch.sqrt(lower_after_add2)
        upper_after_sqrt = torch.sqrt(upper_after_add2)
        # After mul c2
        lower_after_mul = lower_after_sqrt * c2
        upper_after_mul = upper_after_sqrt * c2
        # After tanh
        expected_lower = torch.tanh(lower_after_mul)
        expected_upper = torch.tanh(upper_after_mul)

        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)

    def test_arithmetic_chain(self) -> None:
        """Test chaining of all arithmetic operations."""

        c1 = torch.tensor([2.0, 3.0])
        c2 = torch.tensor([0.5, 0.25])
        c3 = torch.tensor([2.0, -1.0])
        c4 = torch.tensor([4.0, 2.0])

        def arithmetic_chain(x):
            # Chain: add (x+c) -> add (c+x) -> mul -> div
            x = x + c1  # add: x + constant
            x = c2 + x  # add: constant + x
            x = x * c3  # mul
            x = x / c4  # div
            return x

        example_input = torch.randn(
            2,
        )
        graph = _trace_and_convert(arithmetic_chain, (example_input,))
        propagator = IBPPropagator(graph)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 1.0]),
            upper=torch.tensor([1.0, 2.0]),
        )

        outputs = propagator.propagate([input_region])
        out = outputs[0]
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (2,)
        assert out.upper.shape == (2,)

        # Manual calculation:
        # Element 0: [0,1] + 2 = [2,3], + 0.5 = [2.5,3.5], * 2 = [5,7], / 4 = [1.25, 1.75]
        # Element 1: [1,2] + 3 = [4,5], + 0.25 = [4.25,5.25], * -1 = [-5.25,-4.25], / 2 = [-2.625, -2.125]
        assert torch.allclose(out.lower, torch.tensor([1.25, -2.625]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([1.75, -2.125]), atol=1e-5)
