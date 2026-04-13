"""
End-to-end Forward LBP workflow tests.

Tests the complete Forward LBP pipeline:
1. Tracing PyTorch functions/modules
2. Converting to IR Graph
3. Constructing Forward LBP bounding strategies
4. Propagating bounds through the graph
5. Verifying correctness of output bounds

These tests ensure the full workflow works for various operations and network architectures.
"""

from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.propagation.methods import ForwardLBPPropagator
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


class TestForwardLBPWorkflowSimpleFunctions:
    """Test Forward LBP workflow with simple mathematical functions."""

    def test_single_relu(self) -> None:
        """Test Forward LBP on a simple ReLU activation."""

        def relu_fn(x):
            return torch.relu(x)

        # Trace and convert
        example_input = torch.randn(3)
        graph = _trace_and_convert(relu_fn, (example_input,))

        # Create propagator
        propagator = ForwardLBPPropagator(graph)

        # Define input region: [-2, 3] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0, -2.0]),
            upper=torch.tensor([3.0, 3.0, 3.0]),
        )

        # Propagate bounds
        outputs = propagator.propagate([input_region])

        # ReLU([-2, 3]) = [0, 3]
        assert len(outputs) == 1
        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(upper, torch.tensor([3.0, 3.0, 3.0]))

    def test_add_with_constant(self) -> None:
        """Test Forward LBP on addition with a constant."""

        def add_const_fn(x):
            return x + torch.tensor([1.0, -1.0, 0.5])

        example_input = torch.randn(3)
        graph = _trace_and_convert(add_const_fn, (example_input,))

        propagator = ForwardLBPPropagator(graph)

        # Input: [0, 1] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])

        # x + [1, -1, 0.5]: [0, 1] + 1 = [1, 2], [0, 1] + (-1) = [-1, 0], [0, 1] + 0.5 = [0.5, 1.5]
        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([1.0, -1.0, 0.5]))
        assert torch.allclose(upper, torch.tensor([2.0, 0.0, 1.5]))

    def test_add_and_relu(self) -> None:
        """Test Forward LBP on addition followed by ReLU."""

        def add_relu_fn(x):
            y = x + torch.tensor([1.0, -1.0, 0.5])
            return torch.relu(y)

        example_input = torch.randn(3)
        graph = _trace_and_convert(add_relu_fn, (example_input,))

        propagator = ForwardLBPPropagator(graph)

        # Input: [0, 1] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )

        outputs = propagator.propagate([input_region])

        # After add: [1, 2], [-1, 0], [0.5, 1.5]
        # After ReLU: [1, 2], [0, 0], [0.5, 1.5]
        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([1.0, 0.0, 0.5]))
        assert torch.allclose(upper, torch.tensor([2.0, 0.0, 1.5]))

    def test_mul_with_constant(self) -> None:
        """Test Forward LBP on element-wise multiplication with constant."""

        def mul_const_fn(x):
            return x * torch.tensor([2.0, -1.0, 0.5])

        example_input = torch.randn(3)
        graph = _trace_and_convert(mul_const_fn, (example_input,))

        propagator = ForwardLBPPropagator(graph)

        # Input: [1, 3] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 1.0, 1.0]),
            upper=torch.tensor([3.0, 3.0, 3.0]),
        )

        outputs = propagator.propagate([input_region])

        # x * [2, -1, 0.5]: [1, 3] * 2 = [2, 6], [1, 3] * (-1) = [-3, -1], [1, 3] * 0.5 = [0.5, 1.5]
        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([2.0, -3.0, 0.5]))
        assert torch.allclose(upper, torch.tensor([6.0, -1.0, 1.5]))

    def test_linear_layer(self) -> None:
        """Test Forward LBP on a simple linear layer (matmul + add)."""

        def linear_fn(x):
            # x: (batch, 3)
            # Weight: (3, 2)
            weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            bias = torch.tensor([1.0, -1.0])
            return x @ weight + bias

        example_input = torch.randn(1, 3)
        graph = _trace_and_convert(linear_fn, (example_input,))

        propagator = ForwardLBPPropagator(graph)

        # Input: x ∈ [0, 1] for each of 3 dimensions
        input_region = HyperRectangle(
            lower=torch.tensor([[0.0, 0.0, 0.0]]),
            upper=torch.tensor([[1.0, 1.0, 1.0]]),
        )

        outputs = propagator.propagate([input_region])

        # x @ W: [0, 1]^3 @ [[1, 2], [3, 4], [5, 6]]
        # Output dim 0: [0, 1] * 1 + [0, 1] * 3 + [0, 1] * 5 = [0, 9]
        # Output dim 1: [0, 1] * 2 + [0, 1] * 4 + [0, 1] * 6 = [0, 12]
        # After adding bias: [0, 9] + 1 = [1, 10], [0, 12] + (-1) = [-1, 11]
        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([[1.0, -1.0]]))
        assert torch.allclose(upper, torch.tensor([[10.0, 11.0]]))

    def test_two_layer_network(self) -> None:
        """Test Forward LBP on a two-layer network with ReLU."""

        def two_layer_fn(x):
            # Layer 1: 2 -> 3
            w1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            h = torch.relu(x @ w1)
            # Layer 2: 3 -> 1
            w2 = torch.tensor([[1.0], [1.0], [1.0]])
            return h @ w2

        example_input = torch.randn(1, 2)
        graph = _trace_and_convert(two_layer_fn, (example_input,))

        propagator = ForwardLBPPropagator(graph)

        # Input: x ∈ [0, 1] for each of 2 dimensions
        input_region = HyperRectangle(
            lower=torch.tensor([[0.0, 0.0]]),
            upper=torch.tensor([[1.0, 1.0]]),
        )

        outputs = propagator.propagate([input_region])

        # Layer 1: x @ w1
        # Dim 0: [0, 1] * 1 + [0, 1] * 4 = [0, 5]
        # Dim 1: [0, 1] * 2 + [0, 1] * 5 = [0, 7]
        # Dim 2: [0, 1] * 3 + [0, 1] * 6 = [0, 9]
        # After ReLU: [0, 5], [0, 7], [0, 9]
        # Layer 2: [0, 5] + [0, 7] + [0, 9] = [0, 21]
        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([[0.0]]))
        assert torch.allclose(upper, torch.tensor([[21.0]]))

    def test_sqrt_operation(self) -> None:
        """Test Forward LBP on sqrt with linear relaxation."""

        def sqrt_fn(x):
            return torch.sqrt(x)

        example_input = torch.randn(2)
        graph = _trace_and_convert(sqrt_fn, (example_input,))

        propagator = ForwardLBPPropagator(graph)

        # Input: x ∈ [1, 4] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 1.0]),
            upper=torch.tensor([4.0, 4.0]),
        )

        outputs = propagator.propagate([input_region])

        # sqrt([1, 4]) = [1, 2]
        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        # Due to relaxation, bounds should contain [1, 2] (may be conservative)
        assert torch.all(lower <= 1.01)
        assert torch.all(upper >= 1.99)


class TestForwardLBPWorkflowModules:
    """Test Forward LBP workflow with PyTorch modules."""

    def test_simple_mlp(self) -> None:
        """Test Forward LBP on a simple MLP module."""

        class SimpleMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 3, bias=False)
                self.fc2 = torch.nn.Linear(3, 1, bias=False)

                # Set specific weights for testing
                self.fc1.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                self.fc2.weight.data = torch.tensor([[1.0, 1.0, 1.0]])

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleMLP()
        example_input = torch.randn(1, 2)
        graph = _trace_and_convert(model, (example_input,))

        propagator = ForwardLBPPropagator(graph)

        # Input: x ∈ [0, 1] for each dimension
        input_region = HyperRectangle(
            lower=torch.tensor([[0.0, 0.0]]),
            upper=torch.tensor([[1.0, 1.0]]),
        )

        outputs = propagator.propagate([input_region])

        # FC1: similar to previous two-layer test
        # Output should be in [0, 21]
        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.all(lower >= -0.01)
        assert torch.all(upper <= 21.01)
