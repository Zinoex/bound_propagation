"""
End-to-end Forward LBP workflow tests.

Tests the complete Forward LBP pipeline:
1. Tracing PyTorch functions/modules with BoundPropagationTracer
2. Annotating metadata with MetadataPass
3. Propagating bounds with ForwardLBPPropagator
4. Verifying correctness of output bounds
"""

from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import ForwardLBPPropagator
from bound_propagation.propagation.forward_lbp import create_default_forward_lbp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def _trace_and_annotate(fn_or_module, example_inputs: tuple[torch.Tensor, ...]):
    """Trace, annotate metadata, and return GraphModule."""
    registry = create_default_forward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn_or_module)
    MetadataPass(gm).run(*example_inputs)
    return gm


class TestForwardLBPWorkflowSimpleFunctions:
    """Test Forward LBP workflow with simple mathematical functions."""

    def test_single_relu(self) -> None:
        def relu_fn(x):
            return torch.relu(x)

        gm = _trace_and_annotate(relu_fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0, -2.0]),
            upper=torch.tensor([3.0, 3.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        # Forward LBP ReLU uses the non-adaptive crossing relaxation where
        # alpha_lower = alpha_upper = u / (u - l). For l=-2, u=3 this gives 0.6,
        # so the concretized lower bound is 0.6 * (-2) = -1.2 (sound but not tight).
        assert torch.allclose(lower, torch.tensor([-1.2, -1.2, -1.2]))
        assert torch.allclose(upper, torch.tensor([3.0, 3.0, 3.0]))

    def test_add_with_constant(self) -> None:
        def add_const_fn(x):
            return x + torch.tensor([1.0, -1.0, 0.5])

        gm = _trace_and_annotate(add_const_fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([1.0, -1.0, 0.5]))
        assert torch.allclose(upper, torch.tensor([2.0, 0.0, 1.5]))

    def test_add_and_relu(self) -> None:
        def add_relu_fn(x):
            y = x + torch.tensor([1.0, -1.0, 0.5])
            return torch.relu(y)

        gm = _trace_and_annotate(add_relu_fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([1.0, 0.0, 0.5]))
        assert torch.allclose(upper, torch.tensor([2.0, 0.0, 1.5]))

    def test_mul_with_constant(self) -> None:
        def mul_const_fn(x):
            return x * torch.tensor([2.0, -1.0, 0.5])

        gm = _trace_and_annotate(mul_const_fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 1.0, 1.0]),
            upper=torch.tensor([3.0, 3.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([2.0, -3.0, 0.5]))
        assert torch.allclose(upper, torch.tensor([6.0, -1.0, 1.5]))

    def test_linear_layer(self) -> None:
        def linear_fn(x):
            weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            bias = torch.tensor([1.0, -1.0])
            return x @ weight + bias

        gm = _trace_and_annotate(linear_fn, (torch.randn(1, 3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([[0.0, 0.0, 0.0]]),
            upper=torch.tensor([[1.0, 1.0, 1.0]]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([[1.0, -1.0]]))
        assert torch.allclose(upper, torch.tensor([[10.0, 11.0]]))

    def test_two_layer_network(self) -> None:
        def two_layer_fn(x):
            w1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            h = torch.relu(x @ w1)
            w2 = torch.tensor([[1.0], [1.0], [1.0]])
            return h @ w2

        gm = _trace_and_annotate(two_layer_fn, (torch.randn(1, 2),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([[0.0, 0.0]]),
            upper=torch.tensor([[1.0, 1.0]]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([[0.0]]))
        assert torch.allclose(upper, torch.tensor([[21.0]]))

    def test_sqrt_operation(self) -> None:
        def sqrt_fn(x):
            return torch.sqrt(x)

        gm = _trace_and_annotate(sqrt_fn, (torch.randn(2).abs() + 1.0,))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 1.0]),
            upper=torch.tensor([4.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.all(lower <= 1.01)
        assert torch.all(upper >= 1.99)

    def test_constant_division_with_mixed_denominator_regimes(self) -> None:
        def const_div_fn(x):
            return torch.tensor([6.0, -6.0]) / x

        gm = _trace_and_annotate(const_div_fn, (torch.randn(2),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, 2.0]),
            upper=torch.tensor([1.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()

        assert torch.isneginf(lower[0])
        assert torch.isposinf(upper[0])
        assert lower[1].item() <= -3.0 + 1e-6
        assert upper[1].item() >= -1.5 - 1e-6


class TestForwardLBPWorkflowModules:
    """Test Forward LBP workflow with PyTorch modules."""

    def test_simple_mlp(self) -> None:
        class SimpleMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(2, 3, bias=False)
                self.fc2 = torch.nn.Linear(3, 1, bias=False)
                self.fc1.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                self.fc2.weight.data = torch.tensor([[1.0, 1.0, 1.0]])

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleMLP()
        gm = _trace_and_annotate(model, (torch.randn(1, 2),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([[0.0, 0.0]]),
            upper=torch.tensor([[1.0, 1.0]]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs[0]
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.all(lower >= -0.01)
        assert torch.all(upper <= 21.01)
