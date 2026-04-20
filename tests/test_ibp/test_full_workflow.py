"""
End-to-end IBP workflow tests.

Tests the complete IBP pipeline:
1. Tracing PyTorch functions/modules with BoundPropagationTracer
2. Annotating metadata with MetadataPass
3. Propagating bounds with IBPPropagator
4. Verifying correctness of output bounds
"""

from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import IBPPropagator
from bound_propagation.propagation.ibp import create_default_ibp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def _trace_and_annotate(fn_or_module, example_inputs: tuple[torch.Tensor, ...]):
    """Trace, annotate metadata, and return GraphModule."""
    registry = create_default_ibp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn_or_module)
    MetadataPass(gm).run(*example_inputs)
    return gm


class TestIBPWorkflowSimpleFunctions:
    """Test IBP workflow with simple mathematical functions."""

    def test_single_relu(self) -> None:
        def relu_fn(x):
            return torch.relu(x)

        gm = _trace_and_annotate(relu_fn, (torch.randn(3),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0, -2.0]),
            upper=torch.tensor([3.0, 3.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(out.upper, torch.tensor([3.0, 3.0, 3.0]))

    def test_add_and_relu(self) -> None:
        def add_relu_fn(x):
            y = x + torch.tensor([1.0, -1.0, 0.5])
            return torch.relu(y)

        gm = _trace_and_annotate(add_relu_fn, (torch.randn(3),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([1.0, 0.0, 0.5]))
        assert torch.allclose(out.upper, torch.tensor([2.0, 0.0, 1.5]))

    def test_mul_operation(self) -> None:
        def mul_fn(x):
            return x * torch.tensor([2.0, -1.0, 3.0])

        gm = _trace_and_annotate(mul_fn, (torch.randn(3),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([2.0, 2.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([-2.0, -2.0, -3.0]))
        assert torch.allclose(out.upper, torch.tensor([4.0, 1.0, 6.0]))

    def test_exp_operation(self) -> None:
        def exp_fn(x):
            return torch.exp(x)

        gm = _trace_and_annotate(exp_fn, (torch.randn(2),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([1.0, 1.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.exp(torch.tensor([1.0, 1.0])), atol=1e-5)

    def test_sigmoid_operation(self) -> None:
        def sigmoid_fn(x):
            return torch.sigmoid(x)

        gm = _trace_and_annotate(sigmoid_fn, (torch.randn(2),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        expected_lower = torch.sigmoid(torch.tensor([-1.0, -1.0]))
        expected_upper = torch.sigmoid(torch.tensor([1.0, 1.0]))
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)


class TestIBPWorkflowNeuralNetworks:
    """Test IBP workflow with neural network-like operations."""

    def test_matmul_with_bias(self) -> None:
        def matmul_with_bias_fn(x):
            weight_T = torch.tensor([[1.0, -1.0, 0.5], [2.0, 0.0, -2.0]])
            bias = torch.tensor([0.5, -0.5])
            return x @ weight_T.T + bias

        gm = _trace_and_annotate(matmul_with_bias_fn, (torch.randn(3),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([-2.0, -4.5]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([3.0, 3.5]), atol=1e-5)

    def test_two_layer_network(self) -> None:
        def two_layer_fn(x):
            w1 = torch.eye(3, 4)
            x = x @ w1.T
            x = torch.relu(x)
            w2 = torch.eye(2, 3)
            x = x @ w2.T
            return x

        gm = _trace_and_annotate(two_layer_fn, (torch.randn(4),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(lower=torch.zeros(4), upper=torch.ones(4))
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (2,)
        assert torch.allclose(out.lower, torch.zeros(2), atol=1e-5)
        assert torch.allclose(out.upper, torch.ones(2), atol=1e-5)


class TestIBPWorkflowComplexOperations:
    """Test IBP workflow with more complex operations."""

    def test_div_operation(self) -> None:
        def div_fn(x):
            return x / torch.tensor([2.0, 4.0])

        gm = _trace_and_annotate(div_fn, (torch.randn(2),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([2.0, 4.0]),
            upper=torch.tensor([8.0, 16.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([1.0, 1.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([4.0, 4.0]), atol=1e-5)

    def test_sub_operation(self) -> None:
        def sub_fn(x, y):
            return x - y

        gm = _trace_and_annotate(sub_fn, (torch.randn(3), torch.randn(3)))
        propagator = IBPPropagator(gm)

        outputs = propagator.propagate(
            [
                HyperRectangle(
                    lower=torch.tensor([5.0, 5.0, 5.0]),
                    upper=torch.tensor([10.0, 10.0, 10.0]),
                ),
                HyperRectangle(
                    lower=torch.tensor([1.0, 2.0, 3.0]),
                    upper=torch.tensor([1.0, 2.0, 3.0]),
                ),
            ]
        )

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([4.0, 3.0, 2.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([9.0, 8.0, 7.0]), atol=1e-5)

    def test_chained_operations(self) -> None:
        def complex_fn(x):
            x = x + torch.tensor([1.0, 2.0])
            x = x * torch.tensor([2.0, 0.5])
            x = torch.relu(x)
            x = torch.sigmoid(x)
            return x

        gm = _trace_and_annotate(complex_fn, (torch.randn(2),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        expected_lower = torch.sigmoid(torch.tensor([2.0, 1.0]))
        expected_upper = torch.sigmoid(torch.tensor([4.0, 1.5]))
        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)


class TestIBPWorkflowEdgeCases:
    """Test IBP workflow with edge cases."""

    def test_constant_propagation(self) -> None:
        def const_fn(x):
            c = torch.tensor([5.0, 10.0])
            return x + c

        gm = _trace_and_annotate(const_fn, (torch.randn(2),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([5.0, 10.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([6.0, 11.0]), atol=1e-5)

    def test_zero_width_interval(self) -> None:
        def simple_fn(x):
            return x * torch.tensor([2.0, 2.0])

        gm = _trace_and_annotate(simple_fn, (torch.randn(2),))
        propagator = IBPPropagator(gm)

        point = torch.tensor([3.0, 5.0])
        input_region = HyperRectangle(lower=point, upper=point)
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([6.0, 10.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([6.0, 10.0]), atol=1e-5)

    def test_negative_intervals(self) -> None:
        def neg_fn(x):
            return torch.relu(x)

        gm = _trace_and_annotate(neg_fn, (torch.randn(3),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-5.0, -3.0, -1.0]),
            upper=torch.tensor([-2.0, -1.0, -0.5]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.zeros(3), atol=1e-5)
        assert torch.allclose(out.upper, torch.zeros(3), atol=1e-5)


class TestIBPWorkflowVerification:
    """Test that IBP bounds are sound (contain all possible outputs)."""

    def test_bounds_soundness_simple(self) -> None:
        def simple_fn(x):
            return x * torch.tensor([2.0, 2.0]) + torch.tensor([1.0, 1.0])

        gm = _trace_and_annotate(simple_fn, (torch.randn(2),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 1.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)

        test_points = [
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.5, 1.5]),
        ]
        for point in test_points:
            actual_output = simple_fn(point)
            assert torch.all(actual_output >= out.lower - 1e-5)
            assert torch.all(actual_output <= out.upper + 1e-5)

    def test_bounds_soundness_relu_network(self) -> None:
        def relu_network_fn(x):
            w = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
            return torch.relu(x @ w.T)

        gm = _trace_and_annotate(relu_network_fn, (torch.randn(2),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)

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
    """Comprehensive tests for various IBP-supported operations."""

    def test_trigonometric_operations(self) -> None:
        c1 = torch.tensor([2.0, 3.0, 1.5])

        def trig_fn(x):
            x1 = torch.sin(x)
            x2 = torch.cos(x)
            return (x1 + x2) * c1

        gm = _trace_and_annotate(trig_fn, (torch.randn(3),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        sin_1 = torch.sin(torch.tensor(1.0)).item()
        cos_1 = torch.cos(torch.tensor(1.0)).item()
        sum_lower = -sin_1 + cos_1
        sum_upper = sin_1 + 1.0
        expected_lower = torch.tensor([sum_lower * 2.0, sum_lower * 3.0, sum_lower * 1.5])
        expected_upper = torch.tensor([sum_upper * 2.0, sum_upper * 3.0, sum_upper * 1.5])
        assert torch.allclose(out.lower, expected_lower, atol=1e-5)
        assert torch.allclose(out.upper, expected_upper, atol=1e-5)

    def test_neg_abs_operations(self) -> None:
        def neg_abs_fn(x):
            x_neg = torch.neg(x)
            return torch.abs(x_neg)

        gm = _trace_and_annotate(neg_abs_fn, (torch.randn(3),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.5]),
            upper=torch.tensor([1.0, 2.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([0.0, 0.0, 0.5]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([2.0, 2.0, 3.0]), atol=1e-5)

    def test_sqrt_reciprocal_operations(self) -> None:
        def sqrt_recip_fn(x):
            x_sqrt = torch.sqrt(x)
            return torch.reciprocal(x_sqrt)

        gm = _trace_and_annotate(sqrt_recip_fn, (torch.randn(2).abs() + 1.0,))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 4.0]),
            upper=torch.tensor([4.0, 9.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([0.5, 1.0 / 3.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([1.0, 0.5]), atol=1e-5)

    def test_clamp_operation(self) -> None:
        def clamp_fn(x):
            return torch.clamp(x, min=-0.5, max=1.5)

        gm = _trace_and_annotate(clamp_fn, (torch.randn(3),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, 0.0, 1.0]),
            upper=torch.tensor([0.0, 1.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([-0.5, 0.0, 1.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([0.0, 1.0, 1.5]), atol=1e-5)

    def test_reduction_operations(self) -> None:
        def reduction_fn(x):
            x_sum = torch.sum(x, dim=1)
            x_mean = torch.mean(x, dim=1)
            return x_sum + x_mean

        gm = _trace_and_annotate(reduction_fn, (torch.randn(2, 3),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.zeros(2, 3),
            upper=torch.ones(2, 3),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert out.lower.shape == (2,)
        assert torch.allclose(out.lower, torch.tensor([0.0, 0.0]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([4.0, 4.0]), atol=1e-5)

    def test_arithmetic_chain(self) -> None:
        c1 = torch.tensor([2.0, 3.0])
        c2 = torch.tensor([0.5, 0.25])
        c3 = torch.tensor([2.0, -1.0])
        c4 = torch.tensor([4.0, 2.0])

        def arithmetic_chain(x):
            x = x + c1
            x = c2 + x
            x = x * c3
            x = x / c4
            return x

        gm = _trace_and_annotate(arithmetic_chain, (torch.randn(2),))
        propagator = IBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 1.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, IntervalBounds)
        assert torch.allclose(out.lower, torch.tensor([1.25, -2.625]), atol=1e-5)
        assert torch.allclose(out.upper, torch.tensor([1.75, -2.125]), atol=1e-5)
