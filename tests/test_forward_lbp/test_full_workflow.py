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


def _evaluate_linear_bounds_at(linear_bounds, x):
    """Evaluate affine lower/upper bounds at a concrete point x."""
    output_shape = linear_bounds.bias_lower.shape
    output_ndim = len(output_shape)

    lower = linear_bounds.bias_lower.clone()
    upper = linear_bounds.bias_upper.clone()

    for ll, lu in zip(linear_bounds.linear_lowers, linear_bounds.linear_uppers, strict=True):
        input_ndim = x.ndim
        expanded = x.reshape(*([1] * output_ndim), *x.shape)
        sum_dims = tuple(range(-input_ndim, 0))
        if sum_dims:
            lower = lower + (ll * expanded).sum(dim=sum_dims)
            upper = upper + (lu * expanded).sum(dim=sum_dims)
        else:
            lower = lower + ll * expanded
            upper = upper + lu * expanded

    return lower, upper


def _check_soundness(fn, input_region, linear_bounds, num_samples=1000, atol=1e-4):
    """Verify linear bounds are sound by evaluating them at sampled points."""
    rand = torch.rand(num_samples, *input_region.lower.shape)
    samples = input_region.lower + rand * (input_region.upper - input_region.lower)
    for sample in samples:
        output = fn(sample)
        lower, upper = _evaluate_linear_bounds_at(linear_bounds, sample)
        assert torch.all(lower <= output + atol), f"Lower bound violation at x={sample}: lower={lower}, output={output}"
        assert torch.all(output <= upper + atol), f"Upper bound violation at x={sample}: upper={upper}, output={output}"


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


class TestForwardLBPComplexNonlinearities:
    """Test forward LBP with pairwise and reduction nonlinearities."""

    def test_pairwise_maximum_both_abstract(self) -> None:
        """y = max(x[:2], x[2:]): both arguments abstract."""

        def fn(x):
            return torch.maximum(x[:2], x[2:])

        gm = _trace_and_annotate(fn, (torch.randn(4),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0, -2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0, 1.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_pairwise_maximum_with_constant(self) -> None:
        """y = max(x, 0): effectively ReLU via maximum."""

        zero = torch.zeros(3)

        def fn(x):
            return torch.maximum(x, zero)

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, -3.0]),
            upper=torch.tensor([3.0, 2.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_pairwise_minimum_both_abstract(self) -> None:
        """y = min(x[:2], x[2:])."""

        def fn(x):
            return torch.minimum(x[:2], x[2:])

        gm = _trace_and_annotate(fn, (torch.randn(4),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0, -2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0, 1.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_amax_reduction_sound(self) -> None:
        """y = amax(relu(x @ W)): reduction over a nonlinear layer."""

        def fn(x):
            w = torch.tensor([[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]])
            h = torch.relu(x @ w)
            return torch.amax(h)

        gm = _trace_and_annotate(fn, (torch.randn(2),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_amin_reduction_sound(self) -> None:
        """y = amin(sigmoid(x))."""

        def fn(x):
            return torch.amin(torch.sigmoid(x))

        gm = _trace_and_annotate(fn, (torch.randn(4),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.0, 1.0]),
            upper=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_pairwise_mul_both_abstract(self) -> None:
        """y = x[:2] * x[2:]: both arguments abstract (nonlinear)."""

        def fn(x):
            return x[:2] * x[2:]

        gm = _trace_and_annotate(fn, (torch.randn(4),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.5, -2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0, 1.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_reciprocal_near_asymptote(self) -> None:
        """y = 1/x on [0.1, 2]: steep gradient near the x=0 asymptote."""

        def fn(x):
            return torch.reciprocal(x)

        gm = _trace_and_annotate(fn, (torch.rand(3) + 0.5,))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.1, 0.25, 0.5]),
            upper=torch.tensor([2.0, 3.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_log_near_asymptote(self) -> None:
        """y = log(x) with the region's lower edge near the x=0 asymptote."""

        def fn(x):
            return torch.log(x)

        gm = _trace_and_annotate(fn, (torch.rand(2) + 0.1,))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.05, 0.1]),
            upper=torch.tensor([1.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_div_constant_over_abstract_near_asymptote(self) -> None:
        """y = 1 / (x + 0.1): constant/abstract division, region approaches asymptote."""

        shift = torch.tensor([0.1, 0.1])

        def fn(x):
            return 1.0 / (x + shift)

        gm = _trace_and_annotate(fn, (torch.rand(2) + 0.5,))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])


class TestForwardLBPMultiInput:
    """Test forward LBP with functions that take multiple input tensors."""

    def test_two_input_add(self) -> None:
        """y = x + y: two separate input placeholders."""

        def fn(x, y):
            return x + y

        gm = _trace_and_annotate(fn, (torch.randn(3), torch.randn(3)))
        propagator = ForwardLBPPropagator(gm)

        x_region = HyperRectangle(
            lower=torch.tensor([1.0, 2.0, 3.0]),
            upper=torch.tensor([4.0, 5.0, 6.0]),
        )
        y_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0, 1.0]),
            upper=torch.tensor([0.0, 1.0, 2.0]),
        )
        outputs = propagator.propagate([x_region, y_region])

        lower, upper = outputs[0].concretize()
        assert torch.allclose(lower, torch.tensor([0.0, 2.0, 4.0]))
        assert torch.allclose(upper, torch.tensor([4.0, 6.0, 8.0]))

    def test_two_input_sub(self) -> None:
        """y = x - y: exact bounds."""

        def fn(x, y):
            return x - y

        gm = _trace_and_annotate(fn, (torch.randn(2), torch.randn(2)))
        propagator = ForwardLBPPropagator(gm)

        x_region = HyperRectangle(
            lower=torch.tensor([5.0, 10.0]),
            upper=torch.tensor([7.0, 12.0]),
        )
        y_region = HyperRectangle(
            lower=torch.tensor([1.0, 2.0]),
            upper=torch.tensor([3.0, 4.0]),
        )
        outputs = propagator.propagate([x_region, y_region])

        lower, upper = outputs[0].concretize()
        assert torch.allclose(lower, torch.tensor([2.0, 6.0]))
        assert torch.allclose(upper, torch.tensor([6.0, 10.0]))

    def test_two_input_relu_then_combine(self) -> None:
        """y = relu(x) + sigmoid(y): two inputs through different activations."""

        def fn(x, y):
            return torch.relu(x) + torch.sigmoid(y)

        gm = _trace_and_annotate(fn, (torch.randn(2), torch.randn(2)))
        propagator = ForwardLBPPropagator(gm)

        x_region = HyperRectangle(
            lower=torch.tensor([-2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0]),
        )
        y_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        outputs = propagator.propagate([x_region, y_region])

        linear_bounds = outputs[0]
        lower, upper = linear_bounds.concretize()
        num_samples = 1000
        x_rand = x_region.lower + torch.rand(num_samples, *x_region.lower.shape) * (x_region.upper - x_region.lower)
        y_rand = y_region.lower + torch.rand(num_samples, *y_region.lower.shape) * (y_region.upper - y_region.lower)
        for xs, ys in zip(x_rand, y_rand, strict=True):
            out = fn(xs, ys)
            assert torch.all(lower <= out + 1e-4)
            assert torch.all(out <= upper + 1e-4)


class TestForwardLBPDAG:
    """Test forward LBP on non-tree graphs: multiple paths from input to output."""

    def test_diamond_linear_paths(self) -> None:
        """y = 2*x + 3*x: two linear paths from x, should be exactly 5*x."""

        def fn(x):
            return x * 2.0 + x * 3.0

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([1.0, -1.0, 0.0]),
            upper=torch.tensor([2.0, 1.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs[0].concretize()
        assert torch.allclose(lower, torch.tensor([5.0, -5.0, 0.0]))
        assert torch.allclose(upper, torch.tensor([10.0, 5.0, 15.0]))

    def test_diamond_cancellation_exact(self) -> None:
        """y = x - x: two linear paths that cancel exactly to zero."""

        def fn(x):
            return x - x

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, 1.0, 0.5]),
            upper=torch.tensor([3.0, 4.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs[0].concretize()
        assert torch.allclose(lower, torch.zeros(3), atol=1e-5)
        assert torch.allclose(upper, torch.zeros(3), atol=1e-5)

    def test_diamond_relu_plus_identity(self) -> None:
        """y = relu(x) + x: diamond where one path is nonlinear."""

        def fn(x):
            return torch.relu(x) + x

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.5]),
            upper=torch.tensor([2.0, 3.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_diamond_two_linear_layers_shared_input(self) -> None:
        """y = (x @ W1) + (x @ W2): two affine paths from same input."""

        def fn(x):
            w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            w2 = torch.tensor([[-1.0, 0.0], [0.5, -0.5], [1.0, 1.0]])
            return x @ w1 + x @ w2

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        # Linear combination, should be exact.
        assert torch.allclose(outputs[0].linear_lowers[0], outputs[0].linear_uppers[0], atol=1e-5)
        _check_soundness(fn, input_region, outputs[0])

    def test_diamond_relu_sigmoid_sum(self) -> None:
        """y = relu(x) + sigmoid(x): two nonlinear paths from same input."""

        def fn(x):
            return torch.relu(x) + torch.sigmoid(x)

        gm = _trace_and_annotate(fn, (torch.randn(2),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, 0.0]),
            upper=torch.tensor([2.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])


class TestForwardLBPDeepNonlinearDAG:
    """Test forward LBP on DAGs with stacked nonlinearities around branch points."""

    def test_nonlinear_prebranch_two_nonlinear_branches(self) -> None:
        """Deep pre-branch nonlinearities, two nonlinear branches, nonlinear merge."""

        W0 = torch.tensor([[1.0, -0.5, 0.5], [0.5, 1.0, -1.0]])
        b0 = torch.tensor([0.1, -0.1, 0.2])
        Wa = torch.tensor([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]])
        Wb = torch.tensor([[-1.0, 0.5], [0.5, 1.0], [1.0, -0.5]])

        def fn(x):
            z = torch.tanh(torch.sigmoid(torch.relu(x @ W0 + b0)))
            a = torch.sigmoid(torch.relu(z @ Wa))
            b = torch.relu(torch.tanh(z @ Wb))
            return torch.sigmoid(a + b)

        gm = _trace_and_annotate(fn, (torch.randn(2),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_three_branches_nonlinear_each(self) -> None:
        """Three nonlinear branches from a shared nonlinear pre-feature."""

        W0 = torch.tensor([[0.5, -0.5, 1.0], [1.0, 0.5, -0.5]])
        Wa = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [0.5, 0.5]])
        ba = torch.tensor([0.1, -0.1])
        Wb = torch.tensor([[0.5, 0.5], [1.0, -1.0], [-0.5, 1.0]])
        bb = torch.tensor([-0.2, 0.2])
        Wc = torch.tensor([[-0.5, 1.0], [0.5, -0.5], [1.0, 0.5]])
        bc = torch.tensor([0.0, 0.3])

        def fn(x):
            z = torch.sigmoid(x @ W0)
            a = torch.relu(z @ Wa + ba)
            b = torch.tanh(z @ Wb + bb)
            c = torch.sigmoid(z @ Wc + bc)
            return torch.relu(a + b - c)

        gm = _trace_and_annotate(fn, (torch.randn(2),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_residual_skip_with_nonlinearities(self) -> None:
        """Residual-style DAG: deep nonlinear main path plus nonlinear skip."""

        W1 = torch.tensor([[1.0, -0.5, 0.5, 1.0], [0.5, 1.0, -1.0, -0.5]])
        W2 = torch.tensor([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0], [0.5, 0.5]])
        Ws = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-0.5, 0.5]])

        def fn(x):
            h1 = torch.relu(x @ W1)
            h2 = torch.sigmoid(torch.tanh(h1 @ W2))
            skip = torch.relu(h1 @ Ws)
            return torch.tanh(h2 + skip)

        gm = _trace_and_annotate(fn, (torch.randn(2),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_nested_diamond_nonlinear(self) -> None:
        """Diamond inside a diamond, all edges nonlinear."""

        def fn(x):
            p = torch.sigmoid(x)
            outer_a = torch.tanh(p)
            inner_a = torch.relu(p)
            inner_b = torch.sigmoid(p)
            outer_b = inner_a * 0.5 + inner_b * 0.5
            return torch.relu(outer_a + outer_b)

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.0]),
            upper=torch.tensor([1.0, 2.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])

    def test_parallel_nonlinear_towers_then_deep_merge(self) -> None:
        """Two independent deep nonlinear towers, then deep merge."""

        W0 = torch.tensor([[1.0, -0.5, 0.5], [0.5, 1.0, -1.0]])
        Wa1 = torch.tensor([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]])
        Wa2 = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
        Wb1 = torch.tensor([[-1.0, 0.5], [0.5, -1.0], [1.0, 1.0]])
        Wb2 = torch.tensor([[0.5, 1.0], [-1.0, 0.5]])

        def fn(x):
            pre = torch.tanh(torch.relu(x @ W0))
            tower_a = torch.sigmoid(torch.relu(pre @ Wa1) @ Wa2)
            tower_b = torch.relu(torch.tanh(pre @ Wb1) @ Wb2)
            merge = tower_a + tower_b
            return torch.sigmoid(torch.tanh(merge))

        gm = _trace_and_annotate(fn, (torch.randn(2),))
        propagator = ForwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-0.75, -0.75]),
            upper=torch.tensor([0.75, 0.75]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs[0])


class TestForwardLBPEdgeCases:
    """Test forward LBP on degenerate / boundary inputs."""

    def test_zero_width_region_identity(self) -> None:
        """Degenerate region [a, a]: identity gives exact f(a)."""

        def fn(x):
            return x

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        point = torch.tensor([2.0, -1.0, 3.5])
        input_region = HyperRectangle(lower=point, upper=point)
        outputs = propagator.propagate([input_region])

        lower, upper = outputs[0].concretize()
        assert torch.allclose(lower, point)
        assert torch.allclose(upper, point)

    def test_zero_width_region_relu(self) -> None:
        """Degenerate region through ReLU: bounds collapse to f(a)."""

        def fn(x):
            return torch.relu(x)

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        point = torch.tensor([-2.0, 0.0, 3.0])
        input_region = HyperRectangle(lower=point, upper=point)
        outputs = propagator.propagate([input_region])

        lower, upper = outputs[0].concretize()
        expected = torch.relu(point)
        assert torch.allclose(lower, expected, atol=1e-5)
        assert torch.allclose(upper, expected, atol=1e-5)

    def test_zero_width_region_through_network(self) -> None:
        """Degenerate input through a full network collapses to point evaluation."""

        def fn(x):
            w1 = torch.tensor([[1.0, -2.0], [3.0, 1.0]])
            h = torch.relu(x @ w1)
            w2 = torch.tensor([[1.0], [-1.0]])
            return h @ w2

        gm = _trace_and_annotate(fn, (torch.randn(2),))
        propagator = ForwardLBPPropagator(gm)

        point = torch.tensor([0.5, -0.3])
        input_region = HyperRectangle(lower=point, upper=point)
        outputs = propagator.propagate([input_region])

        lower, upper = outputs[0].concretize()
        expected = fn(point)
        assert torch.allclose(lower, expected, atol=1e-5)
        assert torch.allclose(upper, expected, atol=1e-5)

    def test_zero_width_at_relu_kink(self) -> None:
        """Zero-width region at x=0 (the ReLU kink) should give bounds of 0."""

        def fn(x):
            return torch.relu(x)

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = ForwardLBPPropagator(gm)

        point = torch.zeros(3)
        input_region = HyperRectangle(lower=point, upper=point)
        outputs = propagator.propagate([input_region])

        lower, upper = outputs[0].concretize()
        assert torch.allclose(lower, torch.zeros(3), atol=1e-5)
        assert torch.allclose(upper, torch.zeros(3), atol=1e-5)
