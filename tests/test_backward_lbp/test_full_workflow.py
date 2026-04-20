"""
End-to-end Backward LBP (CROWN) workflow tests.

Tests the complete Backward LBP pipeline:
1. Tracing PyTorch functions/modules with BoundPropagationTracer
2. Annotating metadata with MetadataPass
3. Propagating bounds with BackwardLBPPropagator
4. Verifying soundness (true bounds lie within computed bounds)
5. Comparing tightness with Forward LBP
"""

from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import BackwardLBPPropagator, ForwardLBPPropagator
from bound_propagation.propagation.backward_lbp import create_default_backward_lbp_registry
from bound_propagation.propagation.forward_lbp import create_default_forward_lbp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def _trace_and_annotate(fn_or_module, example_inputs, registry=None):
    """Trace, annotate metadata, and return GraphModule."""
    if registry is None:
        registry = create_default_backward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn_or_module)
    MetadataPass(gm).run(*example_inputs)
    return gm


def _evaluate_linear_bounds_at(linear_bounds, x):
    """Evaluate linear bounds at a specific point *x*.

    Returns per-point (lower, upper) that are at least as tight as
    the concretized interval bounds.

    Parameters
    ----------
    linear_bounds : LinearBounds
        Affine bounds ``W_l @ x + b_l <= y <= W_u @ x + b_u``.
    x : torch.Tensor
        A concrete input point.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(lower_at_x, upper_at_x)``
    """
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
    """Verify linear bounds are sound by evaluating them at sampled points.

    For each sample *x*, evaluates the affine lower/upper bounds at *x*
    (tighter than the concretized interval bounds) and checks that the
    true output lies within them.

    Parameters
    ----------
    fn : callable
        The true function.
    input_region : HyperRectangle
        Input region to sample from.
    linear_bounds : LinearBounds
        Affine bounds to verify.
    num_samples : int
        Number of random samples.
    atol : float
        Absolute tolerance for floating-point imprecision accumulated
        through chained matrix operations.
    """
    rand = torch.rand(num_samples, *input_region.lower.shape)
    samples = input_region.lower + rand * (input_region.upper - input_region.lower)
    for sample in samples:
        output = fn(sample)
        lower, upper = _evaluate_linear_bounds_at(linear_bounds, sample)
        assert torch.all(lower <= output + atol), f"Lower bound violation at x={sample}: lower={lower}, output={output}"
        assert torch.all(output <= upper + atol), f"Upper bound violation at x={sample}: upper={upper}, output={output}"


class TestBackwardLBPIdentity:
    """Test backward LBP with identity / linear operations only."""

    def test_identity(self) -> None:
        def identity_fn(x):
            return x

        gm = _trace_and_annotate(identity_fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 2.0, 3.0]),
            upper=torch.tensor([4.0, 5.0, 6.0]),
        )
        outputs = propagator.propagate([input_region])

        out = outputs
        assert isinstance(out, LinearBounds)
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(upper, torch.tensor([4.0, 5.0, 6.0]))

    def test_add_constant(self) -> None:
        def add_fn(x):
            return x + torch.tensor([10.0, 20.0])

        gm = _trace_and_annotate(add_fn, (torch.randn(2),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        assert torch.allclose(lower, torch.tensor([10.0, 20.0]))
        assert torch.allclose(upper, torch.tensor([11.0, 21.0]))

    def test_negation(self) -> None:
        def neg_fn(x):
            return -x

        gm = _trace_and_annotate(neg_fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 2.0, 3.0]),
            upper=torch.tensor([4.0, 5.0, 6.0]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        assert torch.allclose(lower, torch.tensor([-4.0, -5.0, -6.0]))
        assert torch.allclose(upper, torch.tensor([-1.0, -2.0, -3.0]))

    def test_scale_by_constant(self) -> None:
        def scale_fn(x):
            return x * 2.0

        gm = _trace_and_annotate(scale_fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([1.0, -1.0, 0.0]),
            upper=torch.tensor([3.0, 1.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        assert torch.allclose(lower, torch.tensor([2.0, -2.0, 0.0]))
        assert torch.allclose(upper, torch.tensor([6.0, 2.0, 4.0]))


class TestBackwardLBPMatmul:
    """Test backward LBP with matmul operations."""

    def test_matmul_right_constant(self) -> None:
        def matmul_fn(x):
            weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            return x @ weight

        gm = _trace_and_annotate(matmul_fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        # lower: all inputs at 0 -> [0, 0]
        # upper: all inputs at 1 -> [9, 12]
        assert torch.allclose(lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(upper, torch.tensor([9.0, 12.0]))

    def test_matmul_with_bias(self) -> None:
        def affine_fn(x):
            weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            bias = torch.tensor([1.0, -1.0])
            return x @ weight + bias

        gm = _trace_and_annotate(affine_fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        assert torch.allclose(lower, torch.tensor([1.0, -1.0]))
        assert torch.allclose(upper, torch.tensor([10.0, 11.0]))


class TestBackwardLBPRelu:
    """Test backward LBP with ReLU."""

    def test_single_relu(self) -> None:
        def relu_fn(x):
            return torch.relu(x)

        gm = _trace_and_annotate(relu_fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0, -2.0]),
            upper=torch.tensor([3.0, 3.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])

        linear_bounds = outputs
        lower, upper = linear_bounds.concretize()
        # Sound: lower should be <= 0 (min of relu on [-2,3]) = 0
        # Upper should be >= 3 (max of relu on [-2,3]) = 3
        assert torch.all(lower <= 0.0 + 1e-6)
        assert torch.all(upper >= 3.0 - 1e-6)
        _check_soundness(relu_fn, input_region, linear_bounds)

    def test_relu_positive_only(self) -> None:
        def relu_fn(x):
            return torch.relu(x)

        gm = _trace_and_annotate(relu_fn, (torch.randn(2),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([1.0, 2.0]),
            upper=torch.tensor([3.0, 5.0]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        # In positive regime: relu is identity, bounds should be exact
        assert torch.allclose(lower, torch.tensor([1.0, 2.0]))
        assert torch.allclose(upper, torch.tensor([3.0, 5.0]))

    def test_relu_negative_only(self) -> None:
        def relu_fn(x):
            return torch.relu(x)

        gm = _trace_and_annotate(relu_fn, (torch.randn(2),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-5.0, -3.0]),
            upper=torch.tensor([-1.0, -0.5]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        # Negative regime: relu output is always 0
        assert torch.allclose(lower, torch.tensor([0.0, 0.0]))
        assert torch.allclose(upper, torch.tensor([0.0, 0.0]))


class TestBackwardLBPTwoLayer:
    """Test backward LBP with multi-layer networks."""

    def test_linear_relu_linear(self) -> None:
        """Two-layer network: y = relu(x @ W1) @ W2."""

        def two_layer_fn(x):
            w1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            h = torch.relu(x @ w1)
            w2 = torch.tensor([[1.0], [1.0], [1.0]])
            return h @ w2

        gm = _trace_and_annotate(two_layer_fn, (torch.randn(2),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        linear_bounds = outputs
        lower, upper = linear_bounds.concretize()
        # Sound: all inputs >= 0, all weights >= 0, so:
        # min at (0,0): h = [0,0,0], output = 0
        # max at (1,1): h = relu([5,7,9]) = [5,7,9], output = 21
        assert torch.all(lower <= 0.0 + 1e-5)
        assert torch.all(upper >= 21.0 - 1e-5)
        _check_soundness(two_layer_fn, input_region, linear_bounds)

    def test_backward_at_least_as_tight_as_forward(self) -> None:
        """CROWN should give bounds at least as tight as forward LBP for single-input networks."""

        def network(x):
            w1 = torch.tensor([[1.0, -2.0], [3.0, 1.0], [-1.0, 2.0]])
            h = torch.relu(x @ w1)
            w2 = torch.tensor([[1.0], [-1.0]])
            return h @ w2

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )

        # Forward LBP
        fwd_registry = create_default_forward_lbp_registry()
        fwd_gm = _trace_and_annotate(network, (torch.randn(3),), registry=fwd_registry)
        fwd_propagator = ForwardLBPPropagator(fwd_gm)
        fwd_outputs = fwd_propagator.propagate([input_region])
        fwd_bounds = fwd_outputs
        fwd_lower, fwd_upper = fwd_bounds.concretize()

        # Backward LBP
        bwd_gm = _trace_and_annotate(network, (torch.randn(3),))
        bwd_propagator = BackwardLBPPropagator(bwd_gm)
        bwd_outputs = bwd_propagator.propagate([input_region])
        bwd_bounds = bwd_outputs
        bwd_lower, bwd_upper = bwd_bounds.concretize()

        # Both should be sound (checked via per-point linear bound evaluation)
        _check_soundness(network, input_region, fwd_bounds)
        _check_soundness(network, input_region, bwd_bounds)

        # CROWN should be at least as tight (higher lower, lower upper)
        assert torch.all(bwd_lower >= fwd_lower - 1e-5), (
            f"CROWN lower {bwd_lower} should be >= forward lower {fwd_lower}"
        )
        assert torch.all(bwd_upper <= fwd_upper + 1e-5), (
            f"CROWN upper {bwd_upper} should be <= forward upper {fwd_upper}"
        )


class TestBackwardLBPSigmoid:
    """Test backward LBP with sigmoid."""

    def test_sigmoid(self) -> None:
        def sigmoid_fn(x):
            return torch.sigmoid(x)

        gm = _trace_and_annotate(sigmoid_fn, (torch.randn(2),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, 0.0]),
            upper=torch.tensor([2.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])

        linear_bounds = outputs
        lower, upper = linear_bounds.concretize()
        _check_soundness(sigmoid_fn, input_region, linear_bounds)
        # Linear relaxation bounds may exceed sigmoid's [0,1] range
        # (tangent lines extend beyond the function), but must still be sound.
        assert torch.all(lower >= -0.5), f"Lower bound unreasonably low: {lower}"
        assert torch.all(upper <= 1.5), f"Upper bound unreasonably high: {upper}"


class TestBackwardLBPComplexNonlinearities:
    """Test backward LBP with pairwise and reduction nonlinearities."""

    def test_pairwise_maximum_both_abstract(self) -> None:
        """y = max(x[:2], x[2:]): both arguments abstract."""

        def fn(x):
            return torch.maximum(x[:2], x[2:])

        gm = _trace_and_annotate(fn, (torch.randn(4),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0, -2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0, 1.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_pairwise_maximum_with_constant(self) -> None:
        """y = max(x, 0): effectively ReLU via maximum."""

        zero = torch.zeros(3)

        def fn(x):
            return torch.maximum(x, zero)

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, -3.0]),
            upper=torch.tensor([3.0, 2.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_pairwise_minimum_both_abstract(self) -> None:
        """y = min(x[:2], x[2:])."""

        def fn(x):
            return torch.minimum(x[:2], x[2:])

        gm = _trace_and_annotate(fn, (torch.randn(4),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0, -2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0, 1.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_amax_reduction_sound(self) -> None:
        """y = amax(relu(x @ W), dim=0): reduction over a nonlinear layer."""

        def fn(x):
            w = torch.tensor([[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]])
            h = torch.relu(x @ w)
            return torch.amax(h)

        gm = _trace_and_annotate(fn, (torch.randn(2),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_amin_reduction_sound(self) -> None:
        """y = amin(sigmoid(x))."""

        def fn(x):
            return torch.amin(torch.sigmoid(x))

        gm = _trace_and_annotate(fn, (torch.randn(4),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.0, 1.0]),
            upper=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_pairwise_mul_both_abstract(self) -> None:
        """y = x[:2] * x[2:]: both arguments abstract (nonlinear)."""

        def fn(x):
            return x[:2] * x[2:]

        gm = _trace_and_annotate(fn, (torch.randn(4),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.5, -2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0, 1.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_reciprocal_near_asymptote(self) -> None:
        """y = 1/x on [0.1, 2]: steep gradient near the x=0 asymptote."""

        def fn(x):
            return torch.reciprocal(x)

        gm = _trace_and_annotate(fn, (torch.rand(3) + 0.5,))
        propagator = BackwardLBPPropagator(gm)

        # Keep the region strictly positive but close to the asymptote on the low side.
        input_region = HyperRectangle(
            lower=torch.tensor([0.1, 0.25, 0.5]),
            upper=torch.tensor([2.0, 3.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])

        linear_bounds = outputs
        lower, upper = linear_bounds.concretize()
        _check_soundness(fn, input_region, linear_bounds)
        # Sanity: true range of 1/x on [0.1, 2] is [0.5, 10], so bounds must contain it.
        assert torch.all(lower <= 0.5 + 1e-5)
        assert torch.all(upper >= 10.0 - 1e-5) or upper[0] >= 10.0 - 1e-5

    def test_log_near_asymptote(self) -> None:
        """y = log(x) on a region whose lower edge sits near the x=0 asymptote."""

        def fn(x):
            return torch.log(x)

        gm = _trace_and_annotate(fn, (torch.rand(2) + 0.1,))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.05, 0.1]),
            upper=torch.tensor([1.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_div_constant_over_abstract_near_asymptote(self) -> None:
        """y = 1 / (x + 0.1): constant/abstract division, region approaches asymptote."""

        shift = torch.tensor([0.1, 0.1])

        def fn(x):
            return 1.0 / (x + shift)

        gm = _trace_and_annotate(fn, (torch.rand(2) + 0.5,))
        propagator = BackwardLBPPropagator(gm)

        # (x + 0.1) in [0.1, 2.1] — approaches the x=0 asymptote of 1/x.
        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)


class TestBackwardLBPMultiInput:
    """Test backward LBP with functions that take multiple input tensors."""

    def test_two_input_add(self) -> None:
        """y = x + y: two separate input placeholders."""

        def fn(x, y):
            return x + y

        gm = _trace_and_annotate(fn, (torch.randn(3), torch.randn(3)))
        propagator = BackwardLBPPropagator(gm)

        x_region = HyperRectangle(
            lower=torch.tensor([1.0, 2.0, 3.0]),
            upper=torch.tensor([4.0, 5.0, 6.0]),
        )
        y_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0, 1.0]),
            upper=torch.tensor([0.0, 1.0, 2.0]),
        )
        outputs = propagator.propagate([x_region, y_region])

        lower, upper = outputs.concretize()
        assert torch.allclose(lower, torch.tensor([0.0, 2.0, 4.0]))
        assert torch.allclose(upper, torch.tensor([4.0, 6.0, 8.0]))

    def test_two_input_sub(self) -> None:
        """y = x - y: exact bounds."""

        def fn(x, y):
            return x - y

        gm = _trace_and_annotate(fn, (torch.randn(2), torch.randn(2)))
        propagator = BackwardLBPPropagator(gm)

        x_region = HyperRectangle(
            lower=torch.tensor([5.0, 10.0]),
            upper=torch.tensor([7.0, 12.0]),
        )
        y_region = HyperRectangle(
            lower=torch.tensor([1.0, 2.0]),
            upper=torch.tensor([3.0, 4.0]),
        )
        outputs = propagator.propagate([x_region, y_region])

        lower, upper = outputs.concretize()
        assert torch.allclose(lower, torch.tensor([2.0, 6.0]))  # 5-3, 10-4
        assert torch.allclose(upper, torch.tensor([6.0, 10.0]))  # 7-1, 12-2

    def test_two_input_relu_then_combine(self) -> None:
        """y = relu(x) + sigmoid(y): two inputs through different activations."""

        def fn(x, y):
            return torch.relu(x) + torch.sigmoid(y)

        gm = _trace_and_annotate(fn, (torch.randn(2), torch.randn(2)))
        propagator = BackwardLBPPropagator(gm)

        x_region = HyperRectangle(
            lower=torch.tensor([-2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0]),
        )
        y_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        outputs = propagator.propagate([x_region, y_region])

        # Sample both input regions jointly to check soundness.
        linear_bounds = outputs
        lower, upper = linear_bounds.concretize()
        num_samples = 1000
        x_rand = x_region.lower + torch.rand(num_samples, *x_region.lower.shape) * (x_region.upper - x_region.lower)
        y_rand = y_region.lower + torch.rand(num_samples, *y_region.lower.shape) * (y_region.upper - y_region.lower)
        for xs, ys in zip(x_rand, y_rand, strict=True):
            out = fn(xs, ys)
            assert torch.all(lower <= out + 1e-4)
            assert torch.all(out <= upper + 1e-4)


class TestBackwardLBPDAG:
    """Test backward LBP on non-tree graphs: multiple paths from input to output."""

    def test_diamond_linear_paths(self) -> None:
        """y = 2*x + 3*x: two linear paths from x, should be exactly 5*x."""

        def fn(x):
            return x * 2.0 + x * 3.0

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([1.0, -1.0, 0.0]),
            upper=torch.tensor([2.0, 1.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        assert torch.allclose(lower, torch.tensor([5.0, -5.0, 0.0]))
        assert torch.allclose(upper, torch.tensor([10.0, 5.0, 15.0]))

    def test_diamond_cancellation_exact(self) -> None:
        """y = x - x: two linear paths that cancel exactly to zero."""

        def fn(x):
            return x - x

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, 1.0, 0.5]),
            upper=torch.tensor([3.0, 4.0, 2.0]),
        )
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        assert torch.allclose(lower, torch.zeros(3), atol=1e-5)
        assert torch.allclose(upper, torch.zeros(3), atol=1e-5)

    def test_diamond_relu_plus_identity(self) -> None:
        """y = relu(x) + x: diamond where one path is nonlinear."""

        def fn(x):
            return torch.relu(x) + x

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.5]),
            upper=torch.tensor([2.0, 3.0, 4.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_diamond_two_linear_layers_shared_input(self) -> None:
        """y = (x @ W1) + (x @ W2): two affine paths from same input."""

        def fn(x):
            w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            w2 = torch.tensor([[-1.0, 0.0], [0.5, -0.5], [1.0, 1.0]])
            return x @ w1 + x @ w2

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])

        # Linear combination, should be exact.
        assert torch.allclose(outputs.linear_lowers[0], outputs.linear_uppers[0], atol=1e-5)
        _check_soundness(fn, input_region, outputs)

    def test_diamond_relu_sigmoid_sum(self) -> None:
        """y = relu(x) + sigmoid(x): two nonlinear paths from same input."""

        def fn(x):
            return torch.relu(x) + torch.sigmoid(x)

        gm = _trace_and_annotate(fn, (torch.randn(2),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, 0.0]),
            upper=torch.tensor([2.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)


class TestBackwardLBPDeepNonlinearDAG:
    """Test backward LBP on DAGs with stacked nonlinearities around branch points."""

    def test_nonlinear_prebranch_two_nonlinear_branches(self) -> None:
        """Deep pre-branch nonlinearities, two nonlinear branches, nonlinear merge.

        Shape:
            z   = tanh(sigmoid(relu(x @ W0 + b0)))   # pre (3 activations stacked)
            a   = sigmoid(relu(z @ Wa))              # branch A (2 activations)
            b   = relu(tanh(z @ Wb))                 # branch B (2 activations)
            y   = sigmoid(a + b)                     # merge + post nonlinearity
        """

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
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_three_branches_nonlinear_each(self) -> None:
        """Three nonlinear branches from a shared nonlinear pre-feature.

        z = sigmoid(x @ W0)
        a = relu(z @ Wa + ba)
        b = tanh(z @ Wb + bb)
        c = sigmoid(z @ Wc + bc)
        y = relu(a + b - c)  # post-merge nonlinearity
        """

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
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_residual_skip_with_nonlinearities(self) -> None:
        """Residual-style DAG: deep nonlinear main path plus nonlinear skip.

        h1  = relu(x @ W1)           # reused by two paths
        h2  = sigmoid(tanh(h1 @ W2)) # main path
        skip = relu(h1 @ Ws)         # skip path (uses h1 again)
        y   = tanh(h2 + skip)        # post-merge nonlinearity
        """

        W1 = torch.tensor([[1.0, -0.5, 0.5, 1.0], [0.5, 1.0, -1.0, -0.5]])
        W2 = torch.tensor([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0], [0.5, 0.5]])
        Ws = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-0.5, 0.5]])

        def fn(x):
            h1 = torch.relu(x @ W1)
            h2 = torch.sigmoid(torch.tanh(h1 @ W2))
            skip = torch.relu(h1 @ Ws)
            return torch.tanh(h2 + skip)

        gm = _trace_and_annotate(fn, (torch.randn(2),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_nested_diamond_nonlinear(self) -> None:
        """Diamond inside a diamond, all edges nonlinear.

        p     = sigmoid(x)           # pre
        outer_a = tanh(p)            # outer left branch
        inner_a = relu(p)            # inner branches (share p again)
        inner_b = sigmoid(p)
        outer_b = inner_a * 0.5 + inner_b * 0.5  # inner merge
        y     = relu(outer_a + outer_b)          # outer merge + post
        """

        def fn(x):
            p = torch.sigmoid(x)
            outer_a = torch.tanh(p)
            inner_a = torch.relu(p)
            inner_b = torch.sigmoid(p)
            outer_b = inner_a * 0.5 + inner_b * 0.5
            return torch.relu(outer_a + outer_b)

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.0]),
            upper=torch.tensor([1.0, 2.0, 3.0]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)

    def test_parallel_nonlinear_towers_then_deep_merge(self) -> None:
        """Two independent deep nonlinear towers on a shared pre-feature, then deep merge.

        pre  = tanh(relu(x @ W0))
        towerA = sigmoid(relu(pre @ Wa1) @ Wa2)   # 2 nonlinear layers
        towerB = relu(tanh(pre @ Wb1) @ Wb2)      # 2 nonlinear layers
        merge  = towerA + towerB
        y      = sigmoid(tanh(merge))              # 2 post-merge nonlinearities
        """

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
        propagator = BackwardLBPPropagator(gm)

        input_region = HyperRectangle(
            lower=torch.tensor([-0.75, -0.75]),
            upper=torch.tensor([0.75, 0.75]),
        )
        outputs = propagator.propagate([input_region])
        _check_soundness(fn, input_region, outputs)


class TestBackwardLBPEdgeCases:
    """Test backward LBP on degenerate / boundary inputs."""

    def test_zero_width_region_identity(self) -> None:
        """Degenerate region [a, a]: identity gives exact f(a)."""

        def fn(x):
            return x

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        point = torch.tensor([2.0, -1.0, 3.5])
        input_region = HyperRectangle(lower=point, upper=point)
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        assert torch.allclose(lower, point)
        assert torch.allclose(upper, point)

    def test_zero_width_region_relu(self) -> None:
        """Degenerate region through ReLU: bounds collapse to f(a)."""

        def fn(x):
            return torch.relu(x)

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        # Mix of positive and negative points.
        point = torch.tensor([-2.0, 0.0, 3.0])
        input_region = HyperRectangle(lower=point, upper=point)
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
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
        propagator = BackwardLBPPropagator(gm)

        point = torch.tensor([0.5, -0.3])
        input_region = HyperRectangle(lower=point, upper=point)
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        expected = fn(point)
        assert torch.allclose(lower, expected, atol=1e-5)
        assert torch.allclose(upper, expected, atol=1e-5)

    def test_zero_width_at_relu_kink(self) -> None:
        """Zero-width region at x=0 (the ReLU kink) should give bounds of 0."""

        def fn(x):
            return torch.relu(x)

        gm = _trace_and_annotate(fn, (torch.randn(3),))
        propagator = BackwardLBPPropagator(gm)

        point = torch.zeros(3)
        input_region = HyperRectangle(lower=point, upper=point)
        outputs = propagator.propagate([input_region])

        lower, upper = outputs.concretize()
        assert torch.allclose(lower, torch.zeros(3), atol=1e-5)
        assert torch.allclose(upper, torch.zeros(3), atol=1e-5)
