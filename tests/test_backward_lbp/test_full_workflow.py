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
        assert torch.all(lower <= output + atol), (
            f"Lower bound violation at x={sample}: lower={lower}, output={output}"
        )
        assert torch.all(output <= upper + atol), (
            f"Upper bound violation at x={sample}: upper={upper}, output={output}"
        )


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

        out = outputs[0]
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

        lower, upper = outputs[0].concretize()
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

        lower, upper = outputs[0].concretize()
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

        lower, upper = outputs[0].concretize()
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

        lower, upper = outputs[0].concretize()
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

        lower, upper = outputs[0].concretize()
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

        linear_bounds = outputs[0]
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

        lower, upper = outputs[0].concretize()
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

        lower, upper = outputs[0].concretize()
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

        linear_bounds = outputs[0]
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
        fwd_bounds = fwd_outputs[0]
        fwd_lower, fwd_upper = fwd_bounds.concretize()

        # Backward LBP
        bwd_gm = _trace_and_annotate(network, (torch.randn(3),))
        bwd_propagator = BackwardLBPPropagator(bwd_gm)
        bwd_outputs = bwd_propagator.propagate([input_region])
        bwd_bounds = bwd_outputs[0]
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

        linear_bounds = outputs[0]
        lower, upper = linear_bounds.concretize()
        _check_soundness(sigmoid_fn, input_region, linear_bounds)
        # Linear relaxation bounds may exceed sigmoid's [0,1] range
        # (tangent lines extend beyond the function), but must still be sound.
        assert torch.all(lower >= -0.5), f"Lower bound unreasonably low: {lower}"
        assert torch.all(upper <= 1.5), f"Upper bound unreasonably high: {upper}"
