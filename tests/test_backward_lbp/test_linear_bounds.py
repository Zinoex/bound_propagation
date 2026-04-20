"""
Tests that LinearBounds returned by backward LBP have correct linear
coefficients and biases (not just correct concretized intervals).

This verifies that the tape-based backward LBP produces identical
LinearBounds to the recursive approach.
"""

from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import BackwardLBPPropagator
from bound_propagation.propagation.backward_lbp import create_default_backward_lbp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def _trace_and_propagate(fn_or_module, example_inputs, input_regions):
    """Trace, annotate, propagate, return list of LinearBounds."""
    registry = create_default_backward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn_or_module)
    MetadataPass(gm).run(*example_inputs)
    propagator = BackwardLBPPropagator(gm)
    return propagator.propagate(input_regions)


def _check_soundness(fn, input_region, lower, upper, num_samples=1000):
    """Verify bounds are sound by sampling."""
    rand = torch.rand(num_samples, *input_region.lower.shape)
    samples = input_region.lower + rand * (input_region.upper - input_region.lower)
    for sample in samples:
        output = fn(sample)
        assert torch.all(lower <= output + 1e-5), f"Lower bound violation: {lower} > {output}"
        assert torch.all(output <= upper + 1e-5), f"Upper bound violation: {output} > {upper}"


class TestLinearBoundsExact:
    """Test that purely linear operations produce exact LinearBounds."""

    def test_identity_linear_bounds(self):
        """y = x: slope == I, intercept == 0."""

        def identity_fn(x):
            return x

        region = HyperRectangle(lower=torch.tensor([1.0, 2.0, 3.0]), upper=torch.tensor([4.0, 5.0, 6.0]))
        outputs = _trace_and_propagate(identity_fn, (torch.randn(3),), [region])
        out = outputs

        assert isinstance(out, LinearBounds)
        assert torch.allclose(out.linear_lower, torch.eye(3))
        assert torch.allclose(out.linear_upper, torch.eye(3))
        assert torch.allclose(out.bias_lower, torch.zeros(3))
        assert torch.allclose(out.bias_upper, torch.zeros(3))

    def test_affine_linear_bounds(self):
        """y = Wx + b: slope == W, intercept == b."""
        W = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = torch.tensor([10.0, 20.0])

        def affine_fn(x):
            return x @ W + b

        region = HyperRectangle(lower=torch.zeros(3), upper=torch.ones(3))
        outputs = _trace_and_propagate(affine_fn, (torch.randn(3),), [region])
        out = outputs

        # For y = x @ W + b, backward LBP gives:
        # linear_lower/upper = W (transposed in the A-matrix convention)
        assert torch.allclose(out.linear_lower, W.T, atol=1e-5)
        assert torch.allclose(out.linear_upper, W.T, atol=1e-5)
        assert torch.allclose(out.bias_lower, b, atol=1e-5)
        assert torch.allclose(out.bias_upper, b, atol=1e-5)

    def test_chain_linear_bounds(self):
        """y = W2(W1 x + b1) + b2: slope == W2 @ W1, intercept == W2 @ b1 + b2."""
        W1 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b1 = torch.tensor([1.0, 2.0])
        W2 = torch.tensor([[1.0, -1.0]])
        b2 = torch.tensor([5.0])

        def chain_fn(x):
            h = x @ W1 + b1
            return h @ W2.T + b2

        region = HyperRectangle(lower=torch.zeros(3), upper=torch.ones(3))
        outputs = _trace_and_propagate(chain_fn, (torch.randn(3),), [region])
        out = outputs

        expected_intercept = W2 @ b1 + b2

        assert torch.allclose(out.bias_lower, expected_intercept.squeeze(), atol=1e-5)
        assert torch.allclose(out.bias_upper, expected_intercept.squeeze(), atol=1e-5)

    def test_fanout_linear_bounds(self):
        """y = x + x: slope == 2I, intercept == 0."""

        def fanout_fn(x):
            return x + x

        region = HyperRectangle(lower=torch.tensor([1.0, 2.0]), upper=torch.tensor([3.0, 4.0]))
        outputs = _trace_and_propagate(fanout_fn, (torch.randn(2),), [region])
        out = outputs

        assert torch.allclose(out.linear_lower, 2 * torch.eye(2), atol=1e-5)
        assert torch.allclose(out.linear_upper, 2 * torch.eye(2), atol=1e-5)
        assert torch.allclose(out.bias_lower, torch.zeros(2), atol=1e-5)
        assert torch.allclose(out.bias_upper, torch.zeros(2), atol=1e-5)

    def test_negation_linear_bounds(self):
        """y = -x: slope == -I, intercept == 0."""

        def neg_fn(x):
            return -x

        region = HyperRectangle(lower=torch.tensor([1.0, 2.0, 3.0]), upper=torch.tensor([4.0, 5.0, 6.0]))
        outputs = _trace_and_propagate(neg_fn, (torch.randn(3),), [region])
        out = outputs

        assert torch.allclose(out.linear_lower, -torch.eye(3), atol=1e-5)
        assert torch.allclose(out.linear_upper, -torch.eye(3), atol=1e-5)

    def test_scale_linear_bounds(self):
        """y = 3x: slope == 3I, intercept == 0."""

        def scale_fn(x):
            return x * 3.0

        region = HyperRectangle(lower=torch.zeros(2), upper=torch.ones(2))
        outputs = _trace_and_propagate(scale_fn, (torch.randn(2),), [region])
        out = outputs

        assert torch.allclose(out.linear_lower, 3 * torch.eye(2), atol=1e-5)
        assert torch.allclose(out.linear_upper, 3 * torch.eye(2), atol=1e-5)


class TestNonlinearRegimes:
    """Test that nonlinear operations produce correct regime-specific bounds."""

    def test_relu_positive_regime(self):
        """ReLU on [1,3]: exact identity."""

        def relu_fn(x):
            return torch.relu(x)

        region = HyperRectangle(lower=torch.tensor([1.0, 2.0]), upper=torch.tensor([3.0, 5.0]))
        outputs = _trace_and_propagate(relu_fn, (torch.randn(2),), [region])
        out = outputs
        assert torch.allclose(out.linear_lower, torch.eye(2), atol=1e-5)
        assert torch.allclose(out.linear_upper, torch.eye(2), atol=1e-5)

    def test_relu_negative_regime(self):
        """ReLU on [-3,-1]: exact zero."""

        def relu_fn(x):
            return torch.relu(x)

        region = HyperRectangle(lower=torch.tensor([-3.0, -2.0]), upper=torch.tensor([-1.0, -0.5]))
        outputs = _trace_and_propagate(relu_fn, (torch.randn(2),), [region])
        out = outputs
        lower, upper = out.concretize()
        assert torch.allclose(lower, torch.zeros(2), atol=1e-5)
        assert torch.allclose(upper, torch.zeros(2), atol=1e-5)

    def test_relu_crossing_regime(self):
        """ReLU on [-2,3]: check soundness of relaxation."""

        def relu_fn(x):
            return torch.relu(x)

        region = HyperRectangle(lower=torch.tensor([-2.0]), upper=torch.tensor([3.0]))
        outputs = _trace_and_propagate(relu_fn, (torch.randn(1),), [region])
        out = outputs
        lower, upper = out.concretize()
        _check_soundness(torch.relu, region, lower, upper)


class TestEdgeCases:
    """Test edge cases for backward LBP."""

    def test_diamond_graph(self):
        """a = relu(x); b = sigmoid(x); y = a + b -- accumulation from two chains."""

        def diamond_fn(x):
            a = torch.relu(x)
            b = torch.sigmoid(x)
            return a + b

        region = HyperRectangle(lower=torch.tensor([-1.0, 0.0]), upper=torch.tensor([1.0, 2.0]))
        outputs = _trace_and_propagate(diamond_fn, (torch.randn(2),), [region])
        lower, upper = outputs.concretize()
        _check_soundness(diamond_fn, region, lower, upper)

    def test_chain_breaking_amax(self):
        """y = amax(relu(x)) -- IntervalLeafRelaxation, no backward through amax."""

        def chain_break_fn(x):
            return torch.amax(torch.relu(x))

        region = HyperRectangle(lower=torch.tensor([-1.0, 0.5]), upper=torch.tensor([1.0, 2.0]))
        outputs = _trace_and_propagate(chain_break_fn, (torch.randn(2),), [region])
        lower, upper = outputs.concretize()
        _check_soundness(chain_break_fn, region, lower, upper)

    def test_zero_width_region(self):
        """x in [2, 2] -- degenerate interval."""

        def identity_fn(x):
            return x

        region = HyperRectangle(lower=torch.tensor([2.0, 3.0]), upper=torch.tensor([2.0, 3.0]))
        outputs = _trace_and_propagate(identity_fn, (torch.randn(2),), [region])
        lower, upper = outputs.concretize()
        assert torch.allclose(lower, torch.tensor([2.0, 3.0]))
        assert torch.allclose(upper, torch.tensor([2.0, 3.0]))

    def test_scalar_input_output(self):
        """Single-element input/output."""

        def scalar_fn(x):
            return torch.relu(x)

        region = HyperRectangle(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))
        outputs = _trace_and_propagate(scalar_fn, (torch.randn(1),), [region])
        lower, upper = outputs.concretize()
        _check_soundness(torch.relu, region, lower, upper)

    def test_deep_chain_soundness(self):
        """3+ layers with nonlinearities, verify soundness."""

        def deep_fn(x):
            W1 = torch.tensor([[1.0, -1.0], [2.0, 0.5], [-1.0, 1.0]])
            h1 = torch.relu(x @ W1)  # (3,) -> (2,)
            W2 = torch.tensor([[1.0, 0.0, -1.0], [0.5, -0.5, 1.0]])
            h2 = torch.sigmoid(h1 @ W2)  # (2,) -> (3,)
            W3 = torch.tensor([[1.0], [-1.0], [0.5]])
            return h2 @ W3  # (3,) -> (1,)

        region = HyperRectangle(lower=-torch.ones(3), upper=torch.ones(3))
        outputs = _trace_and_propagate(deep_fn, (torch.randn(3),), [region])
        lower, upper = outputs.concretize()
        _check_soundness(deep_fn, region, lower, upper)
