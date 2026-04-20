"""End-to-end Forward-Backward LBP workflow tests."""

from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import (
    BackwardLBPPropagator,
    CROWNIBPPropagator,
    ForwardBackwardLBPPropagator,
    IBPPropagator,
)
from bound_propagation.propagation.backward_lbp import create_default_backward_lbp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer

from .conftest import check_soundness, propagate_forward_backward, trace_and_annotate


class TestForwardBackwardIdentityAndLinear:
    """Linear-only networks: Forward-Backward should match exact interval bounds."""

    def test_identity(self) -> None:
        def fn(x):
            return x

        region = HyperRectangle(
            lower=torch.tensor([1.0, 2.0, 3.0]),
            upper=torch.tensor([4.0, 5.0, 6.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        assert isinstance(bounds, LinearBounds)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, region.lower)
        assert torch.allclose(upper, region.upper)

    def test_affine(self) -> None:
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = torch.tensor([1.0, -1.0])

        def fn(x):
            return x @ w + b

        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.tensor([1.0, -1.0]))
        assert torch.allclose(upper, torch.tensor([10.0, 11.0]))


class TestForwardBackwardNonlinear:
    """Nonlinear networks: verify soundness via sampling."""

    def test_relu(self) -> None:
        def fn(x):
            return torch.relu(x)

        region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0, -2.0]),
            upper=torch.tensor([3.0, 3.0, 3.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_sigmoid(self) -> None:
        def fn(x):
            return torch.sigmoid(x)

        region = HyperRectangle(
            lower=torch.tensor([-2.0, 0.0]),
            upper=torch.tensor([2.0, 3.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_tanh(self) -> None:
        def fn(x):
            return torch.tanh(x)

        region = HyperRectangle(
            lower=torch.tensor([-1.5, -0.5]),
            upper=torch.tensor([1.5, 0.5]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_two_layer_network(self) -> None:
        def fn(x):
            w1 = torch.tensor([[1.0, -2.0], [3.0, 1.0], [-1.0, 2.0]])
            h = torch.relu(x @ w1)
            w2 = torch.tensor([[1.0], [-1.0]])
            return h @ w2

        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)


class TestForwardBackwardVsOtherMethods:
    """Relationships between Forward-Backward LBP, CROWN, CROWN-IBP, and IBP."""

    def test_matches_ibp_on_linear_only(self) -> None:
        """With no nonlinearities, Forward-Backward's bounds match IBP exactly."""
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = torch.tensor([1.0, -1.0])

        def fn(x):
            return x @ w + b

        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        example = torch.randn(3)

        registry = create_default_backward_lbp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(example)

        fb_lb = ForwardBackwardLBPPropagator(gm).propagate([region])
        ibp_b = IBPPropagator(gm).propagate([region])

        fb_lo, fb_hi = fb_lb.concretize()
        assert torch.allclose(fb_lo, ibp_b.lower, atol=1e-5)
        assert torch.allclose(fb_hi, ibp_b.upper, atol=1e-5)

    def test_crown_tighter_than_forward_backward(self) -> None:
        """Standard CROWN uses recursive backward intermediate bounds, which
        are tighter-or-equal to forward LBP concretization, so CROWN's
        output is tighter-or-equal to Forward-Backward's."""

        def fn(x):
            w1 = torch.tensor([[1.0, -2.0], [3.0, 1.0], [-1.0, 2.0]])
            h = torch.relu(x @ w1)
            w2 = torch.tensor([[1.0], [-1.0]])
            return h @ w2

        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        example = torch.randn(3)

        registry = create_default_backward_lbp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(example)

        crown_lb = BackwardLBPPropagator(gm).propagate([region])
        fb_lb = ForwardBackwardLBPPropagator(gm).propagate([region])

        c_lo, c_hi = crown_lb.concretize()
        fb_lo, fb_hi = fb_lb.concretize()

        atol = 1e-5
        assert torch.all(c_lo >= fb_lo - atol)
        assert torch.all(c_hi <= fb_hi + atol)

    def test_matches_crown_ibp_on_linear_only(self) -> None:
        """With no nonlinearities, Forward-Backward and CROWN-IBP agree
        (both concretize to the same IBP bounds)."""
        w = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b = torch.tensor([1.0, -1.0])

        def fn(x):
            return x @ w + b

        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        example = torch.randn(3)

        registry = create_default_backward_lbp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(fn)
        MetadataPass(gm).run(example)

        fb_lb = ForwardBackwardLBPPropagator(gm).propagate([region])
        crown_ibp_lb = CROWNIBPPropagator(gm).propagate([region])

        fb_lo, fb_hi = fb_lb.concretize()
        ci_lo, ci_hi = crown_ibp_lb.concretize()

        assert torch.allclose(fb_lo, ci_lo, atol=1e-5)
        assert torch.allclose(fb_hi, ci_hi, atol=1e-5)


class TestForwardBackwardReturnsLinearBounds:
    def test_output_type(self) -> None:
        def fn(x):
            return torch.relu(x)

        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        gm = trace_and_annotate(fn, torch.randn(2))
        outputs = ForwardBackwardLBPPropagator(gm).propagate([region])
        assert isinstance(outputs, LinearBounds)


class TestForwardBackwardComplexNonlinearities:
    """Pairwise and reduction nonlinearities."""

    def test_pairwise_maximum_both_abstract(self) -> None:
        def fn(x):
            return torch.maximum(x[:2], x[2:])

        region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0, -2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0, 1.0, 4.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_pairwise_maximum_with_constant(self) -> None:
        """y = max(x, 0): effectively ReLU via maximum."""
        zero = torch.zeros(3)

        def fn(x):
            return torch.maximum(x, zero)

        region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, -3.0]),
            upper=torch.tensor([3.0, 2.0, 1.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_pairwise_minimum_both_abstract(self) -> None:
        def fn(x):
            return torch.minimum(x[:2], x[2:])

        region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0, -2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0, 1.0, 4.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_amax_reduction_sound(self) -> None:
        """y = amax(relu(x @ W)): reduction over a nonlinear layer."""

        def fn(x):
            w = torch.tensor([[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]])
            h = torch.relu(x @ w)
            return torch.amax(h)

        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_amin_reduction_sound(self) -> None:
        def fn(x):
            return torch.amin(torch.sigmoid(x))

        region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.0, 1.0]),
            upper=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_pairwise_mul_both_abstract(self) -> None:
        def fn(x):
            return x[:2] * x[2:]

        region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.5, -2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0, 1.0, 4.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_reciprocal_near_asymptote(self) -> None:
        """y = 1/x on [0.1, 2]: steep gradient near the x=0 asymptote."""

        def fn(x):
            return torch.reciprocal(x)

        region = HyperRectangle(
            lower=torch.tensor([0.1, 0.25, 0.5]),
            upper=torch.tensor([2.0, 3.0, 4.0]),
        )
        bounds = propagate_forward_backward(fn, region, example_input=torch.rand(3) + 0.5)
        check_soundness(fn, region, bounds)

    def test_log_near_asymptote(self) -> None:
        """y = log(x): region's lower edge near the x=0 asymptote."""

        def fn(x):
            return torch.log(x)

        region = HyperRectangle(
            lower=torch.tensor([0.05, 0.1]),
            upper=torch.tensor([1.0, 2.0]),
        )
        bounds = propagate_forward_backward(fn, region, example_input=torch.rand(2) + 0.1)
        check_soundness(fn, region, bounds)

    def test_div_constant_over_abstract_near_asymptote(self) -> None:
        """y = 1 / (x + 0.1): constant/abstract division near an asymptote."""
        shift = torch.tensor([0.1, 0.1])

        def fn(x):
            return 1.0 / (x + shift)

        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0]),
            upper=torch.tensor([2.0, 2.0]),
        )
        bounds = propagate_forward_backward(fn, region, example_input=torch.rand(2) + 0.5)
        check_soundness(fn, region, bounds)


class TestForwardBackwardMultiInput:
    """Functions that take multiple input tensors."""

    def test_two_input_add(self) -> None:
        def fn(x, y):
            return x + y

        registry = create_default_backward_lbp_registry()
        gm = BoundPropagationTracer(registry).trace(fn)
        MetadataPass(gm).run(torch.randn(3), torch.randn(3))
        propagator = ForwardBackwardLBPPropagator(gm)

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
        def fn(x, y):
            return x - y

        registry = create_default_backward_lbp_registry()
        gm = BoundPropagationTracer(registry).trace(fn)
        MetadataPass(gm).run(torch.randn(2), torch.randn(2))
        propagator = ForwardBackwardLBPPropagator(gm)

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
        assert torch.allclose(lower, torch.tensor([2.0, 6.0]))
        assert torch.allclose(upper, torch.tensor([6.0, 10.0]))

    def test_two_input_relu_then_combine(self) -> None:
        """y = relu(x) + sigmoid(y): two inputs through different activations."""

        def fn(x, y):
            return torch.relu(x) + torch.sigmoid(y)

        registry = create_default_backward_lbp_registry()
        gm = BoundPropagationTracer(registry).trace(fn)
        MetadataPass(gm).run(torch.randn(2), torch.randn(2))
        propagator = ForwardBackwardLBPPropagator(gm)

        x_region = HyperRectangle(
            lower=torch.tensor([-2.0, 1.0]),
            upper=torch.tensor([2.0, 3.0]),
        )
        y_region = HyperRectangle(
            lower=torch.tensor([-1.0, 0.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        outputs = propagator.propagate([x_region, y_region])

        linear_bounds = outputs
        lower, upper = linear_bounds.concretize()
        num_samples = 1000
        x_rand = x_region.lower + torch.rand(num_samples, *x_region.lower.shape) * (x_region.upper - x_region.lower)
        y_rand = y_region.lower + torch.rand(num_samples, *y_region.lower.shape) * (y_region.upper - y_region.lower)
        for xs, ys in zip(x_rand, y_rand, strict=True):
            out = fn(xs, ys)
            assert torch.all(lower <= out + 1e-4)
            assert torch.all(out <= upper + 1e-4)


class TestForwardBackwardDAG:
    """Non-tree graphs: multiple paths from input to output."""

    def test_diamond_linear_paths(self) -> None:
        """y = 2*x + 3*x: two linear paths, should be exactly 5*x."""

        def fn(x):
            return x * 2.0 + x * 3.0

        region = HyperRectangle(
            lower=torch.tensor([1.0, -1.0, 0.0]),
            upper=torch.tensor([2.0, 1.0, 3.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.tensor([5.0, -5.0, 0.0]))
        assert torch.allclose(upper, torch.tensor([10.0, 5.0, 15.0]))

    def test_diamond_cancellation_exact(self) -> None:
        """y = x - x: should cancel to zero."""

        def fn(x):
            return x - x

        region = HyperRectangle(
            lower=torch.tensor([-2.0, 1.0, 0.5]),
            upper=torch.tensor([3.0, 4.0, 2.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.zeros(3), atol=1e-5)
        assert torch.allclose(upper, torch.zeros(3), atol=1e-5)

    def test_diamond_relu_plus_identity(self) -> None:
        """y = relu(x) + x: diamond where one path is nonlinear."""

        def fn(x):
            return torch.relu(x) + x

        region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.5]),
            upper=torch.tensor([2.0, 3.0, 4.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_diamond_two_linear_layers_shared_input(self) -> None:
        """y = (x @ W1) + (x @ W2): two affine paths from same input."""

        def fn(x):
            w1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            w2 = torch.tensor([[-1.0, 0.0], [0.5, -0.5], [1.0, 1.0]])
            return x @ w1 + x @ w2

        region = HyperRectangle(
            lower=torch.tensor([0.0, 0.0, 0.0]),
            upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        assert torch.allclose(bounds.linear_lowers[0], bounds.linear_uppers[0], atol=1e-5)
        check_soundness(fn, region, bounds)

    def test_diamond_relu_sigmoid_sum(self) -> None:
        """y = relu(x) + sigmoid(x): two nonlinear paths from same input."""

        def fn(x):
            return torch.relu(x) + torch.sigmoid(x)

        region = HyperRectangle(
            lower=torch.tensor([-2.0, 0.0]),
            upper=torch.tensor([2.0, 3.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)


class TestForwardBackwardDeepNonlinearDAG:
    """DAGs with stacked nonlinearities around branch points."""

    def test_nonlinear_prebranch_two_nonlinear_branches(self) -> None:
        W0 = torch.tensor([[1.0, -0.5, 0.5], [0.5, 1.0, -1.0]])
        b0 = torch.tensor([0.1, -0.1, 0.2])
        Wa = torch.tensor([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0]])
        Wb = torch.tensor([[-1.0, 0.5], [0.5, 1.0], [1.0, -0.5]])

        def fn(x):
            z = torch.tanh(torch.sigmoid(torch.relu(x @ W0 + b0)))
            a = torch.sigmoid(torch.relu(z @ Wa))
            b = torch.relu(torch.tanh(z @ Wb))
            return torch.sigmoid(a + b)

        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_three_branches_nonlinear_each(self) -> None:
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

        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_residual_skip_with_nonlinearities(self) -> None:
        W1 = torch.tensor([[1.0, -0.5, 0.5, 1.0], [0.5, 1.0, -1.0, -0.5]])
        W2 = torch.tensor([[0.5, -1.0], [1.0, 0.5], [-0.5, 1.0], [0.5, 0.5]])
        Ws = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-0.5, 0.5]])

        def fn(x):
            h1 = torch.relu(x @ W1)
            h2 = torch.sigmoid(torch.tanh(h1 @ W2))
            skip = torch.relu(h1 @ Ws)
            return torch.tanh(h2 + skip)

        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_nested_diamond_nonlinear(self) -> None:
        def fn(x):
            p = torch.sigmoid(x)
            outer_a = torch.tanh(p)
            inner_a = torch.relu(p)
            inner_b = torch.sigmoid(p)
            outer_b = inner_a * 0.5 + inner_b * 0.5
            return torch.relu(outer_a + outer_b)

        region = HyperRectangle(
            lower=torch.tensor([-2.0, -1.0, 0.0]),
            upper=torch.tensor([1.0, 2.0, 3.0]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)

    def test_parallel_nonlinear_towers_then_deep_merge(self) -> None:
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

        region = HyperRectangle(
            lower=torch.tensor([-0.75, -0.75]),
            upper=torch.tensor([0.75, 0.75]),
        )
        bounds = propagate_forward_backward(fn, region)
        check_soundness(fn, region, bounds)


class TestForwardBackwardEdgeCases:
    """Degenerate / boundary inputs."""

    def test_zero_width_region_identity(self) -> None:
        def fn(x):
            return x

        point = torch.tensor([2.0, -1.0, 3.5])
        region = HyperRectangle(lower=point, upper=point)
        bounds = propagate_forward_backward(fn, region)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, point)
        assert torch.allclose(upper, point)

    def test_zero_width_region_relu(self) -> None:
        def fn(x):
            return torch.relu(x)

        point = torch.tensor([-2.0, 0.0, 3.0])
        region = HyperRectangle(lower=point, upper=point)
        bounds = propagate_forward_backward(fn, region)
        lower, upper = bounds.concretize()
        expected = torch.relu(point)
        assert torch.allclose(lower, expected, atol=1e-5)
        assert torch.allclose(upper, expected, atol=1e-5)

    def test_zero_width_region_through_network(self) -> None:
        def fn(x):
            w1 = torch.tensor([[1.0, -2.0], [3.0, 1.0]])
            h = torch.relu(x @ w1)
            w2 = torch.tensor([[1.0], [-1.0]])
            return h @ w2

        point = torch.tensor([0.5, -0.3])
        region = HyperRectangle(lower=point, upper=point)
        bounds = propagate_forward_backward(fn, region)
        lower, upper = bounds.concretize()
        expected = fn(point)
        assert torch.allclose(lower, expected, atol=1e-5)
        assert torch.allclose(upper, expected, atol=1e-5)

    def test_zero_width_at_relu_kink(self) -> None:
        """Zero-width region at x=0 should give bounds of 0."""

        def fn(x):
            return torch.relu(x)

        point = torch.zeros(3)
        region = HyperRectangle(lower=point, upper=point)
        bounds = propagate_forward_backward(fn, region)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.zeros(3), atol=1e-5)
        assert torch.allclose(upper, torch.zeros(3), atol=1e-5)
