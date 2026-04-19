"""End-to-end CROWN-IBP workflow tests."""

from __future__ import annotations

import torch

from bound_propagation.bounds import LinearBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import BackwardLBPPropagator, CROWNIBPPropagator, IBPPropagator
from bound_propagation.propagation.backward_lbp import create_default_backward_lbp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer

from .conftest import check_soundness, propagate_crown_ibp, trace_and_annotate


class TestCROWNIBPIdentityAndLinear:
    """Linear-only networks: CROWN-IBP should match exact interval bounds."""

    def test_identity(self) -> None:
        def fn(x):
            return x

        region = HyperRectangle(
            lower=torch.tensor([1.0, 2.0, 3.0]),
            upper=torch.tensor([4.0, 5.0, 6.0]),
        )
        bounds = propagate_crown_ibp(fn, region)
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
        bounds = propagate_crown_ibp(fn, region)
        lower, upper = bounds.concretize()
        assert torch.allclose(lower, torch.tensor([1.0, -1.0]))
        assert torch.allclose(upper, torch.tensor([10.0, 11.0]))


class TestCROWNIBPNonlinear:
    """Nonlinear networks: verify soundness via sampling."""

    def test_relu(self) -> None:
        def fn(x):
            return torch.relu(x)

        region = HyperRectangle(
            lower=torch.tensor([-2.0, -2.0, -2.0]),
            upper=torch.tensor([3.0, 3.0, 3.0]),
        )
        bounds = propagate_crown_ibp(fn, region)
        check_soundness(fn, region, bounds)

    def test_sigmoid(self) -> None:
        def fn(x):
            return torch.sigmoid(x)

        region = HyperRectangle(
            lower=torch.tensor([-2.0, 0.0]),
            upper=torch.tensor([2.0, 3.0]),
        )
        bounds = propagate_crown_ibp(fn, region)
        check_soundness(fn, region, bounds)

    def test_tanh(self) -> None:
        def fn(x):
            return torch.tanh(x)

        region = HyperRectangle(
            lower=torch.tensor([-1.5, -0.5]),
            upper=torch.tensor([1.5, 0.5]),
        )
        bounds = propagate_crown_ibp(fn, region)
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
        bounds = propagate_crown_ibp(fn, region)
        check_soundness(fn, region, bounds)


class TestCROWNIBPVsCROWN:
    """Standard CROWN uses recursive backward concretization for intermediate
    bounds, which is tighter-or-equal to IBP. Since CROWN-IBP uses the looser
    IBP intermediate bounds, CROWN's output is tighter-or-equal to CROWN-IBP's."""

    def test_crown_tighter_than_crown_ibp(self) -> None:
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

        crown_lb = BackwardLBPPropagator(gm).propagate([region])[0]
        crown_ibp_lb = CROWNIBPPropagator(gm).propagate([region])[0]

        c_lo, c_hi = crown_lb.concretize()
        ci_lo, ci_hi = crown_ibp_lb.concretize()

        atol = 1e-5
        assert torch.all(c_lo >= ci_lo - atol)
        assert torch.all(c_hi <= ci_hi + atol)

    def test_crown_ibp_matches_ibp_on_linear_only(self) -> None:
        """With no nonlinearities, CROWN-IBP's linear bound concretizes to IBP bounds."""

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

        crown_ibp_lb = CROWNIBPPropagator(gm).propagate([region])[0]
        ibp_b = IBPPropagator(gm).propagate([region])[0]

        ci_lo, ci_hi = crown_ibp_lb.concretize()
        assert torch.allclose(ci_lo, ibp_b.lower, atol=1e-5)
        assert torch.allclose(ci_hi, ibp_b.upper, atol=1e-5)


class TestCROWNIBPReturnsLinearBounds:
    def test_output_type(self) -> None:
        def fn(x):
            return torch.relu(x)

        region = HyperRectangle(
            lower=torch.tensor([-1.0, -1.0]),
            upper=torch.tensor([1.0, 1.0]),
        )
        gm = trace_and_annotate(fn, torch.randn(2))
        outputs = CROWNIBPPropagator(gm).propagate([region])
        assert len(outputs) == 1
        assert isinstance(outputs[0], LinearBounds)
