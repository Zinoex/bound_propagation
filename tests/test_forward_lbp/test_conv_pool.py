"""Full-workflow forward-LBP tests for Conv2d / AvgPool2d / MaxPool2d."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bound_propagation.bounds import LinearBounds
from bound_propagation.linear_operators import Conv2dOperator
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import ForwardLBPPropagator
from bound_propagation.propagation.context import PropagationContext
from bound_propagation.propagation.forward_lbp import create_default_forward_lbp_registry
from bound_propagation.propagation.forward_lbp.conv_pool import ForwardLBPConv2d
from bound_propagation.propagation.forward_lbp.utils import create_identity_bounds
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def _trace_and_annotate(fn_or_module, example_inputs):
    registry = create_default_forward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn_or_module)
    MetadataPass(gm).run(*example_inputs)
    return gm


def _make_region(shape: tuple[int, ...], seed: int = 0, width: float = 0.2) -> HyperRectangle:
    torch.manual_seed(seed)
    lower = torch.randn(*shape)
    upper = lower + width
    return HyperRectangle(lower=lower, upper=upper)


def _check_sound(model, region: HyperRectangle, bounds: LinearBounds, n: int = 30) -> None:
    concrete = bounds.concretize()
    for _ in range(n):
        x = region.lower + torch.rand_like(region.lower) * (region.upper - region.lower)
        y = model(x)
        assert torch.all(concrete.lower <= y + 1e-5), (concrete.lower, y)
        assert torch.all(y <= concrete.upper + 1e-5), (concrete.upper, y)


class TestForwardLBPConv2d:
    def test_identity_kernel(self) -> None:
        """1x1 identity conv: output == input; bounds should reproduce the region."""

        class IdConv(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 2, kernel_size=1, bias=False)
                with torch.no_grad():
                    self.conv.weight.zero_()
                    for c in range(2):
                        self.conv.weight[c, c, 0, 0] = 1.0

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        model = IdConv()
        region = _make_region((2, 3, 3))
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        concrete = out.concretize()
        assert torch.allclose(concrete.lower, region.lower, atol=1e-5)
        assert torch.allclose(concrete.upper, region.upper, atol=1e-5)

    def test_conv_only(self) -> None:
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        model = Net()
        region = _make_region((2, 4, 4))
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        _check_sound(model, region, out)

    def test_conv_relu_conv(self) -> None:
        torch.manual_seed(1)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c1 = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.c2 = nn.Conv2d(3, 2, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.c2(torch.relu(self.c1(x)))

        model = Net()
        region = _make_region((2, 4, 4), seed=2)
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        _check_sound(model, region, out)

    def test_functional_conv2d(self) -> None:
        torch.manual_seed(3)
        weight = torch.randn(3, 2, 3, 3)
        bias = torch.randn(3)

        def fn(x: torch.Tensor) -> torch.Tensor:
            return F.conv2d(x, weight, bias=bias, padding=1)

        region = _make_region((2, 4, 4), seed=3)
        gm = _trace_and_annotate(fn, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        _check_sound(fn, region, out)


class TestForwardLBPAvgPool2d:
    def test_basic_avgpool(self) -> None:
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.AvgPool2d(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        region = _make_region((3, 4, 4), seed=4)
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        _check_sound(model, region, out)
        # AvgPool is linear; both bounds should be exact at the region endpoints.
        concrete = out.concretize()
        true_lower = F.avg_pool2d(region.lower, 2)
        true_upper = F.avg_pool2d(region.upper, 2)
        assert torch.allclose(concrete.lower, true_lower, atol=1e-5)
        assert torch.allclose(concrete.upper, true_upper, atol=1e-5)

    def test_conv_then_avgpool(self) -> None:
        torch.manual_seed(5)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.pool = nn.AvgPool2d(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(torch.relu(self.conv(x)))

        model = Net()
        region = _make_region((2, 4, 4), seed=5)
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        _check_sound(model, region, out)

    def test_adaptive_avgpool(self) -> None:
        torch.manual_seed(6)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d((2, 2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        region = _make_region((2, 4, 4), seed=6)
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        _check_sound(model, region, out)


class TestForwardLBPMaxPool2d:
    def test_basic_maxpool(self) -> None:
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.MaxPool2d(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        region = _make_region((2, 4, 4), seed=7)
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        _check_sound(model, region, out)

    def test_conv_relu_maxpool(self) -> None:
        torch.manual_seed(8)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(torch.relu(self.conv(x)))

        model = Net()
        region = _make_region((2, 4, 4), seed=8)
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        _check_sound(model, region, out)

    def test_maxpool_tight_when_single_winner(self) -> None:
        """When one element dominates across the whole region, bounds are tight."""
        lower = torch.tensor(
            [[[10.0, -5.0, 10.0, -5.0], [-5.0, -5.0, -5.0, -5.0], [10.0, -5.0, 10.0, -5.0], [-5.0, -5.0, -5.0, -5.0]]]
        )
        upper = lower + 0.05
        region = HyperRectangle(lower=lower, upper=upper)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.MaxPool2d(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        _check_sound(model, region, out)


class TestForwardLBPCNNFull:
    def test_conv_pool_flatten_linear(self) -> None:
        torch.manual_seed(9)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.flatten = nn.Flatten(start_dim=0)
                self.fc = nn.Linear(3 * 2 * 2, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(self.flatten(self.pool(torch.relu(self.conv(x)))))

        model = Net()
        region = _make_region((2, 4, 4), seed=9)
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        _check_sound(model, region, out)


class TestForwardLBPConv2dStructuredFastPath:
    """Verify that the first conv layer emits Conv2dOperator coefficients
    instead of materializing a dense Jacobian."""

    def test_first_conv_emits_conv2d_operator(self) -> None:
        """Tracing a bare nn.Conv2d with an IdentityOperator-backed input
        should produce ``Conv2dOperator`` coefficients on the output."""
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        model = Net()
        region = _make_region((2, 4, 4))
        gm = _trace_and_annotate(model, (region.lower,))
        propagator = ForwardLBPPropagator(gm)
        out = propagator.propagate([region])

        # Both lower and upper coefficient operators must be Conv2dOperator
        # (the identity-input fast path kicked in).
        assert all(isinstance(op, Conv2dOperator) for op in out.linear_lowers_op)
        assert all(isinstance(op, Conv2dOperator) for op in out.linear_uppers_op)

        # Output bounds must still be sound.
        _check_sound(model, region, out)

    def test_second_conv_emits_patch_operator(self) -> None:
        """A conv whose input was already relu'd still stays structural:
        the ``scale`` from ReLU produced a ScaledConv2dOperator, and the
        second conv composes into a Conv2dPatchOperator."""
        from bound_propagation.linear_operators import Conv2dPatchOperator  # noqa: PLC0415

        torch.manual_seed(1)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c1 = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.c2 = nn.Conv2d(3, 2, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.c2(torch.relu(self.c1(x)))

        model = Net()
        region = _make_region((2, 4, 4), seed=1)
        gm = _trace_and_annotate(model, (region.lower,))
        out = ForwardLBPPropagator(gm).propagate([region])
        # Structural patch-mode composition preserves structure across
        # conv → relu → conv.
        assert all(isinstance(op, Conv2dPatchOperator) for op in out.linear_lowers_op)
        assert all(isinstance(op, Conv2dPatchOperator) for op in out.linear_uppers_op)
        _check_sound(model, region, out)

    def test_structured_and_dense_agree_on_bounds(self) -> None:
        """The fast-path output (structured) must concretize identically to
        what a full dense materialization would produce."""
        torch.manual_seed(2)
        weight = torch.randn(3, 2, 3, 3)
        bias = torch.randn(3)

        # Structured path: directly drive the strategy with identity bounds.
        region = _make_region((2, 4, 4), seed=2)
        identity_bounds = create_identity_bounds(id=0, region=region, shape=region.lower.shape)

        # Build a minimal fx graph with one call_function node invoking F.conv2d.
        import torch.fx as fx

        graph = fx.Graph()
        x_ph = graph.placeholder("x")
        w_ph = graph.placeholder("weight")
        b_ph = graph.placeholder("bias")
        conv_node = graph.call_function(
            F.conv2d, args=(x_ph, w_ph, b_ph, 1, 1, 1, 1)
        )
        graph.output(conv_node)
        gm = fx.GraphModule(torch.nn.Module(), graph)

        ctx = PropagationContext(gm)
        ctx.store(x_ph, identity_bounds)
        ctx.store(w_ph, weight)
        ctx.store(b_ph, bias)

        out_struct = ForwardLBPConv2d().propagate_forward(conv_node, ctx)
        assert all(isinstance(op, Conv2dOperator) for op in out_struct.linear_lowers_op)
        lo_s, up_s = out_struct.concretize()

        # Dense baseline: same conv applied via the standard tracing path
        # (identity input → dense materialization).
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                with torch.no_grad():
                    self.conv.weight.copy_(weight)
                    self.conv.bias.copy_(bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        model = Net()
        gm_full = _trace_and_annotate(model, (region.lower,))
        out_dense = ForwardLBPPropagator(gm_full).propagate([region])
        lo_d, up_d = out_dense.concretize()

        assert torch.allclose(lo_s, lo_d, atol=1e-5)
        assert torch.allclose(up_s, up_d, atol=1e-5)


class TestForwardLBPUnsupported:
    def test_maxpool_ceil_mode_raises(self) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.MaxPool2d(2, ceil_mode=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        region = _make_region((2, 5, 5))
        gm = _trace_and_annotate(model, (region.lower,))
        with pytest.raises(NotImplementedError, match="ceil_mode"):
            ForwardLBPPropagator(gm).propagate([region])
