"""Full-workflow forward-LBP tests for Conv2d / AvgPool2d / MaxPool2d."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bound_propagation.bounds import LinearBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import ForwardLBPPropagator
from bound_propagation.propagation.forward_lbp import create_default_forward_lbp_registry
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
