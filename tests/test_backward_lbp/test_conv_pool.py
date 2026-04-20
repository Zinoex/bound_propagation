"""Full-workflow backward-LBP tests for Conv2d / AvgPool2d / MaxPool2d."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bound_propagation.bounds import LinearBounds
from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import BackwardLBPPropagator
from bound_propagation.propagation.backward_lbp import create_default_backward_lbp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def _trace_and_annotate(fn_or_module, example_inputs):
    registry = create_default_backward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(fn_or_module)
    MetadataPass(gm).run(*example_inputs)
    return gm


def _make_region(shape: tuple[int, ...], seed: int = 0) -> HyperRectangle:
    torch.manual_seed(seed)
    lower = torch.randn(*shape)
    upper = lower + torch.rand(*shape) + 0.1
    return HyperRectangle(lower=lower, upper=upper)


def _sample_soundness(fn_or_module, region: HyperRectangle, bounds: LinearBounds, num_samples: int = 20) -> None:
    """Assert the concretized bounds enclose the true output over random samples."""
    concrete = bounds.concretize()
    for _ in range(num_samples):
        x = region.lower + torch.rand_like(region.lower) * (region.upper - region.lower)
        y = fn_or_module(x)
        assert torch.all(concrete.lower <= y + 1e-5), f"Lower bound violated: got {concrete.lower}, actual {y}"
        assert torch.all(concrete.upper >= y - 1e-5), f"Upper bound violated: got {concrete.upper}, actual {y}"


class TestBackwardLBPConv2d:
    def test_identity_when_weight_is_identity(self) -> None:
        # 1x1 identity kernel: output == input.
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
        region = _make_region((1, 2, 3, 3))
        gm = _trace_and_annotate(model, (region.lower,))
        out = BackwardLBPPropagator(gm).propagate([region], batch_ndim=1)
        concrete = out.concretize()
        assert torch.allclose(concrete.lower, region.lower)
        assert torch.allclose(concrete.upper, region.upper)

    def test_conv_then_reshape_then_linear(self) -> None:
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.fc = nn.Linear(3 * 4 * 4, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.conv(x)
                h = torch.reshape(h, (1, 48))
                return self.fc(h)

        model = Net()
        region = _make_region((1, 2, 4, 4))
        gm = _trace_and_annotate(model, (region.lower,))
        out = BackwardLBPPropagator(gm).propagate([region], batch_ndim=1)
        _sample_soundness(model, region, out)

    def test_conv_with_stride(self) -> None:
        torch.manual_seed(1)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        model = Net()
        region = _make_region((1, 1, 4, 4), seed=1)
        gm = _trace_and_annotate(model, (region.lower,))
        out = BackwardLBPPropagator(gm).propagate([region], batch_ndim=1)
        _sample_soundness(model, region, out)

    def test_functional_conv2d(self) -> None:
        torch.manual_seed(2)
        weight = torch.randn(3, 2, 3, 3)
        bias = torch.randn(3)

        def fn(x: torch.Tensor) -> torch.Tensor:
            return F.conv2d(x, weight, bias=bias, stride=1, padding=1)

        region = _make_region((1, 2, 5, 5), seed=2)
        gm = _trace_and_annotate(fn, (region.lower,))
        out = BackwardLBPPropagator(gm).propagate([region], batch_ndim=1)
        _sample_soundness(fn, region, out)


class TestBackwardLBPAvgPool2d:
    def test_basic_avgpool(self) -> None:
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.AvgPool2d(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        region = _make_region((1, 3, 4, 4))
        gm = _trace_and_annotate(model, (region.lower,))
        out = BackwardLBPPropagator(gm).propagate([region], batch_ndim=1)
        _sample_soundness(model, region, out)
        # AvgPool is linear; bounds should be tight on both endpoints.
        concrete = out.concretize()
        true_lower = F.avg_pool2d(region.lower, 2)
        true_upper = F.avg_pool2d(region.upper, 2)
        assert torch.allclose(concrete.lower, true_lower, atol=1e-5)
        assert torch.allclose(concrete.upper, true_upper, atol=1e-5)

    def test_adaptive_avgpool(self) -> None:
        torch.manual_seed(3)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        region = _make_region((1, 3, 6, 6), seed=3)
        gm = _trace_and_annotate(model, (region.lower,))
        out = BackwardLBPPropagator(gm).propagate([region], batch_ndim=1)
        _sample_soundness(model, region, out)


class TestBackwardLBPMaxPool2d:
    def test_basic_maxpool(self) -> None:
        torch.manual_seed(4)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.MaxPool2d(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        region = _make_region((1, 2, 4, 4), seed=4)
        gm = _trace_and_annotate(model, (region.lower,))
        out = BackwardLBPPropagator(gm).propagate([region], batch_ndim=1)
        _sample_soundness(model, region, out)

    def test_maxpool_tight_when_single_winner(self) -> None:
        """When one element dominates across the whole region, bounds should be tight."""
        torch.manual_seed(5)
        # Construct a region where position (0, 0) always dominates in each 2x2 window.
        lower = torch.tensor(
            [[[[10.0, -5.0, 10.0, -5.0], [-5.0, -5.0, -5.0, -5.0], [10.0, -5.0, 10.0, -5.0], [-5.0, -5.0, -5.0, -5.0]]]]
        )
        upper = lower + 0.1  # Very tight region
        region = HyperRectangle(lower=lower, upper=upper)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.MaxPool2d(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        gm = _trace_and_annotate(model, (region.lower,))
        out = BackwardLBPPropagator(gm).propagate([region], batch_ndim=1)
        _sample_soundness(model, region, out)

    def test_maxpool_cnn(self) -> None:
        """Conv -> ReLU -> MaxPool composition with sampling soundness."""
        torch.manual_seed(6)

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.conv(x)
                h = torch.relu(h)
                return self.pool(h)

        model = Net()
        region = _make_region((1, 2, 4, 4), seed=6)
        gm = _trace_and_annotate(model, (region.lower,))
        out = BackwardLBPPropagator(gm).propagate([region], batch_ndim=1)
        _sample_soundness(model, region, out)


class TestBackwardLBPUnsupported:
    def test_avgpool_ceil_mode_raises(self) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        region = _make_region((1, 2, 5, 5))
        gm = _trace_and_annotate(model, (region.lower,))
        with pytest.raises(NotImplementedError, match="ceil_mode"):
            BackwardLBPPropagator(gm).propagate([region])

    def test_maxpool_ceil_mode_raises(self) -> None:
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.pool(x)

        model = Net()
        region = _make_region((1, 2, 5, 5))
        gm = _trace_and_annotate(model, (region.lower,))
        with pytest.raises(NotImplementedError, match="ceil_mode"):
            BackwardLBPPropagator(gm).propagate([region])
