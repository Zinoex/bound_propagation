"""Unit tests for IBP avg_pool2d strategy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.conv_pool import IBPAvgPool2d
from tests.helpers import propagate


def _make_bounds(shape: tuple[int, ...], seed: int = 0) -> IntervalBounds:
    torch.manual_seed(seed)
    lower = torch.randn(*shape)
    upper = lower + torch.rand(*shape) + 0.1
    return IntervalBounds(lower, upper)


class TestIBPAvgPool2dFunctional:
    def test_basic(self) -> None:
        bounds = _make_bounds((1, 3, 4, 4))
        out = propagate(IBPAvgPool2d(), bounds, 2)
        assert out.shape == (1, 3, 2, 2)
        assert torch.allclose(out.lower, F.avg_pool2d(bounds.lower, 2))
        assert torch.allclose(out.upper, F.avg_pool2d(bounds.upper, 2))

    def test_with_stride_padding(self) -> None:
        bounds = _make_bounds((1, 2, 6, 6))
        out = propagate(IBPAvgPool2d(), bounds, 3, 2, 1)
        assert torch.allclose(out.lower, F.avg_pool2d(bounds.lower, 3, 2, 1))

    def test_sampling_soundness(self) -> None:
        torch.manual_seed(1)
        bounds = _make_bounds((1, 3, 4, 4))
        out = propagate(IBPAvgPool2d(), bounds, 2)

        for _ in range(30):
            x = bounds.lower + torch.rand_like(bounds.lower) * (bounds.upper - bounds.lower)
            y = F.avg_pool2d(x, 2)
            assert torch.all(out.lower <= y + 1e-6)
            assert torch.all(out.upper >= y - 1e-6)


class TestIBPAvgPool2dModule:
    def test_nn_avgpool_module(self) -> None:
        from bound_propagation.passes import MetadataPass
        from bound_propagation.propagation import IBPPropagator
        from bound_propagation.propagation.ibp import create_default_ibp_registry
        from bound_propagation.regions import HyperRectangle
        from bound_propagation.tracer import BoundPropagationTracer

        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        bounds = _make_bounds((1, 3, 6, 6))
        region = HyperRectangle(lower=bounds.lower, upper=bounds.upper)

        registry = create_default_ibp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(pool)
        MetadataPass(gm).run(bounds.lower)
        out = IBPPropagator(gm).propagate([region])

        assert torch.allclose(out.lower, F.avg_pool2d(bounds.lower, 2))

    def test_adaptive_avgpool_module(self) -> None:
        from bound_propagation.passes import MetadataPass
        from bound_propagation.propagation import IBPPropagator
        from bound_propagation.propagation.ibp import create_default_ibp_registry
        from bound_propagation.regions import HyperRectangle
        from bound_propagation.tracer import BoundPropagationTracer

        pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        bounds = _make_bounds((1, 3, 6, 6))
        region = HyperRectangle(lower=bounds.lower, upper=bounds.upper)

        registry = create_default_ibp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(pool)
        MetadataPass(gm).run(bounds.lower)
        out = IBPPropagator(gm).propagate([region])

        assert out.shape == (1, 3, 2, 2)
        assert torch.allclose(out.lower, F.adaptive_avg_pool2d(bounds.lower, (2, 2)))
