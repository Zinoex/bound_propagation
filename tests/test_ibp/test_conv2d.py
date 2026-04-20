"""Unit tests for IBP conv2d strategy."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.conv_pool import IBPConv2d
from tests.helpers import propagate


def _make_bounds(shape: tuple[int, ...], seed: int = 0) -> IntervalBounds:
    torch.manual_seed(seed)
    lower = torch.randn(*shape)
    upper = lower + torch.rand(*shape) + 0.1
    return IntervalBounds(lower, upper)


class TestIBPConv2dFunctional:
    def test_basic_no_padding(self) -> None:
        bounds = _make_bounds((1, 3, 5, 5))
        weight = torch.randn(4, 3, 3, 3)
        bias = torch.randn(4)

        out = propagate(IBPConv2d(), bounds, weight, bias, 1, 0, 1, 1)

        assert isinstance(out, IntervalBounds)
        assert out.shape == (1, 4, 3, 3)

        weight_pos = weight.clamp(min=0)
        weight_neg = weight.clamp(max=0)
        expected_lower = (
            F.conv2d(bounds.lower, weight_pos) + F.conv2d(bounds.upper, weight_neg) + bias.view(1, -1, 1, 1)
        )
        expected_upper = (
            F.conv2d(bounds.upper, weight_pos) + F.conv2d(bounds.lower, weight_neg) + bias.view(1, -1, 1, 1)
        )

        assert torch.allclose(out.lower, expected_lower)
        assert torch.allclose(out.upper, expected_upper)

    def test_with_padding_stride(self) -> None:
        bounds = _make_bounds((2, 3, 8, 8))
        weight = torch.randn(6, 3, 3, 3)
        bias = torch.randn(6)

        out = propagate(IBPConv2d(), bounds, weight, bias, 2, 1, 1, 1)
        assert out.shape == (2, 6, 4, 4)

    def test_no_bias(self) -> None:
        bounds = _make_bounds((1, 2, 4, 4))
        weight = torch.randn(3, 2, 3, 3)
        out = propagate(IBPConv2d(), bounds, weight, None, 1, 0, 1, 1)
        assert out.shape == (1, 3, 2, 2)

    def test_sampling_soundness(self) -> None:
        torch.manual_seed(42)
        bounds = _make_bounds((1, 2, 4, 4))
        weight = torch.randn(3, 2, 3, 3)
        bias = torch.randn(3)

        out = propagate(IBPConv2d(), bounds, weight, bias, 1, 0, 1, 1)

        for _ in range(30):
            x = bounds.lower + torch.rand_like(bounds.lower) * (bounds.upper - bounds.lower)
            y = F.conv2d(x, weight, bias=bias)
            assert torch.all(out.lower <= y + 1e-6)
            assert torch.all(out.upper >= y - 1e-6)

    def test_lower_le_upper(self) -> None:
        bounds = _make_bounds((1, 3, 6, 6), seed=7)
        weight = torch.randn(5, 3, 3, 3)
        out = propagate(IBPConv2d(), bounds, weight, None, 1, 1, 1, 1)
        assert torch.all(out.lower <= out.upper + 1e-6)


class TestIBPConv2dModule:
    """Exercise the call_module path via full workflow tracing."""

    def test_nn_conv2d_matches_functional(self) -> None:
        torch.manual_seed(3)
        from bound_propagation.passes import MetadataPass
        from bound_propagation.propagation import IBPPropagator
        from bound_propagation.propagation.ibp import create_default_ibp_registry
        from bound_propagation.regions import HyperRectangle
        from bound_propagation.tracer import BoundPropagationTracer

        conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        bounds = _make_bounds((1, 3, 5, 5))
        region = HyperRectangle(lower=bounds.lower, upper=bounds.upper)

        registry = create_default_ibp_registry()
        tracer = BoundPropagationTracer(registry)
        gm = tracer.trace(conv)
        MetadataPass(gm).run(bounds.lower)
        out = IBPPropagator(gm).propagate([region])

        # Sample soundness
        for _ in range(20):
            x = region.lower + torch.rand_like(region.lower) * (region.upper - region.lower)
            y = conv(x)
            assert torch.all(out.lower <= y + 1e-6)
            assert torch.all(out.upper >= y - 1e-6)


class TestIBPConv2dValidation:
    def test_non_interval_input_raises(self) -> None:
        weight = torch.randn(1, 1, 3, 3)
        with pytest.raises(TypeError):
            propagate(IBPConv2d(), torch.randn(1, 1, 5, 5), weight, None, 1, 0, 1, 1)

    def test_bad_weight_rank_raises(self) -> None:
        bounds = _make_bounds((1, 3, 4, 4))
        bad_weight = torch.randn(3, 3)
        with pytest.raises(ValueError, match="4D"):
            propagate(IBPConv2d(), bounds, bad_weight, None, 1, 0, 1, 1)
