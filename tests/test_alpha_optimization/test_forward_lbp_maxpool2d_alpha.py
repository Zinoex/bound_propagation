"""Alpha-CROWN tests for forward-LBP MaxPool2d winner-vs-IBP interpolation."""

from __future__ import annotations

import torch
import torch.nn as nn

from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import AlphaOptimizationConfig, ForwardLBPPropagator
from bound_propagation.propagation.forward_lbp import create_default_forward_lbp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def _trace_and_annotate(model: nn.Module, dummy: torch.Tensor):
    registry = create_default_forward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(model)
    MetadataPass(gm).run(dummy)
    return gm, registry


def _propagate(
    model: nn.Module,
    region: HyperRectangle,
    dummy: torch.Tensor,
    config: AlphaOptimizationConfig | None,
):
    gm, registry = _trace_and_annotate(model, dummy)
    propagator = ForwardLBPPropagator(gm, registry=registry, alpha_config=config)
    return propagator.propagate([region])


def _check_sound(model, region: HyperRectangle, lower: torch.Tensor, upper: torch.Tensor, n: int = 100):
    for _ in range(n):
        x = region.lower + torch.rand_like(region.lower) * (region.upper - region.lower)
        y = model(x)
        assert torch.all(lower <= y + 1e-5), (lower, y)
        assert torch.all(y <= upper + 1e-5), (upper, y)


def _bound_width(lower: torch.Tensor, upper: torch.Tensor) -> float:
    return float((upper - lower).sum().item())


def _make_net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(2, 3, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


def _make_region(shape: tuple[int, ...], width: float = 0.5, seed: int = 1) -> HyperRectangle:
    torch.manual_seed(seed)
    lower = torch.randn(*shape)
    upper = lower + width
    return HyperRectangle(lower=lower, upper=upper)


class TestForwardLBPMaxPool2dAlpha:
    def test_disabled_matches_plain(self) -> None:
        model = _make_net()
        dummy = torch.zeros(2, 4, 4)
        region = _make_region((2, 4, 4))
        plain = _propagate(model, region, dummy, config=None)
        disabled = _propagate(model, region, dummy, config=AlphaOptimizationConfig(enabled=False))

        lo_p, up_p = plain.concretize()
        lo_d, up_d = disabled.concretize()
        assert torch.allclose(lo_p, lo_d, atol=1e-6)
        assert torch.allclose(up_p, up_d, atol=1e-6)

    def test_enabled_is_sound(self) -> None:
        model = _make_net()
        dummy = torch.zeros(2, 4, 4)
        region = _make_region((2, 4, 4), width=1.0, seed=2)

        optimized = _propagate(
            model, region, dummy, config=AlphaOptimizationConfig(enabled=True, iterations=10, lr=0.1)
        )
        lo, up = optimized.concretize()
        _check_sound(model, region, lo, up)

    def test_enabled_does_not_worsen(self) -> None:
        model = _make_net()
        dummy = torch.zeros(2, 4, 4)
        region = _make_region((2, 4, 4), width=1.0, seed=3)

        plain = _propagate(model, region, dummy, config=None)
        lo_p, up_p = plain.concretize()
        optimized = _propagate(
            model, region, dummy, config=AlphaOptimizationConfig(enabled=True, iterations=15, lr=0.2)
        )
        lo_o, up_o = optimized.concretize()

        assert _bound_width(lo_o, up_o) <= _bound_width(lo_p, up_p) + 1e-4
