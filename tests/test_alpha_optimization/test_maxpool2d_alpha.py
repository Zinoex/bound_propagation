"""Alpha-CROWN tests for backward-LBP MaxPool2d winner-vs-IBP interpolation."""

from __future__ import annotations

import torch
import torch.nn as nn

from bound_propagation.passes import MetadataPass
from bound_propagation.propagation import AlphaOptimizationConfig, BackwardLBPPropagator
from bound_propagation.propagation.backward_lbp import create_default_backward_lbp_registry
from bound_propagation.regions import HyperRectangle
from bound_propagation.tracer import BoundPropagationTracer


def _trace_and_annotate(model: nn.Module, dummy: torch.Tensor):
    registry = create_default_backward_lbp_registry()
    tracer = BoundPropagationTracer(registry)
    gm = tracer.trace(model)
    MetadataPass(gm).run(dummy)
    return gm, registry


def _propagate(
    model: nn.Module,
    region: HyperRectangle,
    dummy: torch.Tensor,
    config: AlphaOptimizationConfig | None,
    batch_ndim: int = 1,
):
    gm, registry = _trace_and_annotate(model, dummy)
    propagator = BackwardLBPPropagator(gm, registry=registry, alpha_config=config)
    return propagator.propagate([region], batch_ndim=batch_ndim)


def _check_sound(model, region: HyperRectangle, lower: torch.Tensor, upper: torch.Tensor, n: int = 200):
    for _ in range(n):
        x = region.lower + torch.rand_like(region.lower) * (region.upper - region.lower)
        y = model(x)
        assert torch.all(lower <= y + 1e-5), (lower, y)
        assert torch.all(y <= upper + 1e-5), (upper, y)


def _bound_width(lower: torch.Tensor, upper: torch.Tensor) -> float:
    return float((upper - lower).sum().item())


def _make_maxpool_net() -> nn.Module:
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


class TestMaxPool2dAlphaDisabled:
    def test_disabled_matches_plain_crown(self) -> None:
        model = _make_maxpool_net()
        dummy = torch.zeros(1, 2, 4, 4)
        region = _make_region((1, 2, 4, 4))

        plain = _propagate(model, region, dummy, config=None)
        disabled = _propagate(model, region, dummy, config=AlphaOptimizationConfig(enabled=False))

        lo_p, up_p = plain.concretize()
        lo_d, up_d = disabled.concretize()
        assert torch.allclose(lo_p, lo_d, atol=1e-6)
        assert torch.allclose(up_p, up_d, atol=1e-6)


class TestMaxPool2dAlphaSoundAndTightens:
    def test_enabled_is_sound(self) -> None:
        model = _make_maxpool_net()
        dummy = torch.zeros(1, 2, 4, 4)
        region = _make_region((1, 2, 4, 4), width=1.0, seed=2)

        optimized = _propagate(
            model,
            region,
            dummy,
            config=AlphaOptimizationConfig(enabled=True, iterations=10, lr=0.1),
        )
        lo, up = optimized.concretize()
        _check_sound(model, region, lo, up)

    def test_enabled_does_not_worsen(self) -> None:
        """Alpha-optimized bounds should be at least as tight as the default alpha=1."""
        model = _make_maxpool_net()
        dummy = torch.zeros(1, 2, 4, 4)
        region = _make_region((1, 2, 4, 4), width=1.0, seed=3)

        plain = _propagate(model, region, dummy, config=None)
        lo_p, up_p = plain.concretize()

        optimized = _propagate(
            model,
            region,
            dummy,
            config=AlphaOptimizationConfig(enabled=True, iterations=15, lr=0.2),
        )
        lo_o, up_o = optimized.concretize()

        assert _bound_width(lo_o, up_o) <= _bound_width(lo_p, up_p) + 1e-4, (
            "Optimized bounds should not be looser than plain CROWN"
        )

    def test_manual_alpha_zero_matches_ibp_routing(self) -> None:
        """With alpha=0, the maxpool's contribution should use only constant bounds
        (no routing through the winner). Verify soundness."""
        model = _make_maxpool_net()
        dummy = torch.zeros(1, 2, 4, 4)
        region = _make_region((1, 2, 4, 4), width=0.8, seed=4)

        class ZeroAlphaProvider:
            """Forces all maxpool alphas to 0 (pure IBP fallback on the pool)."""

            def __init__(self) -> None:
                self._cache: dict[tuple[str, str], torch.Tensor] = {}

            def get(self, node, knob_name, shape, init, device, dtype):
                if not knob_name.startswith("maxpool2d_"):
                    return None
                key = (node.name, knob_name)
                if key not in self._cache:
                    self._cache[key] = torch.zeros(shape, device=device, dtype=dtype)
                return self._cache[key]

        gm, registry = _trace_and_annotate(model, dummy)
        propagator = BackwardLBPPropagator(gm, registry=registry)
        # Patch: inject custom provider directly by running _propagate_once through
        # the (private) path. Easiest: temporarily monkeypatch NullAlphaProvider.
        from bound_propagation.propagation.alpha_optimization import NullAlphaProvider  # noqa: PLC0415

        original_get = NullAlphaProvider.get
        provider_instance = ZeroAlphaProvider()
        NullAlphaProvider.get = lambda self, **kwargs: provider_instance.get(**kwargs)  # type: ignore[assignment]
        try:
            bounds = propagator.propagate([region], batch_ndim=1)
        finally:
            NullAlphaProvider.get = original_get  # type: ignore[assignment]

        lo, up = bounds.concretize()
        _check_sound(model, region, lo, up)


class TestMaxPool2dAlphaInCNN:
    def test_deeper_cnn_with_two_maxpools(self) -> None:
        torch.manual_seed(5)
        model = nn.Sequential(
            nn.Conv2d(2, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        dummy = torch.zeros(1, 2, 8, 8)
        region = _make_region((1, 2, 8, 8), width=0.5, seed=6)

        optimized = _propagate(
            model,
            region,
            dummy,
            config=AlphaOptimizationConfig(enabled=True, iterations=10, lr=0.2),
        )
        lo, up = optimized.concretize()
        _check_sound(model, region, lo, up)
