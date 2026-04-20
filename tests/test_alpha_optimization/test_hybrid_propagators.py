"""End-to-end alpha-CROWN tests for CROWN-IBP and ForwardBackward propagators.

Also covers the ``optimize_intermediate=True`` rejection for both hybrids
and for :class:`ForwardLBPPropagator`.
"""

from __future__ import annotations

import pytest
import torch

from bound_propagation.propagation import (
    AlphaOptimizationConfig,
    CROWNIBPPropagator,
    ForwardBackwardLBPPropagator,
    ForwardLBPPropagator,
)
from bound_propagation.propagation.backward_lbp import create_default_backward_lbp_registry
from bound_propagation.propagation.forward_lbp import create_default_forward_lbp_registry
from bound_propagation.propagation.ibp import create_default_ibp_registry

from .conftest import bound_width, check_sound_vs_samples, make_relu_net, region, trace_fn


def _crown_ibp_propagate(config: AlphaOptimizationConfig | None, net, region_):
    ibp_registry = create_default_ibp_registry()
    bwd_registry = create_default_backward_lbp_registry()
    gm = trace_fn(net, region_.lower, bwd_registry)
    prop = CROWNIBPPropagator(gm, ibp_registry=ibp_registry, backward_registry=bwd_registry, alpha_config=config)
    b = prop.propagate([region_])
    return b


def _forward_backward_propagate(config: AlphaOptimizationConfig | None, net, region_):
    fwd_registry = create_default_forward_lbp_registry()
    bwd_registry = create_default_backward_lbp_registry()
    gm = trace_fn(net, region_.lower, bwd_registry)
    prop = ForwardBackwardLBPPropagator(
        gm, forward_registry=fwd_registry, backward_registry=bwd_registry, alpha_config=config
    )
    b = prop.propagate([region_])
    return b


def _forward_propagate(config: AlphaOptimizationConfig | None, net, region_):
    fwd_registry = create_default_forward_lbp_registry()
    gm = trace_fn(net, region_.lower, fwd_registry)
    prop = ForwardLBPPropagator(gm, registry=fwd_registry, alpha_config=config)
    b = prop.propagate([region_])
    return b


class TestCrownIbpAlpha:
    def test_disabled_matches_plain(self):
        net = make_relu_net(seed=3)
        r = region([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        plain = _crown_ibp_propagate(None, net, r)
        disabled = _crown_ibp_propagate(AlphaOptimizationConfig(enabled=False), net, r)
        lo1, up1 = plain.concretize()
        lo2, up2 = disabled.concretize()
        assert torch.allclose(lo1, lo2, atol=1e-7)
        assert torch.allclose(up1, up2, atol=1e-7)

    def test_enabled_is_sound_and_tightens(self):
        net = make_relu_net(seed=5)
        r = region([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5])
        plain = _crown_ibp_propagate(None, net, r)
        lo_plain, up_plain = plain.concretize()

        optimized = _crown_ibp_propagate(AlphaOptimizationConfig(enabled=True, iterations=10, lr=0.2), net, r)
        lo_opt, up_opt = optimized.concretize()
        check_sound_vs_samples(net, r, lo_opt, up_opt)
        assert bound_width(lo_opt, up_opt) <= bound_width(lo_plain, up_plain) + 1e-4

    def test_rejects_optimize_intermediate(self):
        net = make_relu_net(seed=1)
        r = region([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="CROWNIBPPropagator does not support"):
            _crown_ibp_propagate(AlphaOptimizationConfig(enabled=True, optimize_intermediate=True), net, r)


class TestForwardBackwardAlpha:
    def test_disabled_matches_plain(self):
        net = make_relu_net(seed=11)
        r = region([-0.8, -0.8, -0.8], [0.8, 0.8, 0.8])
        plain = _forward_backward_propagate(None, net, r)
        disabled = _forward_backward_propagate(AlphaOptimizationConfig(enabled=False), net, r)
        lo1, up1 = plain.concretize()
        lo2, up2 = disabled.concretize()
        assert torch.allclose(lo1, lo2, atol=1e-7)
        assert torch.allclose(up1, up2, atol=1e-7)

    def test_enabled_is_sound_and_tightens(self):
        net = make_relu_net(seed=21)
        r = region([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5])
        plain = _forward_backward_propagate(None, net, r)
        lo_plain, up_plain = plain.concretize()

        optimized = _forward_backward_propagate(AlphaOptimizationConfig(enabled=True, iterations=10, lr=0.2), net, r)
        lo_opt, up_opt = optimized.concretize()
        check_sound_vs_samples(net, r, lo_opt, up_opt)
        assert bound_width(lo_opt, up_opt) <= bound_width(lo_plain, up_plain) + 1e-4

    def test_rejects_optimize_intermediate(self):
        net = make_relu_net(seed=1)
        r = region([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="ForwardBackwardLBPPropagator does not support"):
            _forward_backward_propagate(AlphaOptimizationConfig(enabled=True, optimize_intermediate=True), net, r)


class TestForwardLBPAlpha:
    def test_disabled_matches_plain(self):
        net = make_relu_net(seed=23)
        r = region([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        plain = _forward_propagate(None, net, r)
        disabled = _forward_propagate(AlphaOptimizationConfig(enabled=False), net, r)
        lo1, up1 = plain.concretize()
        lo2, up2 = disabled.concretize()
        assert torch.allclose(lo1, lo2, atol=1e-7)
        assert torch.allclose(up1, up2, atol=1e-7)

    def test_enabled_is_sound(self):
        net = make_relu_net(seed=29)
        r = region([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        optimized = _forward_propagate(AlphaOptimizationConfig(enabled=True, iterations=5, lr=0.1), net, r)
        lo_opt, up_opt = optimized.concretize()
        check_sound_vs_samples(net, r, lo_opt, up_opt)

    def test_rejects_optimize_intermediate(self):
        net = make_relu_net(seed=1)
        r = region([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="ForwardLBPPropagator does not support"):
            _forward_propagate(AlphaOptimizationConfig(enabled=True, optimize_intermediate=True), net, r)
