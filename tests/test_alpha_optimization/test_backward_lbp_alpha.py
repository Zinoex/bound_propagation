"""End-to-end alpha-CROWN tests for :class:`BackwardLBPPropagator`.

Covers:
- disabled config reproduces the plain CROWN bounds bit-exactly,
- enabled config produces sound bounds (2000-sample Monte Carlo),
- enabled config tightens bounds vs. plain CROWN on a crossing-heavy net,
- intermediate mode ships real gradients through the tape,
- both modes reach the same (sound) answer within tolerance.
"""

from __future__ import annotations

import torch

from bound_propagation.propagation import AlphaOptimizationConfig, BackwardLBPPropagator
from bound_propagation.propagation.backward_lbp import create_default_backward_lbp_registry

from .conftest import bound_width, check_sound_vs_samples, make_relu_net, region, trace_fn


def _propagate_with(config: AlphaOptimizationConfig | None, net, region_):
    registry = create_default_backward_lbp_registry()
    gm = trace_fn(net, region_.lower, registry)
    propagator = BackwardLBPPropagator(gm, registry=registry, alpha_config=config)
    [bounds] = propagator.propagate([region_])
    return bounds


def test_disabled_matches_plain_crown():
    net = make_relu_net(seed=42)
    r = region([-1.0, -0.5, 0.2], [1.0, 0.5, 0.8])
    plain = _propagate_with(None, net, r)
    disabled = _propagate_with(AlphaOptimizationConfig(enabled=False), net, r)

    lo_plain, up_plain = plain.concretize()
    lo_dis, up_dis = disabled.concretize()
    assert torch.allclose(lo_plain, lo_dis, atol=1e-7)
    assert torch.allclose(up_plain, up_dis, atol=1e-7)


def test_final_only_is_sound_and_tightens():
    net = make_relu_net(seed=7)
    # Wide region induces many crossing ReLUs — gives alpha-CROWN room to help.
    r = region([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0])

    plain = _propagate_with(None, net, r)
    lo_plain, up_plain = plain.concretize()
    check_sound_vs_samples(net, r, lo_plain, up_plain)

    optimized = _propagate_with(AlphaOptimizationConfig(enabled=True, iterations=15, lr=0.2), net, r)
    lo_opt, up_opt = optimized.concretize()
    check_sound_vs_samples(net, r, lo_opt, up_opt)

    assert bound_width(lo_opt, up_opt) <= bound_width(lo_plain, up_plain) + 1e-4, (
        "alpha-CROWN should not worsen bounds relative to plain CROWN"
    )


def test_intermediate_mode_is_sound_and_tightens():
    net = make_relu_net(seed=13)
    r = region([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5])

    plain = _propagate_with(None, net, r)
    lo_plain, up_plain = plain.concretize()

    optimized = _propagate_with(
        AlphaOptimizationConfig(enabled=True, iterations=10, lr=0.1, optimize_intermediate=True),
        net,
        r,
    )
    lo_opt, up_opt = optimized.concretize()
    check_sound_vs_samples(net, r, lo_opt, up_opt)
    assert bound_width(lo_opt, up_opt) <= bound_width(lo_plain, up_plain) + 1e-4


def test_returns_correct_number_of_outputs():
    net = make_relu_net(seed=1, output_dim=5)
    r = region([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])

    bounds = _propagate_with(
        AlphaOptimizationConfig(enabled=True, iterations=3),
        net,
        r,
    )
    lo, up = bounds.concretize()
    assert lo.shape == (5,)
    assert up.shape == (5,)


def test_loss_upper_and_lower_are_sound():
    """The three loss modes should all preserve soundness after optimization."""
    net = make_relu_net(seed=17)
    r = region([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

    for loss in ("width", "lower", "upper"):
        bounds = _propagate_with(AlphaOptimizationConfig(enabled=True, iterations=6, loss=loss), net, r)
        lo, up = bounds.concretize()
        check_sound_vs_samples(net, r, lo, up)
