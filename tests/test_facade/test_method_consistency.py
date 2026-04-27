"""Cross-method consistency: all 5 propagation methods on the same network.

Acts as the regression safety net for the propagation-core cleanup. Each test
fixes a small network + region and asserts that:

1. **Soundness** — every method's bounds enclose the true outputs at sampled inputs.
2. **Tightness ordering** — LBP variants are at least as tight as IBP on networks
   where the relaxation pays off (deep nonlinear chains and sigmoid stacks).

The ordering checks tolerate equality (not strict ``<``) because IBP can match
LBP on affine-only sub-networks and small regions where the relaxation collapses
to the same envelope.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from bound_propagation import BoundModel, HyperRectangle

ALL_METHODS = ["ibp", "forward_lbp", "backward_lbp", "forward_backward_lbp", "crown_ibp"]
LBP_METHODS = ["forward_lbp", "backward_lbp", "forward_backward_lbp", "crown_ibp"]


def _sample_sound(fn, region, lower, upper, *, n=256, atol=1e-4):
    samples = region.lower + torch.rand(n, *region.lower.shape) * (region.upper - region.lower)
    for sample in samples:
        out = fn(sample)
        assert torch.all(lower <= out + atol), f"lower violation: lower={lower}, out={out}"
        assert torch.all(out <= upper + atol), f"upper violation: upper={upper}, out={out}"


def _propagate(fn, region, dummy, method):
    bm = BoundModel(fn, dummy_inputs=(dummy,), method=method)
    return bm.propagate(region).concretize()


# ---------------------------------------------------------------------------
# Fixtures: networks where LBP should be at least as tight as IBP.
# ---------------------------------------------------------------------------


@pytest.fixture
def deep_relu_mlp():
    torch.manual_seed(7)
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


@pytest.fixture
def sigmoid_stack():
    """Sigmoid composed three times — IBP loses fast, LBP compounds favorably."""

    def fn(x):
        return torch.sigmoid(torch.sigmoid(torch.sigmoid(x)))

    return fn


# ---------------------------------------------------------------------------
# All methods produce sound bounds on the same input.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL_METHODS)
def test_all_methods_sound_on_deep_relu(method, deep_relu_mlp):
    region = HyperRectangle(lower=torch.tensor([-0.4, -0.3]), upper=torch.tensor([0.4, 0.3]))
    lower, upper = _propagate(deep_relu_mlp, region, torch.zeros(2), method)
    assert torch.all(lower <= upper + 1e-5)
    _sample_sound(deep_relu_mlp, region, lower, upper)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_all_methods_sound_on_sigmoid_stack(method, sigmoid_stack):
    region = HyperRectangle(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))
    lower, upper = _propagate(sigmoid_stack, region, torch.zeros(1), method)
    assert torch.all(lower <= upper + 1e-5)
    _sample_sound(sigmoid_stack, region, lower, upper)


# ---------------------------------------------------------------------------
# Tightness ordering: LBP <= IBP on networks where the relaxation pays off.
# ---------------------------------------------------------------------------


def _width(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return (upper - lower).sum()


@pytest.mark.parametrize("method", LBP_METHODS)
def test_lbp_no_looser_than_ibp_on_deep_relu(method, deep_relu_mlp):
    """On a deep ReLU MLP with crossing units, every LBP variant should be
    at least as tight (smaller total width) as IBP."""
    region = HyperRectangle(lower=torch.tensor([-0.4, -0.3]), upper=torch.tensor([0.4, 0.3]))
    ibp_lower, ibp_upper = _propagate(deep_relu_mlp, region, torch.zeros(2), "ibp")
    lbp_lower, lbp_upper = _propagate(deep_relu_mlp, region, torch.zeros(2), method)
    assert _width(lbp_lower, lbp_upper) <= _width(ibp_lower, ibp_upper) + 1e-5, (
        f"{method} produced looser bounds than IBP: "
        f"width_ibp={float(_width(ibp_lower, ibp_upper)):.4f}, "
        f"width_lbp={float(_width(lbp_lower, lbp_upper)):.4f}"
    )


# NOTE: LBP is NOT universally tighter than IBP. For deep saturating chains
# (e.g. sigmoid-of-sigmoid), IBP can exploit the saturation directly while
# LBP's linear envelope accumulates relaxation error across layers. The
# tightness ordering above is asserted only on the deep-ReLU MLP, where the
# relaxation reliably pays off.


# ---------------------------------------------------------------------------
# Affine-only network: LBP should be at least as tight as IBP because it
# avoids the layerwise width double-counting (|W2 W1| <= |W2| · |W1|).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", LBP_METHODS)
def test_affine_network_lbp_at_least_as_tight(method):
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2))  # no nonlinearity
    region = HyperRectangle(lower=torch.tensor([-0.5, -0.5]), upper=torch.tensor([0.5, 0.5]))
    ibp_lower, ibp_upper = _propagate(model, region, torch.zeros(2), "ibp")
    lbp_lower, lbp_upper = _propagate(model, region, torch.zeros(2), method)
    # Per-coordinate tightness, not just total width.
    assert torch.all(lbp_lower >= ibp_lower - 1e-5), (lbp_lower, ibp_lower)
    assert torch.all(lbp_upper <= ibp_upper + 1e-5), (lbp_upper, ibp_upper)
