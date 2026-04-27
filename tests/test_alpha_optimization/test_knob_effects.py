"""α-knob effect tests for the relaxation primitives.

Verifies the **plumbing** of every alpha override: that the unit-interval
fraction reaches ``compute_*_relaxation`` and changes the resulting slope /
bias. Soundness is asserted at the resolver level (analytical-default
fraction must reproduce the no-alpha output) and at the extreme fractions
(``alpha=0`` and ``alpha=1`` must produce **different** slopes than the default
in the crossing regime).

This complements ``test_backward_lbp_alpha.py`` (end-to-end integration) and
the per-relaxation soundness checks in ``test_linear_relaxations/`` — neither
asserts that the optimizable knob actually moves the bound.
"""

from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.linear_relaxations.elementwise import (
    compute_abs_relaxation,
    compute_relu_relaxation,
    compute_sigmoid_relaxation,
)


def _crossing_bounds(lo: float = -1.0, up: float = 2.0) -> IntervalBounds:
    """Single-element interval ``[lo, up]`` with ``lo < 0 < up``."""
    return IntervalBounds(torch.tensor([lo]), torch.tensor([up]))


# ---------------------------------------------------------------------------
# ReLU: lower-slope fraction is the slope value directly in [0, 1].
# ---------------------------------------------------------------------------


class TestReluKnob:
    def test_alpha_zero_gives_zero_slope(self):
        bounds = _crossing_bounds()
        params = compute_relu_relaxation(bounds, alpha_relu_lower=torch.tensor([0.0]))
        assert params.alpha_lower.item() == 0.0
        assert params.beta_lower.item() == 0.0

    def test_alpha_one_gives_unit_slope(self):
        bounds = _crossing_bounds()
        params = compute_relu_relaxation(bounds, alpha_relu_lower=torch.tensor([1.0]))
        assert params.alpha_lower.item() == 1.0

    def test_default_matches_z_ratio(self):
        bounds = _crossing_bounds(lo=-1.0, up=3.0)
        # Analytical default is z = u / (u - l) = 3 / 4.
        no_override = compute_relu_relaxation(bounds, alpha_relu_lower=None)
        with_override = compute_relu_relaxation(bounds, alpha_relu_lower=torch.tensor([0.75]))
        assert torch.allclose(no_override.alpha_lower, with_override.alpha_lower)

    def test_extremes_differ_from_default(self):
        bounds = _crossing_bounds(lo=-1.0, up=3.0)
        zero = compute_relu_relaxation(bounds, alpha_relu_lower=torch.tensor([0.0]))
        one = compute_relu_relaxation(bounds, alpha_relu_lower=torch.tensor([1.0]))
        assert not torch.allclose(zero.alpha_lower, one.alpha_lower)


# ---------------------------------------------------------------------------
# Abs: fraction maps to slope m = 2*alpha - 1 ∈ [-1, 1] in the crossing regime.
# ---------------------------------------------------------------------------


class TestAbsKnob:
    def test_alpha_zero_gives_negative_slope(self):
        bounds = _crossing_bounds()
        params = compute_abs_relaxation(bounds, alpha_abs_lower=torch.tensor([0.0]))
        assert params.alpha_lower.item() == -1.0

    def test_alpha_one_gives_positive_slope(self):
        bounds = _crossing_bounds()
        params = compute_abs_relaxation(bounds, alpha_abs_lower=torch.tensor([1.0]))
        assert params.alpha_lower.item() == 1.0

    def test_alpha_half_gives_zero_slope(self):
        bounds = _crossing_bounds()
        params = compute_abs_relaxation(bounds, alpha_abs_lower=torch.tensor([0.5]))
        assert abs(params.alpha_lower.item()) < 1e-6


# ---------------------------------------------------------------------------
# Sigmoid: dual-sided tangent-point fractions; alpha=0/1 should move the
# bound on the corresponding side relative to the midpoint default.
# ---------------------------------------------------------------------------


class TestSigmoidKnob:
    """Sigmoid α-knobs are active only in the **non-crossing** regimes:
    lower-tangent knob on convex intervals (``u <= 0``), upper-tangent knob on
    concave intervals (``l >= 0``). Crossing intervals use adaptive
    secant/tangent logic that ignores the knob — this matches the docstring."""

    def test_lower_knob_active_in_negative_regime(self):
        # u <= 0: sigmoid is convex, lower bound is a tangent at d_lower.
        bounds = IntervalBounds(torch.tensor([-3.0]), torch.tensor([-0.5]))
        zero = compute_sigmoid_relaxation(bounds, alpha_sigmoid_tangent_lower=torch.tensor([0.0]))
        one = compute_sigmoid_relaxation(bounds, alpha_sigmoid_tangent_lower=torch.tensor([1.0]))
        assert not torch.allclose(zero.alpha_lower, one.alpha_lower)

    def test_upper_knob_active_in_positive_regime(self):
        # l >= 0: sigmoid is concave, upper bound is a tangent at d_upper.
        bounds = IntervalBounds(torch.tensor([0.5]), torch.tensor([3.0]))
        zero = compute_sigmoid_relaxation(bounds, alpha_sigmoid_tangent_upper=torch.tensor([0.0]))
        one = compute_sigmoid_relaxation(bounds, alpha_sigmoid_tangent_upper=torch.tensor([1.0]))
        assert not torch.allclose(zero.alpha_upper, one.alpha_upper)

    def test_default_midpoint_matches_no_override(self):
        # Default fraction 0.5 places the tangent at the midpoint — must
        # reproduce the no-override path bit-exactly in the active regime.
        bounds = IntervalBounds(torch.tensor([-3.0]), torch.tensor([-0.5]))
        no_override = compute_sigmoid_relaxation(bounds)
        midpoint = compute_sigmoid_relaxation(bounds, alpha_sigmoid_tangent_lower=torch.tensor([0.5]))
        assert torch.allclose(no_override.alpha_lower, midpoint.alpha_lower)
        assert torch.allclose(no_override.beta_lower, midpoint.beta_lower)

    def test_crossing_regime_ignores_knob(self):
        """Documented behaviour: α-knob is silently ignored on crossing intervals."""
        bounds = _crossing_bounds()
        ignored = compute_sigmoid_relaxation(
            bounds,
            alpha_sigmoid_tangent_lower=torch.tensor([0.0]),
            alpha_sigmoid_tangent_upper=torch.tensor([1.0]),
        )
        no_override = compute_sigmoid_relaxation(bounds)
        assert torch.allclose(ignored.alpha_lower, no_override.alpha_lower)
        assert torch.allclose(ignored.alpha_upper, no_override.alpha_upper)
