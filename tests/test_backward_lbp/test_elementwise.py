"""Systematic tests for every element-wise backward LBP strategy.

For each activation function, tests:
1. Soundness (sampling) across different input regimes
2. Exactness in regimes where the function is locally linear
3. Tightness sanity (bounds shouldn't be absurdly loose)
"""

from __future__ import annotations

import torch

from .conftest import assert_exact, assert_sound, region

# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------


class TestBackwardLBPRelu:
    def test_positive_regime_exact(self):
        """ReLU on [1, 3] is identity: bounds must be exact."""
        r = region([1.0, 2.0], [3.0, 5.0])
        assert_exact(lambda x: torch.relu(x), r, r.lower, r.upper)

    def test_negative_regime_exact(self):
        """ReLU on [-3, -1] is zero: bounds must be exact."""
        r = region([-3.0, -2.0], [-1.0, -0.5])
        assert_exact(lambda x: torch.relu(x), r, torch.zeros(2), torch.zeros(2))

    def test_crossing_regime_sound(self):
        """ReLU on [-2, 3]: relaxation must be sound."""
        assert_sound(lambda x: torch.relu(x), region([-2.0, -1.0], [3.0, 4.0]))

    def test_crossing_single_element(self):
        """ReLU crossing on a single element."""
        assert_sound(lambda x: torch.relu(x), region([-1.0], [1.0]))

    def test_narrow_crossing(self):
        """ReLU near zero crossing, very narrow interval."""
        assert_sound(lambda x: torch.relu(x), region([-0.01], [0.01]))


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------


class TestBackwardLBPSigmoid:
    def test_sound_crossing(self):
        assert_sound(lambda x: torch.sigmoid(x), region([-3.0, -1.0], [3.0, 1.0]))

    def test_sound_positive(self):
        """Sigmoid in deep positive regime, should be close to 1."""
        lower, upper = assert_sound(lambda x: torch.sigmoid(x), region([5.0], [10.0]))
        assert lower.item() > 0.99

    def test_sound_negative(self):
        """Sigmoid in deep negative regime, should be close to 0."""
        lower, upper = assert_sound(lambda x: torch.sigmoid(x), region([-10.0], [-5.0]))
        assert upper.item() < 0.01

    def test_zero_centered(self):
        assert_sound(lambda x: torch.sigmoid(x), region([-0.5, 0.0], [0.5, 1.0]))


# ---------------------------------------------------------------------------
# Tanh
# ---------------------------------------------------------------------------


class TestBackwardLBPTanh:
    def test_sound_crossing(self):
        assert_sound(lambda x: torch.tanh(x), region([-2.0, -1.0], [2.0, 1.0]))

    def test_deep_positive(self):
        lower, upper = assert_sound(lambda x: torch.tanh(x), region([3.0], [5.0]))
        assert lower.item() > 0.99

    def test_deep_negative(self):
        lower, upper = assert_sound(lambda x: torch.tanh(x), region([-5.0], [-3.0]))
        assert upper.item() < -0.99


# ---------------------------------------------------------------------------
# Exp
# ---------------------------------------------------------------------------


class TestBackwardLBPExp:
    def test_sound_positive(self):
        assert_sound(lambda x: torch.exp(x), region([0.0, 1.0], [1.0, 2.0]))

    def test_sound_negative(self):
        assert_sound(lambda x: torch.exp(x), region([-3.0], [-1.0]))

    def test_sound_crossing(self):
        assert_sound(lambda x: torch.exp(x), region([-1.0, -2.0], [1.0, 2.0]))

    def test_narrow_interval(self):
        lower, upper = assert_sound(lambda x: torch.exp(x), region([1.0], [1.01]))
        # Narrow interval should give tight bounds
        assert (upper - lower).item() < 0.1


# ---------------------------------------------------------------------------
# Log
# ---------------------------------------------------------------------------


class TestBackwardLBPLog:
    def test_sound_above_one(self):
        assert_sound(lambda x: torch.log(x), region([1.0, 2.0], [3.0, 5.0]))

    def test_sound_below_one(self):
        assert_sound(lambda x: torch.log(x), region([0.1, 0.5], [0.5, 0.9]))

    def test_sound_spanning_one(self):
        assert_sound(lambda x: torch.log(x), region([0.5], [2.0]))


# ---------------------------------------------------------------------------
# Sqrt
# ---------------------------------------------------------------------------


class TestBackwardLBPSqrt:
    def test_sound(self):
        assert_sound(lambda x: torch.sqrt(x), region([1.0, 4.0], [4.0, 9.0]))

    def test_near_zero(self):
        assert_sound(lambda x: torch.sqrt(x), region([0.01], [1.0]))


# ---------------------------------------------------------------------------
# Reciprocal
# ---------------------------------------------------------------------------


class TestBackwardLBPReciprocal:
    def test_sound_positive(self):
        assert_sound(lambda x: torch.reciprocal(x), region([1.0, 2.0], [3.0, 5.0]))

    def test_sound_negative(self):
        assert_sound(lambda x: torch.reciprocal(x), region([-5.0, -3.0], [-2.0, -1.0]))


# ---------------------------------------------------------------------------
# Abs
# ---------------------------------------------------------------------------


class TestBackwardLBPAbs:
    def test_positive_regime_exact(self):
        """abs on [1, 3] is identity: bounds must be exact."""
        r = region([1.0, 2.0], [3.0, 5.0])
        assert_exact(lambda x: torch.abs(x), r, r.lower, r.upper)

    def test_negative_regime_exact(self):
        """abs on [-3, -1] is negation: bounds must be exact."""
        r = region([-3.0, -2.0], [-1.0, -0.5])
        assert_exact(lambda x: torch.abs(x), r, torch.tensor([1.0, 0.5]), torch.tensor([3.0, 2.0]))

    def test_crossing_regime_sound(self):
        assert_sound(lambda x: torch.abs(x), region([-2.0, -1.0], [3.0, 4.0]))


# ---------------------------------------------------------------------------
# Sin
# ---------------------------------------------------------------------------


class TestBackwardLBPSin:
    def test_sound_small_range(self):
        assert_sound(lambda x: torch.sin(x), region([0.0, 0.5], [1.0, 1.5]))

    def test_sound_crossing_zero(self):
        assert_sound(lambda x: torch.sin(x), region([-1.0], [1.0]))


# ---------------------------------------------------------------------------
# Cos
# ---------------------------------------------------------------------------


class TestBackwardLBPCos:
    def test_sound_small_range(self):
        assert_sound(lambda x: torch.cos(x), region([0.0, 0.5], [1.0, 1.5]))

    def test_sound_crossing_zero(self):
        assert_sound(lambda x: torch.cos(x), region([-1.0], [1.0]))


# ---------------------------------------------------------------------------
# Tan
# ---------------------------------------------------------------------------


class TestBackwardLBPTan:
    def test_sound_small_range(self):
        """Tan in a safe range (away from pi/2)."""
        assert_sound(lambda x: torch.tan(x), region([0.0, -0.5], [0.5, 0.5]))


# ---------------------------------------------------------------------------
# Pow
# ---------------------------------------------------------------------------


class TestBackwardLBPPow:
    """``BackwardLBPPow`` only supports integer ``n == 2`` today (produced
    by the ``x*x -> pow(x, 2)`` rewrite). Both crossing and non-crossing
    regimes must produce sound bounds."""

    def test_sound_positive(self):
        assert_sound(lambda x: x**2, region([1.0, 2.0], [3.0, 4.0]))

    def test_sound_negative(self):
        assert_sound(lambda x: x**2, region([-3.0, -2.0], [-1.0, -0.5]))

    def test_sound_crossing_zero(self):
        assert_sound(lambda x: x**2, region([-2.0, -1.0], [2.0, 1.0]))

    def test_narrow_interval(self):
        lower, upper = assert_sound(lambda x: x**2, region([1.0], [1.01]))
        assert (upper - lower).item() < 0.1


# ---------------------------------------------------------------------------
# Clamp
# ---------------------------------------------------------------------------


class TestBackwardLBPClamp:
    def test_clamp_min_only(self):
        """clamp(x, min=0) on [-2, 3] is like ReLU."""
        assert_sound(lambda x: torch.clamp(x, min=0.0), region([-2.0, -1.0], [3.0, 4.0]))

    def test_clamp_max_only(self):
        assert_sound(lambda x: torch.clamp(x, max=1.0), region([-1.0, 0.0], [2.0, 3.0]))

    def test_clamp_both(self):
        assert_sound(lambda x: torch.clamp(x, min=-1.0, max=1.0), region([-3.0], [3.0]))

    def test_clamp_inactive(self):
        """Clamp where all values are within bounds: exact identity."""
        r = region([0.5, 0.5], [1.5, 1.5])
        assert_exact(lambda x: torch.clamp(x, min=0.0, max=2.0), r, r.lower, r.upper)

    def test_clamp_fully_active_min(self):
        """Clamp where all values are below min."""
        r = region([-3.0, -2.0], [-1.0, -0.5])
        assert_exact(
            lambda x: torch.clamp(x, min=0.0),
            r,
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
        )
