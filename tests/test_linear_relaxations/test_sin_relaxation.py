"""
Tests for sin linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for sin
linear relaxations produce valid upper and lower bounds.

Sin is periodic with period 2π and alternates between convex and concave regions.
"""

import math

import torch

from bound_propagation.propagation.linear_relaxations.elementwise import compute_sin_relaxation


class TestSinRelaxationSoundness:
    """Test that sin linear relaxations are sound (bounds are valid)."""

    def verify_bounds_sound(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha_lower: torch.Tensor,
        beta_lower: torch.Tensor,
        alpha_upper: torch.Tensor,
        beta_upper: torch.Tensor,
        num_samples: int = 1000,
    ):
        """
        Verify that linear bounds are sound by sampling points in [lower, upper].

        Returns (is_sound, message).
        """
        # Generate sample points in [lower, upper]
        t = torch.linspace(0, 1, num_samples, dtype=lower.dtype, device=lower.device)
        t = t.view(-1, *([1] * lower.ndim))
        x_samples = lower + t * (upper - lower)

        # Compute actual sin values
        sin_vals = torch.sin(x_samples)

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= sin for all samples)
        upper_violations = sin_vals > upper_bound + 1e-5
        if upper_violations.any():
            max_violation = (sin_vals - upper_bound).max().item()
            return False, f"Upper bound violated. Max violation: {max_violation:.6f}"

        # Check lower bound (must be <= sin for all samples)
        lower_violations = sin_vals < lower_bound - 1e-5
        if lower_violations.any():
            max_violation = (lower_bound - sin_vals).max().item()
            return False, f"Lower bound violated. Max violation: {max_violation:.6f}"

        return True, "Bounds are sound"

    def test_convex_region_first_period(self):
        """Test convex region: sin is convex on [-π, 0]."""
        lower = torch.tensor([-2.5])
        upper = torch.tensor([-0.5])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Convex region (first period): {message}"

    def test_concave_region_first_period(self):
        """Test concave region: sin is concave on [0, π]."""
        lower = torch.tensor([0.5])
        upper = torch.tensor([2.5])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Concave region (first period): {message}"

    def test_crossing_zero(self):
        """Test crossing zero (inflection point between convex and concave)."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero: {message}"

    def test_contains_maximum(self):
        """Test interval containing maximum at π/2."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([2.0])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Contains maximum: {message}"

    def test_contains_minimum(self):
        """Test interval containing minimum at 3π/2."""
        lower = torch.tensor([4.0])
        upper = torch.tensor([5.5])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Contains minimum: {message}"

    def test_full_period(self):
        """Test interval spanning a full period."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([2 * math.pi])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Full period: {message}"
        # Should have constant bounds [-1, 1]
        assert torch.allclose(beta_lower, torch.tensor([-1.0]), atol=1e-5)
        assert torch.allclose(beta_upper, torch.tensor([1.0]), atol=1e-5)

    def test_multiple_periods(self):
        """Test interval spanning multiple periods."""
        lower = torch.tensor([-math.pi])
        upper = torch.tensor([3 * math.pi])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Multiple periods: {message}"
        # Should have constant bounds [-1, 1]
        assert torch.allclose(beta_lower, torch.tensor([-1.0]), atol=1e-5)
        assert torch.allclose(beta_upper, torch.tensor([1.0]), atol=1e-5)

    def test_second_period_convex(self):
        """Test convex region in second period."""
        lower = torch.tensor([math.pi + 0.5])
        upper = torch.tensor([2 * math.pi - 0.5])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Second period convex: {message}"

    def test_negative_region(self):
        """Test region with large negative x values."""
        lower = torch.tensor([-10.0])
        upper = torch.tensor([-9.0])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Negative region: {message}"

    def test_zero_width_interval(self):
        """Test zero-width interval."""
        lower = torch.tensor([1.5])
        upper = torch.tensor([1.5])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width: {message}"

    def test_batch_mixed_regimes(self):
        """Test batch with mixed regimes."""
        lower = torch.tensor([[-2.5, 0.5, 1.0], [-1.0, 4.0, -10.0]])
        upper = torch.tensor([[-0.5, 2.5, 2.0], [1.0, 5.5, -9.0]])

        relaxation = compute_sin_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        # Test each element separately
        for i in range(lower.shape[0]):
            for j in range(lower.shape[1]):
                lo = lower[i, j : j + 1]
                up = upper[i, j : j + 1]
                al = alpha_lower[i, j : j + 1]
                bl = beta_lower[i, j : j + 1]
                au = alpha_upper[i, j : j + 1]
                bu = beta_upper[i, j : j + 1]

                is_sound, message = self.verify_bounds_sound(lo, up, al, bl, au, bu)
                assert is_sound, f"Batch element [{i},{j}]: {message}"

    def test_narrow_around_extrema(self):
        """Test narrow intervals around maximum and minimum."""
        # Around maximum π/2 ≈ 1.5708
        lower_max = torch.tensor([1.4])
        upper_max = torch.tensor([1.7])

        relaxation = compute_sin_relaxation(lower_max, upper_max)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper
        is_sound, message = self.verify_bounds_sound(
            lower_max, upper_max, alpha_lower, beta_lower, alpha_upper, beta_upper
        )
        assert is_sound, f"Narrow around maximum: {message}"

        # Around minimum 3π/2 ≈ 4.7124
        lower_min = torch.tensor([4.5])
        upper_min = torch.tensor([4.9])

        relaxation = compute_sin_relaxation(lower_min, upper_min)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper
        is_sound, message = self.verify_bounds_sound(
            lower_min, upper_min, alpha_lower, beta_lower, alpha_upper, beta_upper
        )
        assert is_sound, f"Narrow around minimum: {message}"


class TestSinRelaxationPeriodicity:
    """Test periodicity properties of sin relaxation."""

    def test_shifted_by_period(self):
        """Test that relaxations are consistent across periods."""
        lower1 = torch.tensor([0.5])
        upper1 = torch.tensor([1.5])

        # Shift by one period
        lower2 = lower1 + 2 * math.pi
        upper2 = upper1 + 2 * math.pi

        r = compute_sin_relaxation(lower1, upper1)
        alpha_lower1 = r.alpha_lower
        beta_lower1 = r.beta_lower
        alpha_upper1 = r.alpha_upper
        beta_upper1 = r.beta_upper
        r = compute_sin_relaxation(lower2, upper2)
        alpha_lower2 = r.alpha_lower
        beta_lower2 = r.beta_lower
        alpha_upper2 = r.alpha_upper
        beta_upper2 = r.beta_upper

        # Alphas should be the same (slopes)
        assert torch.allclose(alpha_lower1, alpha_lower2, atol=1e-5)
        assert torch.allclose(alpha_upper1, alpha_upper2, atol=1e-5)

        # Betas might differ due to shifted x, but the bounds should be sound
        # Just verify both are sound (main test)
        def verify_sound(lower, upper, al, bl, au, bu):
            x = torch.linspace(lower.item(), upper.item(), 100)
            sin_x = torch.sin(x)
            upper_bound = au * x + bu
            lower_bound = al * x + bl
            return (sin_x <= upper_bound + 1e-5).all() and (sin_x >= lower_bound - 1e-5).all()

        assert verify_sound(lower1, upper1, alpha_lower1, beta_lower1, alpha_upper1, beta_upper1)
        assert verify_sound(lower2, upper2, alpha_lower2, beta_lower2, alpha_upper2, beta_upper2)
