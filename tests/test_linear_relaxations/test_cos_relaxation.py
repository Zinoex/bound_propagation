"""
Tests for cos linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for cos
linear relaxations produce valid upper and lower bounds.

Cos is periodic with period 2π and alternates between convex and concave regions.
"""

import math

import torch

from bound_propagation.propagation.linear_relaxations.elementwise import compute_cos_relaxation


class TestCosRelaxationSoundness:
    """Test that cos linear relaxations are sound (bounds are valid)."""

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
        """Verify that linear bounds are sound by sampling points in [lower, upper]."""
        # Generate sample points in [lower, upper]
        t = torch.linspace(0, 1, num_samples, dtype=lower.dtype, device=lower.device)
        t = t.view(-1, *([1] * lower.ndim))
        x_samples = lower + t * (upper - lower)

        # Compute actual cos values
        cos_vals = torch.cos(x_samples)

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= cos for all samples)
        upper_violations = cos_vals > upper_bound + 1e-5
        if upper_violations.any():
            max_violation = (cos_vals - upper_bound).max().item()
            return False, f"Upper bound violated. Max violation: {max_violation:.6f}"

        # Check lower bound (must be <= cos for all samples)
        lower_violations = cos_vals < lower_bound - 1e-5
        if lower_violations.any():
            max_violation = (lower_bound - cos_vals).max().item()
            return False, f"Lower bound violated. Max violation: {max_violation:.6f}"

        return True, "Bounds are sound"

    def test_concave_region_first_period(self):
        """Test concave region: cos is concave on [0, π]."""
        lower = torch.tensor([0.5])
        upper = torch.tensor([2.5])

        relaxation = compute_cos_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Concave region (first period): {message}"

    def test_convex_region_second_half(self):
        """Test convex region: cos is convex on [π, 2π]."""
        lower = torch.tensor([3.5])
        upper = torch.tensor([5.5])

        relaxation = compute_cos_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Convex region (second half): {message}"

    def test_around_zero(self):
        """Test around zero where cos has maximum."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])

        relaxation = compute_cos_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Around zero: {message}"

    def test_contains_maximum_at_zero(self):
        """Test interval containing maximum at 0."""
        lower = torch.tensor([-0.5])
        upper = torch.tensor([0.5])

        relaxation = compute_cos_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Contains maximum at zero: {message}"

    def test_contains_minimum_at_pi(self):
        """Test interval containing minimum at π."""
        lower = torch.tensor([2.5])
        upper = torch.tensor([3.7])

        relaxation = compute_cos_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Contains minimum at π: {message}"

    def test_full_period(self):
        """Test interval spanning a full period."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([2 * math.pi])

        relaxation = compute_cos_relaxation(lower, upper)
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
        lower = torch.tensor([-2 * math.pi])
        upper = torch.tensor([2 * math.pi])

        relaxation = compute_cos_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Multiple periods: {message}"
        # Should have constant bounds [-1, 1]
        assert torch.allclose(beta_lower, torch.tensor([-1.0]), atol=1e-5)
        assert torch.allclose(beta_upper, torch.tensor([1.0]), atol=1e-5)

    def test_second_period(self):
        """Test region in second period."""
        lower = torch.tensor([2 * math.pi + 0.5])
        upper = torch.tensor([2 * math.pi + 2.5])

        relaxation = compute_cos_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Second period: {message}"

    def test_negative_region(self):
        """Test region with negative x values."""
        lower = torch.tensor([-5.0])
        upper = torch.tensor([-4.0])

        relaxation = compute_cos_relaxation(lower, upper)
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

        relaxation = compute_cos_relaxation(lower, upper)
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
        lower = torch.tensor([[0.5, 3.5, -1.0], [2.5, -5.0, 0.0]])
        upper = torch.tensor([[2.5, 5.5, 1.0], [3.7, -4.0, 2 * math.pi]])

        relaxation = compute_cos_relaxation(lower, upper)
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

    def test_crossing_inflection_point(self):
        """Test interval crossing π/2 (inflection point)."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([2.0])

        relaxation = compute_cos_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing inflection point: {message}"


class TestCosRelaxationPeriodicity:
    """Test periodicity properties of cos relaxation."""

    def test_shifted_by_period(self):
        """Test that relaxations are consistent across periods."""
        lower1 = torch.tensor([0.5])
        upper1 = torch.tensor([1.5])

        # Shift by one period
        lower2 = lower1 + 2 * math.pi
        upper2 = upper1 + 2 * math.pi

        r = compute_cos_relaxation(lower1, upper1)
        alpha_lower1 = r.alpha_lower
        beta_lower1 = r.beta_lower
        alpha_upper1 = r.alpha_upper
        beta_upper1 = r.beta_upper
        r = compute_cos_relaxation(lower2, upper2)
        alpha_lower2 = r.alpha_lower
        beta_lower2 = r.beta_lower
        alpha_upper2 = r.alpha_upper
        beta_upper2 = r.beta_upper

        # Alphas should be the same (slopes)
        assert torch.allclose(alpha_lower1, alpha_lower2, atol=1e-5)
        assert torch.allclose(alpha_upper1, alpha_upper2, atol=1e-5)

        # Verify both are sound
        def verify_sound(lower, upper, al, bl, au, bu):
            x = torch.linspace(lower.item(), upper.item(), 100)
            cos_x = torch.cos(x)
            upper_bound = au * x + bu
            lower_bound = al * x + bl
            return (cos_x <= upper_bound + 1e-5).all() and (cos_x >= lower_bound - 1e-5).all()

        assert verify_sound(lower1, upper1, alpha_lower1, beta_lower1, alpha_upper1, beta_upper1)
        assert verify_sound(lower2, upper2, alpha_lower2, beta_lower2, alpha_upper2, beta_upper2)

    def test_phase_shift_from_sin(self):
        """Test that cos(x) = sin(x + π/2)."""
        x_val = torch.tensor([1.5])
        assert torch.allclose(torch.cos(x_val), torch.sin(x_val + math.pi / 2), atol=1e-5)
