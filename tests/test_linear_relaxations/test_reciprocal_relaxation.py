"""
Tests for reciprocal linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for reciprocal (1/x)
linear relaxations produce valid upper and lower bounds.

Reciprocal is convex on (0, ∞) and convex on (-∞, 0), with a discontinuity at 0.
"""

import torch

from bound_propagation.propagation.linear_relaxations.elementwise import compute_reciprocal_relaxation


class TestReciprocalRelaxationSoundness:
    """Test that reciprocal linear relaxations are sound (bounds are valid)."""

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
        # Check for infinite bounds (valid for crossing zero)
        has_inf_lower = torch.isinf(beta_lower).any()
        has_inf_upper = torch.isinf(beta_upper).any()

        if has_inf_lower or has_inf_upper:
            # For intervals crossing zero, we expect -inf and +inf bounds
            if beta_lower.item() == float("-inf") and beta_upper.item() == float("inf"):
                return True, "Infinite bounds (crosses zero)"
            else:
                return True, "Partial infinite bounds"

        # Generate sample points in [lower, upper]
        t = torch.linspace(0, 1, num_samples, dtype=lower.dtype, device=lower.device)
        t = t.view(-1, *([1] * lower.ndim))
        x_samples = lower + t * (upper - lower)

        # Avoid sampling exactly at zero
        x_samples = torch.where(torch.abs(x_samples) < 1e-7, 1e-7 * torch.sign(x_samples + 1e-7), x_samples)

        # Compute actual reciprocal values
        recip_vals = 1.0 / x_samples

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= 1/x for all samples)
        upper_violations = recip_vals > upper_bound + 1e-4
        if upper_violations.any():
            max_violation = (recip_vals - upper_bound).max().item()
            return False, f"Upper bound violated. Max violation: {max_violation:.6f}"

        # Check lower bound (must be <= 1/x for all samples)
        lower_violations = recip_vals < lower_bound - 1e-4
        if lower_violations.any():
            max_violation = (lower_bound - recip_vals).max().item()
            return False, f"Lower bound violated. Max violation: {max_violation:.6f}"

        return True, "Bounds are sound"

    def test_all_positive_small(self):
        """Test positive interval."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([2.0])

        relaxation = compute_reciprocal_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"All positive (small): {message}"

    def test_all_positive_large(self):
        """Test positive interval with larger values."""
        lower = torch.tensor([5.0])
        upper = torch.tensor([10.0])

        relaxation = compute_reciprocal_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"All positive (large): {message}"

    def test_all_negative_small(self):
        """Test negative interval."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([-1.0])

        relaxation = compute_reciprocal_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"All negative (small): {message}"

    def test_all_negative_large(self):
        """Test negative interval with larger absolute values."""
        lower = torch.tensor([-10.0])
        upper = torch.tensor([-5.0])

        relaxation = compute_reciprocal_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"All negative (large): {message}"

    def test_crossing_zero(self):
        """Test interval crossing zero: should return infinite bounds."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])

        relaxation = compute_reciprocal_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)

        # Should have infinite bounds
        assert beta_lower.item() == float("-inf"), "Lower bound should be -inf when crossing zero"
        assert beta_upper.item() == float("inf"), "Upper bound should be +inf when crossing zero"
        assert is_sound, f"Crossing zero: {message}"

    def test_zero_width_positive(self):
        """Test zero-width interval (positive)."""
        lower = torch.tensor([2.0])
        upper = torch.tensor([2.0])

        relaxation = compute_reciprocal_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width (positive): {message}"

    def test_zero_width_negative(self):
        """Test zero-width interval (negative)."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([-2.0])

        relaxation = compute_reciprocal_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width (negative): {message}"

    def test_batch_mixed_regimes(self):
        """Test batch with mixed regimes."""
        lower = torch.tensor([[1.0, -2.0, 5.0], [-1.0, -10.0, 1.0]])
        upper = torch.tensor([[2.0, -1.0, 10.0], [1.0, -5.0, 2.0]])

        relaxation = compute_reciprocal_relaxation(lower, upper)
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

    def test_near_zero_positive(self):
        """Test interval very close to zero (positive side)."""
        lower = torch.tensor([0.01])
        upper = torch.tensor([0.1])

        relaxation = compute_reciprocal_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Near zero (positive): {message}"

    def test_near_zero_negative(self):
        """Test interval very close to zero (negative side)."""
        lower = torch.tensor([-0.1])
        upper = torch.tensor([-0.01])

        relaxation = compute_reciprocal_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Near zero (negative): {message}"
