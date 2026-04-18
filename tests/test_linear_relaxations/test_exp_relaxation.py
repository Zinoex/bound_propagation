"""
Tests for exp linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for exp
linear relaxations produce valid upper and lower bounds.

Exp is a convex function globally.
"""

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.linear_relaxations.elementwise import compute_exp_relaxation


class TestExpRelaxationSoundness:
    """Test that exp linear relaxations are sound (bounds are valid)."""

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
        t = torch.linspace(0, 1, num_samples, dtype=lower.dtype, device=lower.device)
        t = t.view(-1, *([1] * lower.ndim))
        x_samples = lower + t * (upper - lower)

        # Compute actual exp values
        exp_vals = torch.exp(x_samples)

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= exp for all samples)
        upper_violations = exp_vals > upper_bound + 1e-3  # Relaxed tolerance for exp
        if upper_violations.any():
            max_violation = (exp_vals - upper_bound).max().item()
            return False, f"Upper bound violated. Max violation: {max_violation:.6f}"

        # Check lower bound (must be <= exp for all samples)
        lower_violations = exp_vals < lower_bound - 1e-3
        if lower_violations.any():
            max_violation = (lower_bound - exp_vals).max().item()
            return False, f"Lower bound violated. Max violation: {max_violation:.6f}"

        return True, "Bounds are sound"

    def test_small_positive_range(self):
        """Test small positive range."""
        lower = torch.tensor([0.1])
        upper = torch.tensor([0.5])

        relaxation = compute_exp_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Small positive range: {message}"

    def test_moderate_positive_range(self):
        """Test moderate positive range."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([3.0])

        relaxation = compute_exp_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Moderate positive range: {message}"

    def test_negative_range(self):
        """Test negative range."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([-0.5])

        relaxation = compute_exp_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Negative range: {message}"

    def test_crossing_zero(self):
        """Test interval crossing zero."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])

        relaxation = compute_exp_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero: {message}"

    def test_large_positive_range(self):
        """Test large positive range."""
        lower = torch.tensor([3.0])
        upper = torch.tensor([5.0])

        relaxation = compute_exp_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Large positive range: {message}"

    def test_zero_width_interval(self):
        """Test zero-width interval."""
        lower = torch.tensor([1.5])
        upper = torch.tensor([1.5])

        relaxation = compute_exp_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width: {message}"

    def test_very_negative(self):
        """Test very negative values (exp approaches 0)."""
        lower = torch.tensor([-10.0])
        upper = torch.tensor([-5.0])

        relaxation = compute_exp_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Very negative: {message}"

    def test_batch_processing(self):
        """Test batch with mixed ranges."""
        lower = torch.tensor([[0.1, -2.0, 1.0], [-1.0, 3.0, -10.0]])
        upper = torch.tensor([[0.5, -0.5, 3.0], [1.0, 5.0, -5.0]])

        relaxation = compute_exp_relaxation(IntervalBounds(lower, upper))
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


class TestExpRelaxationConvexity:
    """Test convexity properties of exp."""

    def test_tangent_below_curve(self):
        """Test that tangent line is below exp curve (convex property)."""
        x0 = torch.tensor([1.0])
        exp_x0 = torch.exp(x0)
        slope = exp_x0  # exp'(x) = exp(x)

        # Tangent line: y = exp(x0) + exp(x0) * (x - x0) = exp(x0) * x + exp(x0) * (1 - x0)
        x_test = torch.linspace(-1.0, 3.0, 100)
        tangent = slope * x_test + (exp_x0 - slope * x0)
        exp_test = torch.exp(x_test)

        # Tangent should be below or at curve
        assert (tangent <= exp_test + 1e-5).all()

    def test_secant_above_curve(self):
        """Test that secant line is above exp curve (convex property)."""
        x1 = torch.tensor([0.0])
        x2 = torch.tensor([2.0])
        y1 = torch.exp(x1)
        y2 = torch.exp(x2)

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Test points between x1 and x2
        x_test = torch.linspace(x1.item(), x2.item(), 100)
        secant = slope * x_test + intercept
        exp_test = torch.exp(x_test)

        # Secant should be above or at curve
        assert (secant >= exp_test - 1e-5).all()
