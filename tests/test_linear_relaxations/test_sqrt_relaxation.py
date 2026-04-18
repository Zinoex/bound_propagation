"""
Tests for sqrt linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for sqrt
linear relaxations produce valid upper and lower bounds.
"""

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.linear_relaxations.elementwise import compute_sqrt_relaxation


class TestSqrtRelaxationSoundness:
    """Test that sqrt linear relaxations are sound (bounds are valid)."""

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

        A sound upper bound satisfies: alpha_upper * x + beta_upper >= sqrt(x) for all x in [lower, upper]
        A sound lower bound satisfies: alpha_lower * x + beta_lower <= sqrt(x) for all x in [lower, upper]
        """
        # Generate sample points in [lower, upper]
        t = torch.linspace(0, 1, num_samples, dtype=lower.dtype, device=lower.device)
        # Broadcast to match shape
        t = t.view(-1, *([1] * lower.ndim))
        x_samples = lower + t * (upper - lower)

        # Compute actual sqrt values
        sqrt_vals = torch.sqrt(x_samples)

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= sqrt for all samples)
        upper_violations = sqrt_vals > upper_bound + 1e-6  # Allow small numerical tolerance
        if upper_violations.any():
            max_violation = (sqrt_vals - upper_bound).max().item()
            indices = torch.where(upper_violations)
            return False, f"Upper bound violated. Max violation: {max_violation:.6f} at indices {indices}"

        # Check lower bound (must be <= sqrt for all samples)
        lower_violations = sqrt_vals < lower_bound - 1e-6  # Allow small numerical tolerance
        if lower_violations.any():
            max_violation = (lower_bound - sqrt_vals).max().item()
            indices = torch.where(lower_violations)
            return False, f"Lower bound violated. Max violation: {max_violation:.6f} at indices {indices}"

        return True, "Bounds are sound"

    def test_small_to_one(self):
        """Test small interval to 1."""
        lower = torch.tensor([0.01])
        upper = torch.tensor([1.0])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Small to one: {message}"

    def test_moderate_range(self):
        """Test moderate range."""
        lower = torch.tensor([0.1])
        upper = torch.tensor([4.0])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Moderate range: {message}"

    def test_larger_range(self):
        """Test larger range."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([9.0])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Larger range: {message}"

    def test_starting_from_zero(self):
        """Test interval starting from 0."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([1.0])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Starting from zero: {message}"

    def test_zero_width_interval(self):
        """Test zero-width interval."""
        lower = torch.tensor([4.0])
        upper = torch.tensor([4.0])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width: {message}"

    def test_wide_range(self):
        """Test wide range."""
        lower = torch.tensor([0.25])
        upper = torch.tensor([16.0])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Wide range: {message}"

    def test_batch_processing(self):
        """Test batch with different ranges."""
        lower = torch.tensor([[0.01, 0.1, 1.0], [0.0, 4.0, 0.25]])
        upper = torch.tensor([[1.0, 4.0, 9.0], [1.0, 4.0, 16.0]])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Batch processing: {message}"

    def test_extreme_values(self):
        """Test with extreme input values."""
        lower = torch.tensor([0.0001, 1.0, 100.0])
        upper = torch.tensor([0.01, 100.0, 10000.0])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Extreme values: {message}"


class TestSqrtRelaxationInvalidInputs:
    """Test that sqrt linear relaxations handle invalid inputs correctly."""

    def test_negative_lower_bound(self):
        """Test that negative lower bounds result in nan."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        # All parameters should be nan for invalid inputs
        assert torch.isnan(alpha_lower).all()
        assert torch.isnan(beta_lower).all()
        assert torch.isnan(alpha_upper).all()
        assert torch.isnan(beta_upper).all()

    def test_all_negative(self):
        """Test that all negative bounds result in nan."""
        lower = torch.tensor([-5.0])
        upper = torch.tensor([-1.0])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        # All parameters should be nan for invalid inputs
        assert torch.isnan(alpha_lower).all()
        assert torch.isnan(beta_lower).all()
        assert torch.isnan(alpha_upper).all()
        assert torch.isnan(beta_upper).all()

    def test_batch_with_some_invalid(self):
        """Test batch processing with some invalid inputs."""
        lower = torch.tensor([0.1, -1.0, 1.0])
        upper = torch.tensor([1.0, 1.0, 4.0])

        relaxation = compute_sqrt_relaxation(IntervalBounds(lower, upper))
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        # First and third should be valid, second should be nan
        assert not torch.isnan(alpha_lower[0])
        assert torch.isnan(alpha_lower[1])
        assert not torch.isnan(alpha_lower[2])

        assert not torch.isnan(beta_lower[0])
        assert torch.isnan(beta_lower[1])
        assert not torch.isnan(beta_lower[2])

        assert not torch.isnan(alpha_upper[0])
        assert torch.isnan(alpha_upper[1])
        assert not torch.isnan(alpha_upper[2])

        assert not torch.isnan(beta_upper[0])
        assert torch.isnan(beta_upper[1])
        assert not torch.isnan(beta_upper[2])
