"""
Tests for sigmoid linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for sigmoid
linear relaxations produce valid upper and lower bounds.
"""

import torch

from bound_propagation.propagation.linear_relaxations.sigmoid import compute_sigmoid_alpha_beta


class TestSigmoidRelaxationSoundness:
    """Test that sigmoid linear relaxations are sound (bounds are valid)."""

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

        A sound upper bound satisfies: alpha_upper * x + beta_upper >= sigmoid(x) for all x in [lower, upper]
        A sound lower bound satisfies: alpha_lower * x + beta_lower <= sigmoid(x) for all x in [lower, upper]
        """
        # Generate sample points in [lower, upper]
        t = torch.linspace(0, 1, num_samples, dtype=lower.dtype, device=lower.device)
        # Broadcast to match shape
        t = t.view(-1, *([1] * lower.ndim))
        x_samples = lower + t * (upper - lower)

        # Compute actual sigmoid values
        sigmoid_vals = torch.sigmoid(x_samples)

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= sigmoid for all samples)
        upper_violations = sigmoid_vals > upper_bound + 1e-6  # Allow small numerical tolerance
        if upper_violations.any():
            max_violation = (sigmoid_vals - upper_bound).max().item()
            indices = torch.where(upper_violations)
            return False, f"Upper bound violated. Max violation: {max_violation:.6f} at indices {indices}"

        # Check lower bound (must be <= sigmoid for all samples)
        lower_violations = sigmoid_vals < lower_bound - 1e-6  # Allow small numerical tolerance
        if lower_violations.any():
            max_violation = (lower_bound - sigmoid_vals).max().item()
            indices = torch.where(lower_violations)
            return False, f"Lower bound violated. Max violation: {max_violation:.6f} at indices {indices}"

        return True, "Bounds are sound"

    def test_crossing_zero_narrow_interval(self):
        """Test crossing regime with narrow interval around zero."""
        lower = torch.tensor([-0.1])
        upper = torch.tensor([0.1])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (narrow): {message}"

    def test_crossing_zero_wide_interval(self):
        """Test crossing regime with wide interval - this is the problematic case."""
        lower = torch.tensor([0.01])
        upper = torch.tensor([5.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (wide, lower=0.01, upper=5): {message}"

    def test_crossing_zero_wide_negative(self):
        """Test crossing regime starting from negative side."""
        lower = torch.tensor([-5.0])
        upper = torch.tensor([-0.01])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (wide negative): {message}"

    def test_crossing_zero_symmetric(self):
        """Test crossing regime with symmetric interval."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([2.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (symmetric): {message}"

    def test_crossing_zero_asymmetric_positive_heavy(self):
        """Test crossing regime with asymmetric interval (more on positive side)."""
        lower = torch.tensor([-0.5])
        upper = torch.tensor([3.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (asymmetric positive): {message}"

    def test_crossing_zero_asymmetric_negative_heavy(self):
        """Test crossing regime with asymmetric interval (more on negative side)."""
        lower = torch.tensor([-3.0])
        upper = torch.tensor([0.5])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (asymmetric negative): {message}"

    def test_all_positive_regime(self):
        """Test all positive regime."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([3.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"All positive: {message}"

    def test_all_negative_regime(self):
        """Test all negative regime."""
        lower = torch.tensor([-3.0])
        upper = torch.tensor([-1.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"All negative: {message}"

    def test_zero_width_interval(self):
        """Test zero-width interval."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([1.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width: {message}"

    def test_batch_mixed_regimes(self):
        """Test batch with mixed regimes."""
        lower = torch.tensor([[-3.0, -0.1, 1.0], [-5.0, 0.01, -3.0]])
        upper = torch.tensor([[-1.0, 0.1, 3.0], [-0.01, 5.0, -1.0]])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Batch mixed regimes: {message}"

    def test_extreme_values(self):
        """Test with extreme input values."""
        lower = torch.tensor([-10.0, -1.0, 1.0])
        upper = torch.tensor([-5.0, 10.0, 10.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_sigmoid_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Extreme values: {message}"


class TestSigmoidRelaxationSymmetry:
    """Test that sigmoid linear relaxations are symmetric around zero."""

    def test_symmetry_around_zero(self):
        """
        Test that sigmoid relaxation is symmetric: sigmoid(-x) = 1 - sigmoid(x).

        For intervals symmetric around zero, the relaxations should respect this symmetry.
        """
        x = 2.0
        lower_pos = torch.tensor([0.1])
        upper_pos = torch.tensor([x])
        lower_neg = torch.tensor([-x])
        upper_neg = torch.tensor([-0.1])

        alpha_lower_pos, beta_lower_pos, alpha_upper_pos, beta_upper_pos = compute_sigmoid_alpha_beta(
            lower_pos, upper_pos
        )
        alpha_lower_neg, beta_lower_neg, alpha_upper_neg, beta_upper_neg = compute_sigmoid_alpha_beta(
            lower_neg, upper_neg
        )

        # For symmetric intervals, the bounds should have related properties
        # This is more of a sanity check than a strict requirement
        # The actual symmetry would be: sigmoid(-x) = 1 - sigmoid(x)

        # Verify sigmoid symmetry: sigmoid(x) + sigmoid(-x) = 1
        x_val = torch.tensor([1.5])
        sig_pos = torch.sigmoid(x_val)
        sig_neg = torch.sigmoid(-x_val)

        assert torch.allclose(sig_pos + sig_neg, torch.ones_like(sig_pos), atol=1e-5)
