"""
Tests for abs linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for abs
linear relaxations produce valid upper and lower bounds.

Abs is piecewise linear: abs(x) = x for x >= 0, abs(x) = -x for x < 0.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from bound_propagation.propagation.linear_relaxations.abs import compute_abs_alpha_beta


class TestAbsRelaxationSoundness:
    """Test that abs linear relaxations are sound (bounds are valid)."""

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

        # Compute actual abs values
        abs_vals = torch.abs(x_samples)

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= abs for all samples)
        upper_violations = abs_vals > upper_bound + 1e-5
        if upper_violations.any():
            max_violation = (abs_vals - upper_bound).max().item()
            return False, f"Upper bound violated. Max violation: {max_violation:.6f}"

        # Check lower bound (must be <= abs for all samples)
        lower_violations = abs_vals < lower_bound - 1e-5
        if lower_violations.any():
            max_violation = (lower_bound - abs_vals).max().item()
            return False, f"Lower bound violated. Max violation: {max_violation:.6f}"

        return True, "Bounds are sound"

    def test_all_positive(self):
        """Test all positive interval: abs(x) = x."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([3.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"All positive: {message}"

        # For all positive, should be identity
        assert torch.allclose(alpha_lower, torch.tensor([1.0]))
        assert torch.allclose(alpha_upper, torch.tensor([1.0]))

    def test_all_negative(self):
        """Test all negative interval: abs(x) = -x."""
        lower = torch.tensor([-3.0])
        upper = torch.tensor([-1.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"All negative: {message}"

        # For all negative, should be negation
        assert torch.allclose(alpha_lower, torch.tensor([-1.0]))
        assert torch.allclose(alpha_upper, torch.tensor([-1.0]))

    def test_crossing_zero_symmetric(self):
        """Test interval crossing zero (symmetric)."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([2.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (symmetric): {message}"

    def test_crossing_zero_asymmetric_positive_heavy(self):
        """Test interval crossing zero (more positive than negative)."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([3.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (positive heavy): {message}"

    def test_crossing_zero_asymmetric_negative_heavy(self):
        """Test interval crossing zero (more negative than positive)."""
        lower = torch.tensor([-3.0])
        upper = torch.tensor([1.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (negative heavy): {message}"

    def test_zero_width_positive(self):
        """Test zero-width interval (positive)."""
        lower = torch.tensor([2.0])
        upper = torch.tensor([2.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width (positive): {message}"

    def test_zero_width_negative(self):
        """Test zero-width interval (negative)."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([-2.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width (negative): {message}"

    def test_zero_width_at_zero(self):
        """Test zero-width interval at zero."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([0.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width (at zero): {message}"

    def test_batch_mixed_regimes(self):
        """Test batch with mixed regimes."""
        lower = torch.tensor([[1.0, -3.0, -2.0], [-1.0, -3.0, 0.0]])
        upper = torch.tensor([[3.0, -1.0, 2.0], [3.0, 1.0, 2.0]])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        # Test each element separately
        for i in range(lower.shape[0]):
            for j in range(lower.shape[1]):
                l = lower[i, j : j + 1]
                u = upper[i, j : j + 1]
                al = alpha_lower[i, j : j + 1]
                bl = beta_lower[i, j : j + 1]
                au = alpha_upper[i, j : j + 1]
                bu = beta_upper[i, j : j + 1]

                is_sound, message = self.verify_bounds_sound(l, u, al, bl, au, bu)
                assert is_sound, f"Batch element [{i},{j}]: {message}"

    def test_including_zero_from_negative(self):
        """Test interval from negative to zero."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([0.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Negative to zero: {message}"

    def test_including_zero_from_positive(self):
        """Test interval from zero to positive."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([2.0])

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_abs_alpha_beta(lower, upper)

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Zero to positive: {message}"
