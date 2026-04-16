"""
Tests for log linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for log
linear relaxations produce valid upper and lower bounds.

Log is a concave function for x > 0.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from bound_propagation.propagation.linear_relaxations.log import compute_log_relaxation


class TestLogRelaxationSoundness:
    """Test that log linear relaxations are sound (bounds are valid)."""

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

        # Compute actual log values
        log_vals = torch.log(x_samples)

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= log for all samples)
        upper_violations = log_vals > upper_bound + 1e-5
        if upper_violations.any():
            max_violation = (log_vals - upper_bound).max().item()
            return False, f"Upper bound violated. Max violation: {max_violation:.6f}"

        # Check lower bound (must be <= log for all samples)
        lower_violations = log_vals < lower_bound - 1e-5
        if lower_violations.any():
            max_violation = (lower_bound - log_vals).max().item()
            return False, f"Lower bound violated. Max violation: {max_violation:.6f}"

        return True, "Bounds are sound"

    def test_small_positive_range(self):
        """Test small positive range."""
        lower = torch.tensor([0.1])
        upper = torch.tensor([0.5])

        relaxation = compute_log_relaxation(lower, upper)
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

        relaxation = compute_log_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Moderate positive range: {message}"

    def test_around_one(self):
        """Test interval around 1 (where log(1) = 0)."""
        lower = torch.tensor([0.5])
        upper = torch.tensor([2.0])

        relaxation = compute_log_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Around one: {message}"

    def test_large_positive_range(self):
        """Test large positive range."""
        lower = torch.tensor([5.0])
        upper = torch.tensor([10.0])

        relaxation = compute_log_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Large positive range: {message}"

    def test_zero_width_interval(self):
        """Test zero-width interval."""
        lower = torch.tensor([2.0])
        upper = torch.tensor([2.0])

        relaxation = compute_log_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width: {message}"

    def test_near_zero(self):
        """Test interval very close to zero (log approaches -inf)."""
        lower = torch.tensor([0.01])
        upper = torch.tensor([0.1])

        relaxation = compute_log_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Near zero: {message}"

    def test_batch_processing(self):
        """Test batch with mixed positive ranges."""
        lower = torch.tensor([[0.1, 1.0, 0.5], [5.0, 0.01, 2.0]])
        upper = torch.tensor([[0.5, 3.0, 2.0], [10.0, 0.1, 2.0]])

        relaxation = compute_log_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

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


class TestLogRelaxationInvalidInputs:
    """Test log relaxation with invalid (non-positive) inputs."""

    def test_negative_lower_bound(self):
        """Test that negative inputs produce nan."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([0.5])

        relaxation = compute_log_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        # Should get nan for invalid inputs
        assert torch.isnan(alpha_lower).all()
        assert torch.isnan(beta_lower).all()
        assert torch.isnan(alpha_upper).all()
        assert torch.isnan(beta_upper).all()

    def test_zero_lower_bound(self):
        """Test that zero lower bound produces nan."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([1.0])

        relaxation = compute_log_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        # Should get nan for invalid inputs
        assert torch.isnan(alpha_lower).all()
        assert torch.isnan(beta_lower).all()
        assert torch.isnan(alpha_upper).all()
        assert torch.isnan(beta_upper).all()

    def test_batch_with_some_invalid(self):
        """Test batch where some elements are invalid."""
        lower = torch.tensor([[0.1, -1.0, 1.0], [0.0, 2.0, 0.5]])
        upper = torch.tensor([[0.5, 0.5, 3.0], [1.0, 5.0, 2.0]])

        relaxation = compute_log_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        # First element [0,0] should be valid
        assert not torch.isnan(alpha_lower[0, 0])

        # Elements [0,1] and [1,0] should be nan (invalid)
        assert torch.isnan(alpha_lower[0, 1])
        assert torch.isnan(alpha_lower[1, 0])

        # Other valid elements should not be nan
        assert not torch.isnan(alpha_lower[0, 2])
        assert not torch.isnan(alpha_lower[1, 1])
        assert not torch.isnan(alpha_lower[1, 2])
