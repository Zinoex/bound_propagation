"""
Tests for clamp linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for clamp
linear relaxations produce valid upper and lower bounds.

Clamp is piecewise linear: clamp(x, min, max) = min(max(x, min), max).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from bound_propagation.propagation.linear_relaxations.clamp import compute_clamp_alpha_beta


class TestClampRelaxationSoundness:
    """Test that clamp linear relaxations are sound (bounds are valid)."""

    def verify_bounds_sound(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha_lower: torch.Tensor,
        beta_lower: torch.Tensor,
        alpha_upper: torch.Tensor,
        beta_upper: torch.Tensor,
        min_val: float = None,
        max_val: float = None,
        num_samples: int = 1000,
    ):
        """Verify that linear bounds are sound by sampling points in [lower, upper]."""
        t = torch.linspace(0, 1, num_samples, dtype=lower.dtype, device=lower.device)
        t = t.view(-1, *([1] * lower.ndim))
        x_samples = lower + t * (upper - lower)

        # Compute actual clamp values
        clamp_vals = torch.clamp(x_samples, min=min_val, max=max_val)

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= clamp for all samples)
        upper_violations = clamp_vals > upper_bound + 1e-5
        if upper_violations.any():
            max_violation = (clamp_vals - upper_bound).max().item()
            return False, f"Upper bound violated. Max violation: {max_violation:.6f}"

        # Check lower bound (must be <= clamp for all samples)
        lower_violations = clamp_vals < lower_bound - 1e-5
        if lower_violations.any():
            max_violation = (lower_bound - clamp_vals).max().item()
            return False, f"Lower bound violated. Max violation: {max_violation:.6f}"

        return True, "Bounds are sound"

    def test_in_range(self):
        """Test interval fully within [min, max]: clamp(x) = x."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([3.0])
        min_val, max_val = -5.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val
        )
        assert is_sound, f"In range: {message}"

        # Should be identity
        assert torch.allclose(alpha_lower, torch.tensor([1.0]))
        assert torch.allclose(alpha_upper, torch.tensor([1.0]))

    def test_below_min(self):
        """Test interval fully below min: clamp(x) = min."""
        lower = torch.tensor([-10.0])
        upper = torch.tensor([-6.0])
        min_val, max_val = -5.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val
        )
        assert is_sound, f"Below min: {message}"

        # Should be constant at min
        assert torch.allclose(beta_lower, torch.tensor([min_val]))
        assert torch.allclose(beta_upper, torch.tensor([min_val]))

    def test_above_max(self):
        """Test interval fully above max: clamp(x) = max."""
        lower = torch.tensor([6.0])
        upper = torch.tensor([10.0])
        min_val, max_val = -5.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val
        )
        assert is_sound, f"Above max: {message}"

        # Should be constant at max
        assert torch.allclose(beta_lower, torch.tensor([max_val]))
        assert torch.allclose(beta_upper, torch.tensor([max_val]))

    def test_crosses_min(self):
        """Test interval crossing min threshold."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([2.0])
        min_val, max_val = 0.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val
        )
        assert is_sound, f"Crosses min: {message}"

    def test_crosses_max(self):
        """Test interval crossing max threshold."""
        lower = torch.tensor([3.0])
        upper = torch.tensor([7.0])
        min_val, max_val = -5.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val
        )
        assert is_sound, f"Crosses max: {message}"

    def test_crosses_both(self):
        """Test interval crossing both min and max."""
        lower = torch.tensor([-10.0])
        upper = torch.tensor([10.0])
        min_val, max_val = -2.0, 2.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val
        )
        assert is_sound, f"Crosses both: {message}"

    def test_only_min_constraint(self):
        """Test with only minimum constraint (no max)."""
        lower = torch.tensor([-5.0])
        upper = torch.tensor([5.0])
        min_val = 0.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(
            lower, upper, min_val=min_val, max_val=None
        )

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val=min_val, max_val=None
        )
        assert is_sound, f"Only min constraint: {message}"

    def test_only_max_constraint(self):
        """Test with only maximum constraint (no min)."""
        lower = torch.tensor([-5.0])
        upper = torch.tensor([5.0])
        max_val = 2.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(
            lower, upper, min_val=None, max_val=max_val
        )

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val=None, max_val=max_val
        )
        assert is_sound, f"Only max constraint: {message}"

    def test_zero_width_in_range(self):
        """Test zero-width interval within range."""
        lower = torch.tensor([2.0])
        upper = torch.tensor([2.0])
        min_val, max_val = 0.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val, num_samples=10
        )
        assert is_sound, f"Zero width (in range): {message}"

    def test_zero_width_below_min(self):
        """Test zero-width interval below min."""
        lower = torch.tensor([-5.0])
        upper = torch.tensor([-5.0])
        min_val, max_val = 0.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val, num_samples=10
        )
        assert is_sound, f"Zero width (below min): {message}"

    def test_zero_width_above_max(self):
        """Test zero-width interval above max."""
        lower = torch.tensor([10.0])
        upper = torch.tensor([10.0])
        min_val, max_val = 0.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val, num_samples=10
        )
        assert is_sound, f"Zero width (above max): {message}"

    def test_batch_mixed_regimes(self):
        """Test batch with mixed regimes."""
        lower = torch.tensor([[1.0, -10.0, 6.0], [-2.0, 3.0, -10.0]])
        upper = torch.tensor([[3.0, -6.0, 10.0], [2.0, 7.0, 10.0]])
        min_val, max_val = -5.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        # Test each element separately
        for i in range(lower.shape[0]):
            for j in range(lower.shape[1]):
                l = lower[i, j : j + 1]
                u = upper[i, j : j + 1]
                al = alpha_lower[i, j : j + 1]
                bl = beta_lower[i, j : j + 1]
                au = alpha_upper[i, j : j + 1]
                bu = beta_upper[i, j : j + 1]

                is_sound, message = self.verify_bounds_sound(l, u, al, bl, au, bu, min_val, max_val)
                assert is_sound, f"Batch element [{i},{j}]: {message}"

    def test_narrow_crossing_min(self):
        """Test narrow interval crossing min threshold."""
        lower = torch.tensor([-0.5])
        upper = torch.tensor([0.5])
        min_val, max_val = 0.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val
        )
        assert is_sound, f"Narrow crossing min: {message}"

    def test_narrow_crossing_max(self):
        """Test narrow interval crossing max threshold."""
        lower = torch.tensor([4.5])
        upper = torch.tensor([5.5])
        min_val, max_val = -5.0, 5.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val
        )
        assert is_sound, f"Narrow crossing max: {message}"

    def test_symmetric_range(self):
        """Test symmetric interval with symmetric clamp."""
        lower = torch.tensor([-3.0])
        upper = torch.tensor([3.0])
        min_val, max_val = -1.0, 1.0

        alpha_lower, beta_lower, alpha_upper, beta_upper = compute_clamp_alpha_beta(lower, upper, min_val, max_val)

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, min_val, max_val
        )
        assert is_sound, f"Symmetric range: {message}"
