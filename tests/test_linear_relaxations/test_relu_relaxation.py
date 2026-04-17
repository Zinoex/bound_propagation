"""
Tests for ReLU linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for ReLU
linear relaxations produce valid upper and lower bounds.

ReLU is piecewise linear: ReLU(x) = max(0, x).
"""

import torch

from bound_propagation.propagation.linear_relaxations.elementwise import compute_relu_relaxation


class TestReLURelaxationSoundness:
    """Test that ReLU linear relaxations are sound (bounds are valid)."""

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

        # Compute actual ReLU values
        relu_vals = torch.relu(x_samples)

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= ReLU for all samples)
        upper_violations = relu_vals > upper_bound + 1e-5
        if upper_violations.any():
            max_violation = (relu_vals - upper_bound).max().item()
            return False, f"Upper bound violated. Max violation: {max_violation:.6f}"

        # Check lower bound (must be <= ReLU for all samples)
        lower_violations = relu_vals < lower_bound - 1e-5
        if lower_violations.any():
            max_violation = (lower_bound - relu_vals).max().item()
            return False, f"Lower bound violated. Max violation: {max_violation:.6f}"

        return True, "Bounds are sound"

    def test_all_positive(self):
        """Test all positive interval: ReLU(x) = x."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([3.0])

        relaxation = compute_relu_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"All positive: {message}"

        # For all positive, should be identity
        assert torch.allclose(alpha_lower, torch.tensor([1.0]))
        assert torch.allclose(alpha_upper, torch.tensor([1.0]))

    def test_all_negative(self):
        """Test all negative interval: ReLU(x) = 0."""
        lower = torch.tensor([-3.0])
        upper = torch.tensor([-1.0])

        relaxation = compute_relu_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"All negative: {message}"

        # For all negative, should be constant 0
        assert torch.allclose(alpha_lower, torch.tensor([0.0]))
        assert torch.allclose(alpha_upper, torch.tensor([0.0]))
        assert torch.allclose(beta_lower, torch.tensor([0.0]))
        assert torch.allclose(beta_upper, torch.tensor([0.0]))

    def test_crossing_zero_narrow(self):
        """Test narrow interval crossing zero."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])

        relaxation = compute_relu_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (narrow): {message}"

    def test_crossing_zero_wide(self):
        """Test wide interval crossing zero."""
        lower = torch.tensor([-5.0])
        upper = torch.tensor([5.0])

        relaxation = compute_relu_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (wide): {message}"

    def test_crossing_zero_asymmetric_positive_heavy(self):
        """Test interval crossing zero (more positive than negative)."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([5.0])

        relaxation = compute_relu_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (positive heavy): {message}"

    def test_crossing_zero_asymmetric_negative_heavy(self):
        """Test interval crossing zero (more negative than positive)."""
        lower = torch.tensor([-5.0])
        upper = torch.tensor([1.0])

        relaxation = compute_relu_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (negative heavy): {message}"

    def test_zero_width_positive(self):
        """Test zero-width interval (positive)."""
        lower = torch.tensor([2.0])
        upper = torch.tensor([2.0])

        relaxation = compute_relu_relaxation(lower, upper)
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

        relaxation = compute_relu_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width (negative): {message}"

    def test_zero_width_at_zero(self):
        """Test zero-width interval at zero."""
        lower = torch.tensor([0.0])
        upper = torch.tensor([0.0])

        relaxation = compute_relu_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width (at zero): {message}"

    def test_batch_mixed_regimes(self):
        """Test batch with mixed regimes."""
        lower = torch.tensor([[1.0, -3.0, -1.0], [-5.0, 0.0, 2.0]])
        upper = torch.tensor([[3.0, -1.0, 1.0], [1.0, 2.0, 2.0]])

        relaxation = compute_relu_relaxation(lower, upper)
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


class TestReLURelaxationAdaptive:
    """Test adaptive ReLU relaxation mode."""

    def test_adaptive_vs_standard_positive_heavy(self):
        """Test adaptive mode when interval is more positive."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([3.0])

        r = compute_relu_relaxation(lower, upper, adaptive=False)
        alpha_l_std = r.alpha_lower
        beta_l_std = r.beta_lower
        alpha_u_std = r.alpha_upper
        beta_u_std = r.beta_upper
        r = compute_relu_relaxation(lower, upper, adaptive=True)
        alpha_l_adp = r.alpha_lower
        beta_l_adp = r.beta_lower
        alpha_u_adp = r.alpha_upper
        beta_u_adp = r.beta_upper

        # Both should be sound
        def verify_sound(al, bl, au, bu):
            x = torch.linspace(lower.item(), upper.item(), 100)
            relu_x = torch.relu(x)
            upper_bound = au * x + bu
            lower_bound = al * x + bl
            return (relu_x <= upper_bound + 1e-5).all() and (relu_x >= lower_bound - 1e-5).all()

        assert verify_sound(alpha_l_std, beta_l_std, alpha_u_std, beta_u_std)
        assert verify_sound(alpha_l_adp, beta_l_adp, alpha_u_adp, beta_u_adp)

        # Adaptive should choose tighter bound (alpha=1 for positive heavy)
        assert torch.allclose(alpha_l_adp, torch.tensor([1.0]))

    def test_adaptive_vs_standard_negative_heavy(self):
        """Test adaptive mode when interval is more negative."""
        lower = torch.tensor([-3.0])
        upper = torch.tensor([1.0])

        r = compute_relu_relaxation(lower, upper, adaptive=False)
        alpha_l_std = r.alpha_lower
        beta_l_std = r.beta_lower
        alpha_u_std = r.alpha_upper
        beta_u_std = r.beta_upper
        r = compute_relu_relaxation(lower, upper, adaptive=True)
        alpha_l_adp = r.alpha_lower
        beta_l_adp = r.beta_lower
        alpha_u_adp = r.alpha_upper
        beta_u_adp = r.beta_upper

        # Both should be sound
        def verify_sound(al, bl, au, bu):
            x = torch.linspace(lower.item(), upper.item(), 100)
            relu_x = torch.relu(x)
            upper_bound = au * x + bu
            lower_bound = al * x + bl
            return (relu_x <= upper_bound + 1e-5).all() and (relu_x >= lower_bound - 1e-5).all()

        assert verify_sound(alpha_l_std, beta_l_std, alpha_u_std, beta_u_std)
        assert verify_sound(alpha_l_adp, beta_l_adp, alpha_u_adp, beta_u_adp)

        # Adaptive should choose tighter bound (alpha=0 for negative heavy)
        assert torch.allclose(alpha_l_adp, torch.tensor([0.0]))
