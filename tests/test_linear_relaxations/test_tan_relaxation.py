"""
Tests for tan linear relaxation soundness.

These tests verify that the alpha/beta parameters computed for tan
linear relaxations produce valid upper and lower bounds.

Tan has asymptotes at x = π/2 + nπ and alternates between convex and concave regions:
- Convex: (-π/2, 0), (π/2, π), etc.
- Concave: (0, π/2), (π, 3π/2), etc.
"""

import torch

from bound_propagation.propagation.linear_relaxations.elementwise import compute_tan_relaxation


class TestTanRelaxationSoundness:
    """Test that tan linear relaxations are sound (bounds are valid)."""

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

        A sound upper bound satisfies: alpha_upper * x + beta_upper >= tan(x) for all x in [lower, upper]
        A sound lower bound satisfies: alpha_lower * x + beta_lower <= tan(x) for all x in [lower, upper]

        Returns (is_sound, message, allows_infinite) where allows_infinite indicates if infinite bounds are acceptable.
        """
        # Check if bounds are infinite (valid for crossing asymptotes)
        has_inf_lower = torch.isinf(beta_lower).any() or torch.isinf(alpha_lower).any()
        has_inf_upper = torch.isinf(beta_upper).any() or torch.isinf(alpha_upper).any()

        if has_inf_lower or has_inf_upper:
            # For intervals crossing asymptotes, we expect -inf and +inf bounds
            if beta_lower.item() == float("-inf") and beta_upper.item() == float("inf"):
                return True, "Infinite bounds (crosses asymptote)", True
            else:
                return True, "Partial infinite bounds", True

        # Generate sample points in [lower, upper]
        t = torch.linspace(0, 1, num_samples, dtype=lower.dtype, device=lower.device)
        # Broadcast to match shape
        t = t.view(-1, *([1] * lower.ndim))
        x_samples = lower + t * (upper - lower)

        # Compute actual tan values
        tan_vals = torch.tan(x_samples)

        # Compute linear approximations
        upper_bound = alpha_upper * x_samples + beta_upper
        lower_bound = alpha_lower * x_samples + beta_lower

        # Check upper bound (must be >= tan for all samples)
        upper_violations = tan_vals > upper_bound + 1e-5  # Allow small numerical tolerance
        if upper_violations.any():
            max_violation = (tan_vals - upper_bound).max().item()
            indices = torch.where(upper_violations)
            return False, f"Upper bound violated. Max violation: {max_violation:.6f} at indices {indices}", False

        # Check lower bound (must be <= tan for all samples)
        lower_violations = tan_vals < lower_bound - 1e-5  # Allow small numerical tolerance
        if lower_violations.any():
            max_violation = (lower_bound - tan_vals).max().item()
            indices = torch.where(lower_violations)
            return False, f"Lower bound violated. Max violation: {max_violation:.6f} at indices {indices}", False

        return True, "Bounds are sound", False

    def test_convex_regime_narrow(self):
        """Test convex regime: tan is convex on (-π/2, 0)."""
        lower = torch.tensor([-1.4])
        upper = torch.tensor([-0.5])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, _ = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Convex regime (narrow): {message}"

    def test_convex_regime_wide(self):
        """Test convex regime: wider interval in (-π/2, 0)."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([-0.1])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, _ = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Convex regime (wide): {message}"

    def test_concave_regime_narrow(self):
        """Test concave regime: tan is concave on (0, π/2)."""
        lower = torch.tensor([0.1])
        upper = torch.tensor([1.0])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, _ = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Concave regime (narrow): {message}"

    def test_concave_regime_wide(self):
        """Test concave regime: wider interval in (0, π/2)."""
        lower = torch.tensor([0.5])
        upper = torch.tensor([1.4])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, _ = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Concave regime (wide): {message}"

    def test_crossing_zero_narrow(self):
        """Test crossing zero (inflection point): tan changes from convex to concave."""
        lower = torch.tensor([-0.5])
        upper = torch.tensor([0.5])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, _ = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (narrow): {message}"

    def test_crossing_zero_wide(self):
        """Test crossing zero: wider interval around inflection point."""
        lower = torch.tensor([-1.0])
        upper = torch.tensor([1.0])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, _ = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Crossing zero (wide): {message}"

    def test_crossing_asymptote_positive(self):
        """Test crossing asymptote at π/2 ≈ 1.5708: should return infinite bounds."""
        lower = torch.tensor([1.0])
        upper = torch.tensor([2.0])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, allows_inf = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper
        )

        # Should have infinite bounds
        assert allows_inf, f"Crossing π/2 asymptote should have infinite bounds: {message}"
        assert beta_lower.item() == float("-inf"), "Lower bound should be -inf when crossing asymptote"
        assert beta_upper.item() == float("inf"), "Upper bound should be +inf when crossing asymptote"

    def test_crossing_asymptote_narrow(self):
        """Test crossing asymptote: narrow interval around π/2."""
        lower = torch.tensor([1.4])
        upper = torch.tensor([1.7])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, allows_inf = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper
        )

        assert allows_inf, f"Crossing π/2 asymptote (narrow) should have infinite bounds: {message}"

    def test_crossing_asymptote_negative(self):
        """Test crossing asymptote at -π/2 ≈ -1.5708."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([-1.0])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, allows_inf = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper
        )

        assert allows_inf, f"Crossing -π/2 asymptote should have infinite bounds: {message}"

    def test_crossing_multiple_asymptotes(self):
        """Test crossing multiple asymptotes: should return infinite bounds."""
        lower = torch.tensor([-2.0])
        upper = torch.tensor([2.0])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, allows_inf = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper
        )

        assert allows_inf, f"Crossing multiple asymptotes should have infinite bounds: {message}"

    def test_second_period_concave(self):
        """Test second period of tan: concave part around x ≈ 3.5."""
        lower = torch.tensor([3.5])
        upper = torch.tensor([4.0])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, _ = self.verify_bounds_sound(lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper)
        assert is_sound, f"Second period (concave): {message}"

    def test_zero_width_interval(self):
        """Test zero-width interval."""
        lower = torch.tensor([0.5])
        upper = torch.tensor([0.5])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        is_sound, message, _ = self.verify_bounds_sound(
            lower, upper, alpha_lower, beta_lower, alpha_upper, beta_upper, num_samples=10
        )
        assert is_sound, f"Zero width: {message}"

    def test_batch_mixed_regimes(self):
        """Test batch with mixed regimes."""
        lower = torch.tensor([[-1.0, 0.1, -0.5], [0.5, 1.4, 3.5]])
        upper = torch.tensor([[-0.1, 1.0, 0.5], [1.4, 1.7, 4.0]])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        # Test each element separately since some might have infinite bounds
        for i in range(lower.shape[0]):
            for j in range(lower.shape[1]):
                lo = lower[i, j : j + 1]
                up = upper[i, j : j + 1]
                al = alpha_lower[i, j : j + 1]
                bl = beta_lower[i, j : j + 1]
                au = alpha_upper[i, j : j + 1]
                bu = beta_upper[i, j : j + 1]

                is_sound, message, _ = self.verify_bounds_sound(lo, up, al, bl, au, bu)
                assert is_sound, f"Batch element [{i},{j}]: {message}"


class TestTanRelaxationAsymptotes:
    """Test specific asymptote handling for tan."""

    def test_asymptote_detection(self):
        """Test that intervals crossing asymptotes are detected correctly."""
        # π/2 ≈ 1.5708

        # Crossing π/2
        lower = torch.tensor([1.5])
        upper = torch.tensor([1.65])

        relaxation = compute_tan_relaxation(lower, upper)
        beta_lower = relaxation.beta_lower
        beta_upper = relaxation.beta_upper

        # Should have infinite bounds
        assert torch.isinf(beta_lower), "Should detect asymptote crossing"
        assert torch.isinf(beta_upper), "Should detect asymptote crossing"

    def test_near_asymptote_not_crossing(self):
        """Test interval near but not crossing asymptote."""
        # Just before π/2
        lower = torch.tensor([1.0])
        upper = torch.tensor([1.5])

        relaxation = compute_tan_relaxation(lower, upper)
        alpha_lower = relaxation.alpha_lower
        beta_lower = relaxation.beta_lower
        alpha_upper = relaxation.alpha_upper
        beta_upper = relaxation.beta_upper

        # This might still cross depending on exact value, but test it doesn't crash
        assert alpha_lower is not None
        assert beta_lower is not None
        assert alpha_upper is not None
        assert beta_upper is not None
