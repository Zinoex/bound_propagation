"""Tests for ``pow`` linear relaxation soundness (currently power=2 only).

``y = x²`` is convex; the relaxation uses a chord upper bound and a tangent
lower bound at a configurable point. These tests confirm soundness across
several regimes and that the parameters match the analytic formulas.
"""

from __future__ import annotations

import pytest
import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.linear_relaxations.elementwise import compute_pow_relaxation


def _verify_sound(
    lower: torch.Tensor,
    upper: torch.Tensor,
    alpha_lower: torch.Tensor,
    beta_lower: torch.Tensor,
    alpha_upper: torch.Tensor,
    beta_upper: torch.Tensor,
    *,
    power: int = 2,
    num_samples: int = 200,
) -> None:
    t = torch.linspace(0, 1, num_samples, dtype=lower.dtype, device=lower.device)
    t = t.view(-1, *([1] * lower.ndim))
    samples = lower + t * (upper - lower)
    actual = samples**power
    upper_bound = alpha_upper * samples + beta_upper
    lower_bound = alpha_lower * samples + beta_lower
    assert torch.all(actual <= upper_bound + 1e-5), (actual.max(), upper_bound.min())
    assert torch.all(actual >= lower_bound - 1e-5), (actual.min(), lower_bound.max())


class TestPowRelaxationSoundness:
    def test_crossing_zero(self) -> None:
        lower = torch.tensor([-2.0, -1.0])
        upper = torch.tensor([3.0, 1.5])
        params = compute_pow_relaxation(IntervalBounds(lower, upper), power=2)
        _verify_sound(lower, upper, params.alpha_lower, params.beta_lower, params.alpha_upper, params.beta_upper)

    def test_all_positive(self) -> None:
        lower = torch.tensor([0.5, 2.0])
        upper = torch.tensor([1.5, 4.0])
        params = compute_pow_relaxation(IntervalBounds(lower, upper), power=2)
        _verify_sound(lower, upper, params.alpha_lower, params.beta_lower, params.alpha_upper, params.beta_upper)

    def test_all_negative(self) -> None:
        lower = torch.tensor([-3.0, -2.0])
        upper = torch.tensor([-1.0, -0.5])
        params = compute_pow_relaxation(IntervalBounds(lower, upper), power=2)
        _verify_sound(lower, upper, params.alpha_lower, params.beta_lower, params.alpha_upper, params.beta_upper)

    def test_zero_width(self) -> None:
        # Both bounds equal x²; alpha=0, beta=x².
        x = torch.tensor([2.0, -1.5, 0.0])
        params = compute_pow_relaxation(IntervalBounds(x, x), power=2)
        assert torch.allclose(params.beta_lower, x * x)
        assert torch.allclose(params.beta_upper, x * x)

    def test_chord_passes_through_endpoints(self) -> None:
        # Upper bound at the endpoints should equal x².
        lower = torch.tensor([-1.0])
        upper = torch.tensor([2.0])
        params = compute_pow_relaxation(IntervalBounds(lower, upper), power=2)
        upper_at_lower = params.alpha_upper * lower + params.beta_upper
        upper_at_upper = params.alpha_upper * upper + params.beta_upper
        assert torch.allclose(upper_at_lower, lower**2)
        assert torch.allclose(upper_at_upper, upper**2)

    def test_tangent_default_at_center(self) -> None:
        # Default tangent point t = (l+u)/2; lower bound there equals t².
        lower = torch.tensor([-1.0])
        upper = torch.tensor([3.0])
        center = (lower + upper) / 2  # = 1.0
        params = compute_pow_relaxation(IntervalBounds(lower, upper), power=2)
        lower_at_center = params.alpha_lower * center + params.beta_lower
        assert torch.allclose(lower_at_center, center**2)

    def test_tangent_alpha_override(self) -> None:
        # Override the tangent fraction to t = lower (alpha=0).
        lower = torch.tensor([-1.0])
        upper = torch.tensor([3.0])
        alpha = torch.zeros_like(lower)
        params = compute_pow_relaxation(IntervalBounds(lower, upper), power=2, alpha_pow_tangent=alpha)
        lower_at_lower = params.alpha_lower * lower + params.beta_lower
        assert torch.allclose(lower_at_lower, lower**2)

    def test_unsupported_power_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="power=2"):
            compute_pow_relaxation(IntervalBounds(torch.tensor([-1.0]), torch.tensor([1.0])), power=3)


class TestPowEndToEndSoundness:
    """End-to-end: ``pow`` works through forward / backward LBP without errors."""

    def _propagate(self, fn, *, lo: float, hi: float, method: str):
        from bound_propagation import BoundModel, HyperRectangle

        dummy = torch.zeros(1)
        model = BoundModel(fn, dummy_inputs=(dummy,), method=method)
        region = HyperRectangle(lower=torch.tensor([lo]), upper=torch.tensor([hi]))
        return model.propagate(region)

    def _samples(self, lo: float, hi: float, n: int = 50):
        return torch.linspace(lo, hi, n)

    def test_pow_forward_lbp_is_sound(self) -> None:
        bounds = self._propagate(lambda x: torch.pow(x, 2), lo=-1.5, hi=2.0, method="forward_lbp").concretize()
        for x in self._samples(-1.5, 2.0):
            with torch.no_grad():
                y = torch.pow(x.reshape(1), 2)
            assert torch.all(bounds.lower <= y + 1e-5)
            assert torch.all(y <= bounds.upper + 1e-5)

    def test_pow_backward_lbp_is_sound(self) -> None:
        bounds = self._propagate(lambda x: torch.pow(x, 2), lo=-1.5, hi=2.0, method="backward_lbp").concretize()
        for x in self._samples(-1.5, 2.0):
            with torch.no_grad():
                y = torch.pow(x.reshape(1), 2)
            assert torch.all(bounds.lower <= y + 1e-5)
            assert torch.all(y <= bounds.upper + 1e-5)

    def test_x_squared_via_simplify_pass(self) -> None:
        """``x*x`` rewritten to ``pow(x, 2)`` by SimplificationPass works in LBP modes."""
        from bound_propagation import BoundModel, HyperRectangle

        dummy = torch.zeros(1)
        for method in ("forward_lbp", "backward_lbp"):
            model = BoundModel(lambda x: x * x, dummy_inputs=(dummy,), method=method, simplify=True)
            region = HyperRectangle(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))
            bounds = model.propagate(region).concretize()
            assert torch.all(bounds.lower <= bounds.upper + 1e-6), method
