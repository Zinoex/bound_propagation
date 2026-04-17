from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.elementwise import IBPSin
from bound_propagation.propagation.ibp.linear import IBPNeg
from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for sin operation."""
    strategy = IBPSin()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_sin_small_positive_interval() -> None:
    """Test sin on small positive interval [0.1, 0.5]."""
    out = _propagate(
        lower=torch.tensor([0.1]),
        upper=torch.tensor([0.5]),
    )

    # sin is monotonically increasing on [0, π/2], so:
    # sin([0.1, 0.5]) ≈ [0.0998, 0.4794]
    assert torch.allclose(out.lower, torch.sin(torch.tensor([0.1])), atol=1e-4)
    assert torch.allclose(out.upper, torch.sin(torch.tensor([0.5])), atol=1e-4)


def test_sin_zero_interval() -> None:
    """Test sin(0)."""
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    # sin(0) = 0
    assert torch.allclose(out.lower, torch.tensor([0.0]))


def test_sin_negative_interval() -> None:
    """Test sin on negative interval."""
    out = _propagate(
        lower=torch.tensor([-0.5]),
        upper=torch.tensor([-0.1]),
    )

    # sin is monotonically increasing, so sin([-0.5, -0.1]) ≈ [sin(-0.5), sin(-0.1)]
    assert torch.allclose(out.lower, torch.sin(torch.tensor([-0.5])), atol=1e-4)
    assert torch.allclose(out.upper, torch.sin(torch.tensor([-0.1])), atol=1e-4)


def test_sin_bounds_in_range() -> None:
    """Test that sin bounds are always in [-1, 1]."""
    test_cases = [
        (torch.tensor([-10.0]), torch.tensor([10.0])),
        (torch.tensor([0.0]), torch.tensor([5.0])),
        (torch.tensor([-5.0]), torch.tensor([0.0])),
    ]

    for lower, upper in test_cases:
        out = _propagate(lower, upper)
        assert torch.all(out.lower >= -1.0)
        assert torch.all(out.upper <= 1.0)


def test_sin_batched_intervals() -> None:
    """Test sin with batched intervals."""
    out = _propagate(
        lower=torch.tensor([0.0, 0.5, -0.5]),
        upper=torch.tensor([0.5, 2.0, 0.5]),
    )

    # All should be bounded in [-1, 1]
    assert torch.all(out.lower >= -1.0)
    assert torch.all(out.upper <= 1.0)
    # Should produce valid bounds
    assert torch.all(out.lower <= out.upper)


def test_sin_multidimensional() -> None:
    """Test sin with multi-dimensional intervals."""
    out = _propagate(
        lower=torch.tensor([[0.0, 0.5], [-0.5, 1.0]]),
        upper=torch.tensor([[0.5, 1.0], [0.5, 2.0]]),
    )

    # All should be bounded in [-1, 1]
    assert torch.all(out.lower >= -1.0)
    assert torch.all(out.upper <= 1.0)
    # Shape should be preserved
    assert out.lower.shape == (2, 2)
    assert out.upper.shape == (2, 2)


def test_sin_periodicity() -> None:
    """Test that sin respects periodicity for small intervals."""
    two_pi = 2 * torch.pi

    # Small interval not crossing extrema
    out1 = _propagate(torch.tensor([0.2]), torch.tensor([0.4]))

    # Same interval shifted by 2π
    out2 = _propagate(torch.tensor([0.2 + two_pi]), torch.tensor([0.4 + two_pi]))

    # Results should be approximately equal
    assert torch.allclose(out1.lower, out2.lower, atol=1e-5)
    assert torch.allclose(out1.upper, out2.upper, atol=1e-5)


def test_sin_odd_function_property() -> None:
    """Test that sin bounds respect odd function properties."""
    # For small intervals not containing extrema
    a = IntervalBounds(torch.tensor([0.2, 0.5]), torch.tensor([0.3, 0.6]))

    strategy = IBPSin()

    neg_strategy = IBPNeg()

    # sin(a)
    sin_a = propagate(strategy, a)

    # -a
    neg_a_val = propagate(neg_strategy, a)

    # sin(-a)
    sin_neg_a = propagate(strategy, neg_a_val)

    # -sin(a)
    neg_sin_a = propagate(neg_strategy, sin_a)

    # For small intervals, sin(-a) should be close to -sin(a)
    # Due to interval overestimation with extrema, allow some tolerance
    assert torch.allclose(sin_neg_a.lower, neg_sin_a.lower, atol=0.1)
    assert torch.allclose(sin_neg_a.upper, neg_sin_a.upper, atol=0.1)


def test_sin_monotonicity() -> None:
    """Test that sin preserves monotonicity for nested intervals."""
    inner = IntervalBounds(torch.tensor([0.2]), torch.tensor([0.4]))
    outer = IntervalBounds(torch.tensor([0.1]), torch.tensor([0.5]))

    strategy = IBPSin()
    out_inner = propagate(strategy, inner)
    out_outer = propagate(strategy, outer)

    # Outer interval should contain inner interval results
    assert torch.all(out_outer.lower <= out_inner.lower)
    assert torch.all(out_outer.upper >= out_inner.upper)


def test_sin_peak_at_pi_over_2() -> None:
    """Test that sin correctly detects peak at π/2 ≈ 1.571."""
    # Interval [1.0, 2.0] contains π/2 ≈ 1.571 where sin reaches maximum of 1
    out = _propagate(
        lower=torch.tensor([1.0]),
        upper=torch.tensor([2.0]),
    )

    # Peak at π/2 means upper bound should be 1.0
    assert torch.allclose(out.upper, torch.tensor([1.0]), atol=1e-4)
    # Lower bound should be sin(1.0) ≈ 0.8415
    assert torch.allclose(out.lower, torch.sin(torch.tensor([1.0])), atol=1e-4)


def test_sin_trough_at_3pi_over_2() -> None:
    """Test that sin correctly detects trough at 3π/2 ≈ 4.712."""
    # Interval [4.5, 5.0] contains 3π/2 ≈ 4.712 where sin reaches minimum of -1
    out = _propagate(
        lower=torch.tensor([4.5]),
        upper=torch.tensor([5.0]),
    )

    # Trough at 3π/2 means lower bound should be -1.0
    assert torch.allclose(out.lower, torch.tensor([-1.0]), atol=1e-4)
    # Upper bound should be sin(5.0) ≈ -0.9589
    assert torch.allclose(out.upper, torch.sin(torch.tensor([5.0])), atol=1e-4)


def test_sin_peak_at_5pi_over_2() -> None:
    """Test that sin correctly detects peak at 5π/2 ≈ 7.854 (second period)."""
    # Interval [7.0, 8.0] contains 5π/2 ≈ 7.854 where sin reaches maximum of 1
    out = _propagate(
        lower=torch.tensor([7.0]),
        upper=torch.tensor([8.0]),
    )

    # Peak at 5π/2 means upper bound should be 1.0
    assert torch.allclose(out.upper, torch.tensor([1.0]), atol=1e-4)
    # Lower bound should be sin(7.0) ≈ 0.6570
    assert torch.allclose(out.lower, torch.sin(torch.tensor([7.0])), atol=1e-4)


def test_sin_negative_peak() -> None:
    """Test that sin correctly detects peak in negative range."""
    # Interval [-2.0, -1.0] contains -π/2 ≈ -1.571 where sin reaches minimum of -1
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([-1.0]),
    )

    # Trough at -π/2 means lower bound should be -1.0
    assert torch.allclose(out.lower, torch.tensor([-1.0]), atol=1e-4)
    # Upper bound should be sin(-1.0) ≈ -0.8415
    assert torch.allclose(out.upper, torch.sin(torch.tensor([-1.0])), atol=1e-4)
