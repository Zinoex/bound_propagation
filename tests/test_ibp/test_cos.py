from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.add import IBPAddWithConstant
from bound_propagation.propagation.ibp.cos import IBPCos
from bound_propagation.propagation.ibp.neg import IBPNeg
from bound_propagation.propagation.ibp.sin import IBPSin


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for cos operation."""
    strategy = IBPCos()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return strategy.propagate_forwards(node=None, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]


def test_cos_small_positive_interval() -> None:
    """Test cos of small positive interval not containing extrema."""
    # cos([0.1, 0.5]) ≈ [cos(0.5), cos(0.1)] ≈ [0.8776, 0.9950]
    out = _propagate(
        lower=torch.tensor([0.1]),
        upper=torch.tensor([0.5]),
    )

    # cos is decreasing on [0, π], so cos(0.5) < cos(0.1)
    expected_lower = torch.cos(torch.tensor([0.5]))
    expected_upper = torch.cos(torch.tensor([0.1]))

    assert torch.allclose(out.lower, expected_lower, atol=1e-4)
    assert torch.allclose(out.upper, expected_upper, atol=1e-4)


def test_cos_zero() -> None:
    """Test cos(0) = 1."""
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([1.0]))
    assert torch.allclose(out.upper, torch.tensor([1.0]))


def test_cos_interval_from_zero() -> None:
    """Test cos([0, 1]) (contains maximum at 0)."""
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([1.0]),
    )

    # cos(0) = 1, cos(1) ≈ 0.540, max is at 0
    assert torch.allclose(out.upper, torch.tensor([1.0]), atol=1e-6)
    expected_lower = torch.cos(torch.tensor([1.0]))
    assert torch.allclose(out.lower, expected_lower, atol=1e-4)


def test_cos_interval_containing_max_at_2pi() -> None:
    """Test cos interval that may contain maximum at 2π."""
    # cos([6, 7]) contains 2π ≈ 6.283
    out = _propagate(
        lower=torch.tensor([6.0]),
        upper=torch.tensor([7.0]),
    )

    # Should be bounded in [-1, 1]
    assert torch.all(out.lower >= -1.0)
    assert torch.all(out.upper <= 1.0)
    # Should produce valid bounds
    assert out.lower <= out.upper


def test_cos_interval_containing_min_at_pi() -> None:
    """Test cos interval that may contain minimum at π."""
    # cos([3, 3.5]) contains π ≈ 3.142
    out = _propagate(
        lower=torch.tensor([3.0]),
        upper=torch.tensor([3.5]),
    )

    # Should be bounded in [-1, 1]
    assert torch.all(out.lower >= -1.0)
    assert torch.all(out.upper <= 1.0)
    # Should produce valid bounds
    assert out.lower <= out.upper


def test_cos_full_period() -> None:
    """Test cos over a full period [0, 2π]."""
    two_pi = 2 * torch.pi
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([two_pi]),
    )

    # Should be bounded in [-1, 1]
    assert torch.all(out.lower >= -1.0)
    assert torch.all(out.upper <= 1.0)
    # Should produce valid bounds
    assert out.lower <= out.upper


def test_cos_negative_interval() -> None:
    """Test cos of negative interval."""
    # cos([-0.5, -0.1]) (cos is even, so same as cos([0.1, 0.5]))
    out = _propagate(
        lower=torch.tensor([-0.5]),
        upper=torch.tensor([-0.1]),
    )

    # cos is increasing on [-π, 0], so cos(-0.5) < cos(-0.1)
    expected_lower = torch.cos(torch.tensor([-0.5]))
    expected_upper = torch.cos(torch.tensor([-0.1]))

    assert torch.allclose(out.lower, expected_lower, atol=1e-4)
    assert torch.allclose(out.upper, expected_upper, atol=1e-4)


def test_cos_symmetric_interval_around_zero() -> None:
    """Test cos of symmetric interval around zero."""
    # cos([-0.5, 0.5]) should have maximum at 0
    out = _propagate(
        lower=torch.tensor([-0.5]),
        upper=torch.tensor([0.5]),
    )

    # cos(0) = 1, cos(±0.5) ≈ 0.8776
    assert torch.allclose(out.upper, torch.tensor([1.0]), atol=1e-6)
    expected_lower = torch.cos(torch.tensor([0.5]))
    assert torch.allclose(out.lower, expected_lower, atol=1e-4)


def test_cos_batched_intervals() -> None:
    """Test cos with batched intervals."""
    out = _propagate(
        lower=torch.tensor([0.0, 3.0, 6.0]),
        upper=torch.tensor([1.0, 3.5, 6.5]),
    )

    # [0, 1]: contains max at 0
    assert torch.allclose(out.upper[0], torch.tensor([1.0]), atol=1e-6)

    # [3, 3.5]: contains π (min)
    assert torch.allclose(out.lower[1], torch.tensor([-1.0]), atol=1e-6)

    # [6, 6.5]: contains 2π (max)
    assert torch.allclose(out.upper[2], torch.tensor([1.0]), atol=1e-6)


def test_cos_bounded_by_one() -> None:
    """Test that cos always returns bounds in [-1, 1]."""
    # Test various intervals
    test_cases = [
        (torch.tensor([-10.0]), torch.tensor([10.0])),
        (torch.tensor([0.0]), torch.tensor([100.0])),
        (torch.tensor([-50.0]), torch.tensor([-40.0])),
    ]

    for lower, upper in test_cases:
        out = _propagate(lower, upper)
        assert torch.all(out.lower >= -1.0)
        assert torch.all(out.upper <= 1.0)


def test_cos_periodicity() -> None:
    """Test that cos respects periodicity: bounds for cos(x) and cos(x + 2π) should be related."""
    two_pi = 2 * torch.pi

    # Small interval not crossing extrema
    out1 = _propagate(torch.tensor([0.2]), torch.tensor([0.4]))

    # Same interval shifted by 2π
    out2 = _propagate(torch.tensor([0.2 + two_pi]), torch.tensor([0.4 + two_pi]))

    # Results should be approximately equal
    assert torch.allclose(out1.lower, out2.lower, atol=1e-5)
    assert torch.allclose(out1.upper, out2.upper, atol=1e-5)


def test_cos_even_function_property() -> None:
    """Test that cos is an even function: cos(-x) = cos(x)."""
    a = IntervalBounds(torch.tensor([0.5, 1.0]), torch.tensor([1.5, 2.0]))

    strategy = IBPCos()

    neg_strategy = IBPNeg()

    # cos(a)
    cos_a = strategy.propagate_forwards(None, [a])  # ty:ignore[invalid-argument-type]

    # -a
    neg_a = neg_strategy.propagate_forwards(None, [a])  # ty:ignore[invalid-argument-type]

    # cos(-a)
    cos_neg_a = strategy.propagate_forwards(None, [neg_a])  # ty:ignore[invalid-argument-type]

    # cos(-a) should equal cos(a) for small intervals not containing extrema
    # Due to interval overestimation, this may not be exact, but should be close
    assert torch.allclose(cos_neg_a.lower, cos_a.lower, atol=1e-5)
    assert torch.allclose(cos_neg_a.upper, cos_a.upper, atol=1e-5)


def test_cos_relation_to_sin() -> None:
    """Test that cos(x) = sin(x + π/2)."""

    # Small interval not containing extrema
    a = IntervalBounds(torch.tensor([0.1]), torch.tensor([0.3]))

    cos_strategy = IBPCos()
    sin_strategy = IBPSin()
    add_strategy = IBPAddWithConstant()

    # cos(a)
    cos_a = cos_strategy.propagate_forwards(None, [a])  # ty:ignore[invalid-argument-type]

    # a + π/2
    pi_over_2 = torch.pi / 2
    a_shifted = add_strategy.propagate_forwards(None, [a, pi_over_2])  # ty:ignore[invalid-argument-type]

    # sin(a + π/2)
    sin_shifted = sin_strategy.propagate_forwards(None, [a_shifted])  # ty:ignore[invalid-argument-type]

    # Should be approximately equal
    assert torch.allclose(cos_a.lower, sin_shifted.lower, atol=1e-5)
    assert torch.allclose(cos_a.upper, sin_shifted.upper, atol=1e-5)
