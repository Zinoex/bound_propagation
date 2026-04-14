from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.mul import IBPMul
from bound_propagation.propagation.ibp.reciprocal import IBPReciprocal

from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for reciprocal (1/x) operation."""
    strategy = IBPReciprocal()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_reciprocal_positive_interval() -> None:
    """Test reciprocal of positive interval."""
    # 1/[2, 4] = [1/4, 1/2] = [0.25, 0.5]
    out = _propagate(
        lower=torch.tensor([2.0]),
        upper=torch.tensor([4.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.25]))
    assert torch.allclose(out.upper, torch.tensor([0.5]))


def test_reciprocal_positive_small_interval() -> None:
    """Test reciprocal of small positive interval."""
    # 1/[0.1, 0.5] = [2, 10]
    out = _propagate(
        lower=torch.tensor([0.1]),
        upper=torch.tensor([0.5]),
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([10.0]))


def test_reciprocal_negative_interval() -> None:
    """Test reciprocal of negative interval."""
    # 1/[-4, -2] = [-1/2, -1/4] = [-0.5, -0.25]
    out = _propagate(
        lower=torch.tensor([-4.0]),
        upper=torch.tensor([-2.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-0.5]))
    assert torch.allclose(out.upper, torch.tensor([-0.25]))


def test_reciprocal_negative_small_interval() -> None:
    """Test reciprocal of small negative interval."""
    # 1/[-0.5, -0.1] = [-10, -2]
    out = _propagate(
        lower=torch.tensor([-0.5]),
        upper=torch.tensor([-0.1]),
    )

    assert torch.allclose(out.lower, torch.tensor([-10.0]))
    assert torch.allclose(out.upper, torch.tensor([-2.0]))


def test_reciprocal_interval_containing_zero() -> None:
    """Test reciprocal of interval containing zero - should return unbounded."""
    # 1/[-2, 3] = [-inf, inf]
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([3.0]),
    )

    assert torch.isneginf(out.lower).all()
    assert torch.isposinf(out.upper).all()


def test_reciprocal_interval_with_zero_lower() -> None:
    """Test reciprocal when lower bound is zero - should return unbounded."""
    # 1/[0, 2] = [-inf, inf]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([2.0]),
    )

    assert torch.isneginf(out.lower).all()
    assert torch.isposinf(out.upper).all()


def test_reciprocal_interval_with_zero_upper() -> None:
    """Test reciprocal when upper bound is zero - should return unbounded."""
    # 1/[-2, 0] = [-inf, inf]
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.isneginf(out.lower).all()
    assert torch.isposinf(out.upper).all()


def test_reciprocal_point_interval() -> None:
    """Test reciprocal of point interval (lower = upper)."""
    # 1/[4, 4] = [0.25, 0.25]
    out = _propagate(
        lower=torch.tensor([4.0]),
        upper=torch.tensor([4.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.25]))
    assert torch.allclose(out.upper, torch.tensor([0.25]))


def test_reciprocal_batched_intervals() -> None:
    """Test reciprocal with batched intervals."""
    out = _propagate(
        lower=torch.tensor([2.0, -4.0, -2.0, 0.1]),
        upper=torch.tensor([4.0, -2.0, 3.0, 0.5]),
    )

    # 1/[2, 4] = [0.25, 0.5]
    # 1/[-4, -2] = [-0.5, -0.25]
    # 1/[-2, 3] = [-inf, inf]
    # 1/[0.1, 0.5] = [2, 10]
    assert torch.allclose(out.lower[0], torch.tensor(0.25))
    assert torch.allclose(out.upper[0], torch.tensor(0.5))

    assert torch.allclose(out.lower[1], torch.tensor(-0.5))
    assert torch.allclose(out.upper[1], torch.tensor(-0.25))

    assert torch.isneginf(out.lower[2])
    assert torch.isposinf(out.upper[2])

    assert torch.allclose(out.lower[3], torch.tensor(2.0))
    assert torch.allclose(out.upper[3], torch.tensor(10.0))


def test_reciprocal_multidimensional() -> None:
    """Test reciprocal with multi-dimensional intervals."""
    out = _propagate(
        lower=torch.tensor([[2.0, -4.0], [0.5, -0.5]]),
        upper=torch.tensor([[4.0, -1.0], [2.0, 0.5]]),
    )

    expected_lower = torch.tensor([[0.25, -1.0], [0.5, float("-inf")]])
    expected_upper = torch.tensor([[0.5, -0.25], [2.0, float("inf")]])

    assert torch.allclose(out.lower[0, 0], expected_lower[0, 0])
    assert torch.allclose(out.lower[0, 1], expected_lower[0, 1])
    assert torch.allclose(out.lower[1, 0], expected_lower[1, 0])
    assert torch.isneginf(out.lower[1, 1])

    assert torch.allclose(out.upper[0, 0], expected_upper[0, 0])
    assert torch.allclose(out.upper[0, 1], expected_upper[0, 1])
    assert torch.allclose(out.upper[1, 0], expected_upper[1, 0])
    assert torch.isposinf(out.upper[1, 1])


def test_reciprocal_involutory_property() -> None:
    """Test that 1/(1/x) = x for intervals not containing zero."""
    strategy = IBPReciprocal()
    a = IntervalBounds(torch.tensor([2.0, -4.0]), torch.tensor([5.0, -1.0]))

    # 1/a
    recip_a = propagate(strategy, a)

    # 1/(1/a)
    recip_recip_a = propagate(strategy, recip_a)

    # Should recover original interval
    assert torch.allclose(a.lower, recip_recip_a.lower, rtol=1e-5)
    assert torch.allclose(a.upper, recip_recip_a.upper, rtol=1e-5)


def test_reciprocal_ordering_reversal() -> None:
    """Test that reciprocal reverses ordering for same-sign intervals."""
    # For positive interval [a, b] with a < b, we have 1/b < 1/a
    lower_pos = torch.tensor([2.0])
    upper_pos = torch.tensor([8.0])
    out_pos = _propagate(lower_pos, upper_pos)

    # 1/8 < 1/2, so lower should be 1/8 and upper 1/2
    assert out_pos.lower < out_pos.upper
    assert torch.allclose(out_pos.lower, 1.0 / upper_pos)
    assert torch.allclose(out_pos.upper, 1.0 / lower_pos)

    # For negative interval, same property holds
    lower_neg = torch.tensor([-8.0])
    upper_neg = torch.tensor([-2.0])
    out_neg = _propagate(lower_neg, upper_neg)

    # -1/2 < -1/8, so lower should be -1/2 and upper -1/8
    assert out_neg.lower < out_neg.upper
    assert torch.allclose(out_neg.lower, 1.0 / upper_neg)
    assert torch.allclose(out_neg.upper, 1.0 / lower_neg)


def test_reciprocal_with_multiplication_property() -> None:
    """Test that x * (1/x) contains [1, 1] for non-zero intervals."""
    a = IntervalBounds(torch.tensor([2.0, -5.0]), torch.tensor([4.0, -2.0]))

    recip_strategy = IBPReciprocal()
    mul_strategy = IBPMul()

    # 1/a
    recip_a = propagate(recip_strategy, a)

    # a * (1/a)
    result = propagate(mul_strategy, a, recip_a)

    # Should contain 1 (may be wider due to overestimation)
    assert torch.all(result.lower <= 1.0)
    assert torch.all(result.upper >= 1.0)
