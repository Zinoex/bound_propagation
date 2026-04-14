from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.div import IBPDiv
from bound_propagation.propagation.ibp.mul import IBPMul
from tests.helpers import propagate


def _propagate_interval_interval(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> IntervalBounds:
    """Propagate bounds for interval / interval operation."""
    strategy = IBPDiv()
    left_bounds = IntervalBounds(lower=left_lower, upper=left_upper)
    right_bounds = IntervalBounds(lower=right_lower, upper=right_upper)
    return propagate(strategy, left_bounds, right_bounds)


def _propagate_interval_div_constant(
    interval_lower: torch.Tensor, interval_upper: torch.Tensor, constant: torch.Tensor | float
) -> IntervalBounds:
    """Propagate bounds for interval / constant operation."""
    strategy = IBPDiv()
    interval_bounds = IntervalBounds(lower=interval_lower, upper=interval_upper)
    return propagate(strategy, interval_bounds, constant)


def _propagate_constant_div_interval(
    constant: torch.Tensor | float, interval_lower: torch.Tensor, interval_upper: torch.Tensor
) -> IntervalBounds:
    """Propagate bounds for constant / interval operation."""
    strategy = IBPDiv()
    interval_bounds = IntervalBounds(lower=interval_lower, upper=interval_upper)
    return propagate(strategy, constant, interval_bounds)


def test_div_positive_intervals() -> None:
    """Test division of two positive intervals."""
    # [6, 12] / [2, 3] = [2, 6]
    # Quotients: 6/2=3, 6/3=2, 12/2=6, 12/3=4
    # min=2, max=6
    out = _propagate_interval_interval(
        left_lower=torch.tensor([6.0]),
        left_upper=torch.tensor([12.0]),
        right_lower=torch.tensor([2.0]),
        right_upper=torch.tensor([3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([6.0]))


def test_div_negative_intervals() -> None:
    """Test division of two negative intervals."""
    # [-12, -6] / [-3, -2] = [2, 6]
    # Quotients: -12/-3=4, -12/-2=6, -6/-3=2, -6/-2=3
    # min=2, max=6
    out = _propagate_interval_interval(
        left_lower=torch.tensor([-12.0]),
        left_upper=torch.tensor([-6.0]),
        right_lower=torch.tensor([-3.0]),
        right_upper=torch.tensor([-2.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([6.0]))


def test_div_positive_by_negative() -> None:
    """Test division of positive by negative interval."""
    # [6, 12] / [-3, -2] = [-6, -2]
    # Quotients: 6/-3=-2, 6/-2=-3, 12/-3=-4, 12/-2=-6
    # min=-6, max=-2
    out = _propagate_interval_interval(
        left_lower=torch.tensor([6.0]),
        left_upper=torch.tensor([12.0]),
        right_lower=torch.tensor([-3.0]),
        right_upper=torch.tensor([-2.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-6.0]))
    assert torch.allclose(out.upper, torch.tensor([-2.0]))


def test_div_negative_by_positive() -> None:
    """Test division of negative by positive interval."""
    # [-12, -6] / [2, 3] = [-6, -2]
    # Quotients: -12/2=-6, -12/3=-4, -6/2=-3, -6/3=-2
    # min=-6, max=-2
    out = _propagate_interval_interval(
        left_lower=torch.tensor([-12.0]),
        left_upper=torch.tensor([-6.0]),
        right_lower=torch.tensor([2.0]),
        right_upper=torch.tensor([3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-6.0]))
    assert torch.allclose(out.upper, torch.tensor([-2.0]))


def test_div_by_interval_containing_zero() -> None:
    """Test division when divisor contains zero - should return unbounded."""
    # [2, 4] / [-1, 1] should be unbounded
    out = _propagate_interval_interval(
        left_lower=torch.tensor([2.0]),
        left_upper=torch.tensor([4.0]),
        right_lower=torch.tensor([-1.0]),
        right_upper=torch.tensor([1.0]),
    )

    assert torch.isneginf(out.lower).all()
    assert torch.isposinf(out.upper).all()


def test_div_by_interval_with_zero_lower() -> None:
    """Test division when divisor has zero as lower bound - should be unbounded."""
    # [2, 4] / [0, 2] should be unbounded
    out = _propagate_interval_interval(
        left_lower=torch.tensor([2.0]),
        left_upper=torch.tensor([4.0]),
        right_lower=torch.tensor([0.0]),
        right_upper=torch.tensor([2.0]),
    )

    assert torch.isneginf(out.lower).all()
    assert torch.isposinf(out.upper).all()


def test_div_by_interval_with_zero_upper() -> None:
    """Test division when divisor has zero as upper bound - should be unbounded."""
    # [2, 4] / [-2, 0] should be unbounded
    out = _propagate_interval_interval(
        left_lower=torch.tensor([2.0]),
        left_upper=torch.tensor([4.0]),
        right_lower=torch.tensor([-2.0]),
        right_upper=torch.tensor([0.0]),
    )

    assert torch.isneginf(out.lower).all()
    assert torch.isposinf(out.upper).all()


def test_div_point_intervals() -> None:
    """Test division of point intervals (lower = upper)."""
    # [12, 12] / [3, 3] = [4, 4]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([12.0]),
        left_upper=torch.tensor([12.0]),
        right_lower=torch.tensor([3.0]),
        right_upper=torch.tensor([3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([4.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0]))


def test_div_batched_intervals() -> None:
    """Test division with batched intervals."""
    out = _propagate_interval_interval(
        left_lower=torch.tensor([6.0, -12.0, 4.0]),
        left_upper=torch.tensor([12.0, -6.0, 8.0]),
        right_lower=torch.tensor([2.0, -3.0, 1.0]),
        right_upper=torch.tensor([3.0, -2.0, 2.0]),
    )

    # [6, 12] / [2, 3]: min=2, max=6
    # [-12, -6] / [-3, -2]: min=2, max=6
    # [4, 8] / [1, 2]: min=2, max=8
    expected_lower = torch.tensor([2.0, 2.0, 2.0])
    expected_upper = torch.tensor([6.0, 6.0, 8.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_div_interval_by_positive_constant() -> None:
    """Test division of interval by positive constant."""
    # [6, 12] / 3 = [2, 4]
    out = _propagate_interval_div_constant(
        interval_lower=torch.tensor([6.0]),
        interval_upper=torch.tensor([12.0]),
        constant=3.0,
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0]))


def test_div_interval_by_negative_constant() -> None:
    """Test division of interval by negative constant (bounds should flip)."""
    # [6, 12] / (-3) = [-4, -2]
    out = _propagate_interval_div_constant(
        interval_lower=torch.tensor([6.0]),
        interval_upper=torch.tensor([12.0]),
        constant=-3.0,
    )

    assert torch.allclose(out.lower, torch.tensor([-4.0]))
    assert torch.allclose(out.upper, torch.tensor([-2.0]))


def test_div_interval_by_zero_constant() -> None:
    """Test division of interval by zero constant - should return unbounded."""
    # [2, 4] / 0 should be unbounded
    out = _propagate_interval_div_constant(
        interval_lower=torch.tensor([2.0]),
        interval_upper=torch.tensor([4.0]),
        constant=0.0,
    )

    assert torch.isneginf(out.lower).all()
    assert torch.isposinf(out.upper).all()


def test_div_mixed_sign_interval_by_positive_constant() -> None:
    """Test division of mixed-sign interval by positive constant."""
    # [-6, 9] / 3 = [-2, 3]
    out = _propagate_interval_div_constant(
        interval_lower=torch.tensor([-6.0]),
        interval_upper=torch.tensor([9.0]),
        constant=3.0,
    )

    assert torch.allclose(out.lower, torch.tensor([-2.0]))
    assert torch.allclose(out.upper, torch.tensor([3.0]))


def test_div_mixed_sign_interval_by_negative_constant() -> None:
    """Test division of mixed-sign interval by negative constant."""
    # [-6, 9] / (-3) = [-3, 2]
    out = _propagate_interval_div_constant(
        interval_lower=torch.tensor([-6.0]),
        interval_upper=torch.tensor([9.0]),
        constant=-3.0,
    )

    assert torch.allclose(out.lower, torch.tensor([-3.0]))
    assert torch.allclose(out.upper, torch.tensor([2.0]))


def test_div_by_tensor_constant_with_zeros() -> None:
    """Test division by tensor constant containing zeros."""
    # Should handle element-wise zero division correctly
    out = _propagate_interval_div_constant(
        interval_lower=torch.tensor([2.0, 4.0, 6.0]),
        interval_upper=torch.tensor([4.0, 8.0, 12.0]),
        constant=torch.tensor([2.0, 0.0, 3.0]),
    )

    # First element: [2, 4] / 2 = [1, 2]
    assert torch.allclose(out.lower[0], torch.tensor(1.0))
    assert torch.allclose(out.upper[0], torch.tensor(2.0))

    # Second element: [4, 8] / 0 = [-inf, inf]
    assert torch.isneginf(out.lower[1])
    assert torch.isposinf(out.upper[1])

    # Third element: [6, 12] / 3 = [2, 4]
    assert torch.allclose(out.lower[2], torch.tensor(2.0))
    assert torch.allclose(out.upper[2], torch.tensor(4.0))


def test_div_by_tensor_constant_mixed_signs() -> None:
    """Test division by tensor constant with mixed signs."""
    out = _propagate_interval_div_constant(
        interval_lower=torch.tensor([6.0, 6.0]),
        interval_upper=torch.tensor([12.0, 12.0]),
        constant=torch.tensor([3.0, -3.0]),
    )

    # First element: [6, 12] / 3 = [2, 4]
    assert torch.allclose(out.lower[0], torch.tensor(2.0))
    assert torch.allclose(out.upper[0], torch.tensor(4.0))

    # Second element: [6, 12] / (-3) = [-4, -2]
    assert torch.allclose(out.lower[1], torch.tensor(-4.0))
    assert torch.allclose(out.upper[1], torch.tensor(-2.0))


def test_div_reciprocal_property() -> None:
    """Test that a / b * b approximately contains a (for non-zero b)."""
    a = IntervalBounds(torch.tensor([2.0]), torch.tensor([5.0]))
    b = IntervalBounds(torch.tensor([1.0]), torch.tensor([2.0]))

    div_strategy = IBPDiv()

    mul_strategy = IBPMul()

    # a / b
    a_div_b = propagate(div_strategy, a, b)

    # (a / b) * b
    result = propagate(mul_strategy, a_div_b, b)

    # Result should contain original interval a (may be wider due to overestimation)
    assert torch.all(result.lower <= a.lower)
    assert torch.all(result.upper >= a.upper)
