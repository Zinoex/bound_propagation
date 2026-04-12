from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.sub import IBPSub, IBPSubConstantLeft, IBPSubConstantRight


def _propagate_interval_interval(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> IntervalBounds:
    """Propagate bounds for interval - interval operation."""
    strategy = IBPSub()
    left_bounds = IntervalBounds(lower=left_lower, upper=left_upper)
    right_bounds = IntervalBounds(lower=right_lower, upper=right_upper)
    return strategy.propagate_forwards(node=None, input_bounds=[left_bounds, right_bounds])  # ty:ignore[invalid-argument-type]


def _propagate_interval_minus_constant(
    interval_lower: torch.Tensor, interval_upper: torch.Tensor, constant: torch.Tensor | float
) -> IntervalBounds:
    """Propagate bounds for interval - constant operation."""
    strategy = IBPSubConstantRight()
    interval_bounds = IntervalBounds(lower=interval_lower, upper=interval_upper)
    return strategy.propagate_forwards(node=None, input_bounds=[interval_bounds, constant])  # ty:ignore[invalid-argument-type]


def _propagate_constant_minus_interval(
    constant: torch.Tensor | float, interval_lower: torch.Tensor, interval_upper: torch.Tensor
) -> IntervalBounds:
    """Propagate bounds for constant - interval operation."""
    strategy = IBPSubConstantLeft()
    interval_bounds = IntervalBounds(lower=interval_lower, upper=interval_upper)
    return strategy.propagate_forwards(node=None, input_bounds=[constant, interval_bounds])  # ty:ignore[invalid-argument-type]


def test_sub_positive_intervals() -> None:
    """Test subtraction of two positive intervals."""
    # [5, 8] - [2, 3] = [2, 6]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([5.0]),
        left_upper=torch.tensor([8.0]),
        right_lower=torch.tensor([2.0]),
        right_upper=torch.tensor([3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([6.0]))


def test_sub_negative_intervals() -> None:
    """Test subtraction of two negative intervals."""
    # [-3, -1] - [-5, -2] = [1, 4]
    # Lower: -3 - (-2) = -1, but correct is -3 - (-5) = 2
    # Actually: a - b where a in [-3, -1], b in [-5, -2]
    # min: -3 - (-2) = -1, max: -1 - (-5) = 4
    out = _propagate_interval_interval(
        left_lower=torch.tensor([-3.0]),
        left_upper=torch.tensor([-1.0]),
        right_lower=torch.tensor([-5.0]),
        right_upper=torch.tensor([-2.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-1.0]))  # -3 - (-2)
    assert torch.allclose(out.upper, torch.tensor([4.0]))  # -1 - (-5)


def test_sub_mixed_sign_intervals() -> None:
    """Test subtraction of intervals with mixed signs."""
    # [-2, 3] - [-1, 4] = [-6, 4]
    # Lower: -2 - 4 = -6
    # Upper: 3 - (-1) = 4
    out = _propagate_interval_interval(
        left_lower=torch.tensor([-2.0]),
        left_upper=torch.tensor([3.0]),
        right_lower=torch.tensor([-1.0]),
        right_upper=torch.tensor([4.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-6.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0]))


def test_sub_zero_intervals() -> None:
    """Test subtraction with zero-containing intervals."""
    # [0, 0] - [1, 2] = [-2, -1]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([0.0]),
        left_upper=torch.tensor([0.0]),
        right_lower=torch.tensor([1.0]),
        right_upper=torch.tensor([2.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-2.0]))
    assert torch.allclose(out.upper, torch.tensor([-1.0]))


def test_sub_point_intervals() -> None:
    """Test subtraction of point intervals (lower = upper)."""
    # [7, 7] - [3, 3] = [4, 4]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([7.0]),
        left_upper=torch.tensor([7.0]),
        right_lower=torch.tensor([3.0]),
        right_upper=torch.tensor([3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([4.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0]))


def test_sub_batched_intervals() -> None:
    """Test subtraction with batched intervals."""
    out = _propagate_interval_interval(
        left_lower=torch.tensor([5.0, -2.0, 0.0]),
        left_upper=torch.tensor([8.0, 3.0, 2.0]),
        right_lower=torch.tensor([1.0, -1.0, -1.0]),
        right_upper=torch.tensor([2.0, 1.0, 0.5]),
    )

    # [5, 8] - [1, 2] = [3, 7]
    # [-2, 3] - [-1, 1] = [-3, 4]
    # [0, 2] - [-1, 0.5] = [-0.5, 3]
    expected_lower = torch.tensor([3.0, -3.0, -0.5])
    expected_upper = torch.tensor([7.0, 4.0, 3.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sub_interval_minus_constant_positive() -> None:
    """Test subtraction of positive constant from interval."""
    # [5, 8] - 3 = [2, 5]
    out = _propagate_interval_minus_constant(
        interval_lower=torch.tensor([5.0]),
        interval_upper=torch.tensor([8.0]),
        constant=3.0,
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_sub_interval_minus_constant_negative() -> None:
    """Test subtraction of negative constant from interval."""
    # [2, 4] - (-3) = [5, 7]
    out = _propagate_interval_minus_constant(
        interval_lower=torch.tensor([2.0]),
        interval_upper=torch.tensor([4.0]),
        constant=-3.0,
    )

    assert torch.allclose(out.lower, torch.tensor([5.0]))
    assert torch.allclose(out.upper, torch.tensor([7.0]))


def test_sub_constant_minus_interval_positive() -> None:
    """Test subtraction of interval from positive constant."""
    # 10 - [2, 5] = [5, 8]
    out = _propagate_constant_minus_interval(
        constant=10.0,
        interval_lower=torch.tensor([2.0]),
        interval_upper=torch.tensor([5.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([5.0]))
    assert torch.allclose(out.upper, torch.tensor([8.0]))


def test_sub_constant_minus_interval_negative() -> None:
    """Test subtraction of interval from negative constant."""
    # -5 - [1, 3] = [-8, -6]
    out = _propagate_constant_minus_interval(
        constant=-5.0,
        interval_lower=torch.tensor([1.0]),
        interval_upper=torch.tensor([3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-8.0]))
    assert torch.allclose(out.upper, torch.tensor([-6.0]))


def test_sub_constant_minus_zero_interval() -> None:
    """Test subtraction of zero interval from constant."""
    # 7 - [0, 0] = [7, 7]
    out = _propagate_constant_minus_interval(
        constant=7.0,
        interval_lower=torch.tensor([0.0]),
        interval_upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([7.0]))
    assert torch.allclose(out.upper, torch.tensor([7.0]))


def test_sub_non_commutativity() -> None:
    """Test that interval subtraction is NOT commutative."""
    a = IntervalBounds(torch.tensor([1.0]), torch.tensor([3.0]))
    b = IntervalBounds(torch.tensor([0.5]), torch.tensor([2.0]))

    strategy = IBPSub()

    # a - b
    ab = strategy.propagate_forwards(None, [a, b])  # ty:ignore[invalid-argument-type]
    # b - a
    ba = strategy.propagate_forwards(None, [b, a])  # ty:ignore[invalid-argument-type]

    # These should be different
    # a - b = [1, 3] - [0.5, 2] = [-1, 2.5]
    # b - a = [0.5, 2] - [1, 3] = [-2.5, 1]
    assert not torch.allclose(ab.lower, ba.lower)
    assert not torch.allclose(ab.upper, ba.upper)

    # But they should satisfy: a - b = -(b - a)
    assert torch.allclose(ab.lower, -ba.upper)
    assert torch.allclose(ab.upper, -ba.lower)


def test_sub_self_contains_zero() -> None:
    """Test that a - a contains zero."""
    a = IntervalBounds(torch.tensor([1.0, -2.0, 0.5]), torch.tensor([3.0, 1.0, 2.0]))

    strategy = IBPSub()
    result = strategy.propagate_forwards(None, [a, a])  # ty:ignore[invalid-argument-type]

    # a - a should contain zero for all elements
    assert torch.all(result.lower <= 0.0)
    assert torch.all(result.upper >= 0.0)
