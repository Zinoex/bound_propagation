from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.linear import IBPAdd
from bound_propagation.propagation.ibp.pairwise import IBPMul
from tests.helpers import propagate


def _propagate_interval_interval(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> IntervalBounds:
    """Propagate bounds for interval * interval operation."""
    strategy = IBPMul()
    left_bounds = IntervalBounds(lower=left_lower, upper=left_upper)
    right_bounds = IntervalBounds(lower=right_lower, upper=right_upper)
    return propagate(strategy, left_bounds, right_bounds)


def _propagate_interval_constant(
    interval_lower: torch.Tensor, interval_upper: torch.Tensor, constant: torch.Tensor | float
) -> IntervalBounds:
    """Propagate bounds for interval * constant operation."""
    strategy = IBPMul()
    interval_bounds = IntervalBounds(lower=interval_lower, upper=interval_upper)
    return propagate(strategy, interval_bounds, constant)


def test_mul_positive_intervals() -> None:
    """Test multiplication of two positive intervals."""
    # [2, 3] * [4, 5] = [8, 15]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([2.0]),
        left_upper=torch.tensor([3.0]),
        right_lower=torch.tensor([4.0]),
        right_upper=torch.tensor([5.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([8.0]))
    assert torch.allclose(out.upper, torch.tensor([15.0]))


def test_mul_negative_intervals() -> None:
    """Test multiplication of two negative intervals."""
    # [-4, -2] * [-5, -3] = [6, 20]
    # Products: (-4)*(-5)=20, (-4)*(-3)=12, (-2)*(-5)=10, (-2)*(-3)=6
    # min=6, max=20
    out = _propagate_interval_interval(
        left_lower=torch.tensor([-4.0]),
        left_upper=torch.tensor([-2.0]),
        right_lower=torch.tensor([-5.0]),
        right_upper=torch.tensor([-3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([6.0]))
    assert torch.allclose(out.upper, torch.tensor([20.0]))


def test_mul_positive_negative_intervals() -> None:
    """Test multiplication of positive and negative intervals."""
    # [2, 4] * [-3, -1] = [-12, -2]
    # Products: 2*(-3)=-6, 2*(-1)=-2, 4*(-3)=-12, 4*(-1)=-4
    # min=-12, max=-2
    out = _propagate_interval_interval(
        left_lower=torch.tensor([2.0]),
        left_upper=torch.tensor([4.0]),
        right_lower=torch.tensor([-3.0]),
        right_upper=torch.tensor([-1.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-12.0]))
    assert torch.allclose(out.upper, torch.tensor([-2.0]))


def test_mul_mixed_sign_intervals() -> None:
    """Test multiplication of intervals with mixed signs."""
    # [-2, 3] * [-1, 4] = [-8, 12]
    # Products: (-2)*(-1)=2, (-2)*4=-8, 3*(-1)=-3, 3*4=12
    # min=-8, max=12
    out = _propagate_interval_interval(
        left_lower=torch.tensor([-2.0]),
        left_upper=torch.tensor([3.0]),
        right_lower=torch.tensor([-1.0]),
        right_upper=torch.tensor([4.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-8.0]))
    assert torch.allclose(out.upper, torch.tensor([12.0]))


def test_mul_zero_containing_intervals() -> None:
    """Test multiplication with zero-containing intervals."""
    # [-1, 2] * [0, 3] = [-3, 6]
    # Products: (-1)*0=0, (-1)*3=-3, 2*0=0, 2*3=6
    # min=-3, max=6
    out = _propagate_interval_interval(
        left_lower=torch.tensor([-1.0]),
        left_upper=torch.tensor([2.0]),
        right_lower=torch.tensor([0.0]),
        right_upper=torch.tensor([3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-3.0]))
    assert torch.allclose(out.upper, torch.tensor([6.0]))


def test_mul_zero_intervals() -> None:
    """Test multiplication with zero interval [0, 0]."""
    # [2, 5] * [0, 0] = [0, 0]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([2.0]),
        left_upper=torch.tensor([5.0]),
        right_lower=torch.tensor([0.0]),
        right_upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_mul_point_intervals() -> None:
    """Test multiplication of point intervals (lower = upper)."""
    # [3, 3] * [4, 4] = [12, 12]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([3.0]),
        left_upper=torch.tensor([3.0]),
        right_lower=torch.tensor([4.0]),
        right_upper=torch.tensor([4.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([12.0]))
    assert torch.allclose(out.upper, torch.tensor([12.0]))


def test_mul_batched_intervals() -> None:
    """Test multiplication with batched intervals."""
    out = _propagate_interval_interval(
        left_lower=torch.tensor([2.0, -3.0, -1.0, 1.0]),
        left_upper=torch.tensor([4.0, -1.0, 2.0, 3.0]),
        right_lower=torch.tensor([1.0, -4.0, -2.0, 0.0]),
        right_upper=torch.tensor([3.0, -2.0, 1.0, 5.0]),
    )

    # [2, 4] * [1, 3]: min=2, max=12
    # [-3, -1] * [-4, -2]: min=2, max=12
    # [-1, 2] * [-2, 1]: min=-4, max=2
    # [1, 3] * [0, 5]: min=0, max=15
    expected_lower = torch.tensor([2.0, 2.0, -4.0, 0.0])
    expected_upper = torch.tensor([12.0, 12.0, 2.0, 15.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_mul_with_constant_positive() -> None:
    """Test multiplication of interval with positive constant."""
    # [1, 3] * 2 = [2, 6]
    out = _propagate_interval_constant(
        interval_lower=torch.tensor([1.0]),
        interval_upper=torch.tensor([3.0]),
        constant=2.0,
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([6.0]))


def test_mul_with_constant_negative() -> None:
    """Test multiplication of interval with negative constant."""
    # [2, 5] * (-3) = [-15, -6]
    out = _propagate_interval_constant(
        interval_lower=torch.tensor([2.0]),
        interval_upper=torch.tensor([5.0]),
        constant=-3.0,
    )

    assert torch.allclose(out.lower, torch.tensor([-15.0]))
    assert torch.allclose(out.upper, torch.tensor([-6.0]))


def test_mul_with_constant_zero() -> None:
    """Test multiplication of interval with zero constant."""
    # [1.5, 2.5] * 0 = [0, 0]
    out = _propagate_interval_constant(
        interval_lower=torch.tensor([1.5]),
        interval_upper=torch.tensor([2.5]),
        constant=0.0,
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_mul_with_constant_mixed_sign_interval() -> None:
    """Test multiplication of mixed-sign interval with constant."""
    # [-2, 3] * 4 = [-8, 12]
    out = _propagate_interval_constant(
        interval_lower=torch.tensor([-2.0]),
        interval_upper=torch.tensor([3.0]),
        constant=4.0,
    )

    assert torch.allclose(out.lower, torch.tensor([-8.0]))
    assert torch.allclose(out.upper, torch.tensor([12.0]))

    # [-2, 3] * (-4) = [-12, 8]
    out_neg = _propagate_interval_constant(
        interval_lower=torch.tensor([-2.0]),
        interval_upper=torch.tensor([3.0]),
        constant=-4.0,
    )

    assert torch.allclose(out_neg.lower, torch.tensor([-12.0]))
    assert torch.allclose(out_neg.upper, torch.tensor([8.0]))


def test_mul_commutativity() -> None:
    """Test that interval multiplication is commutative."""
    a = IntervalBounds(torch.tensor([-1.0, 2.0]), torch.tensor([2.0, 5.0]))
    b = IntervalBounds(torch.tensor([0.5, -3.0]), torch.tensor([3.0, -1.0]))

    strategy = IBPMul()

    # a * b
    ab = propagate(strategy, a, b)
    # b * a
    ba = propagate(strategy, b, a)

    assert torch.allclose(ab.lower, ba.lower)
    assert torch.allclose(ab.upper, ba.upper)


def test_mul_distributivity_over_addition_inclusion() -> None:
    """Test that mul distributes over add with interval inclusion."""
    # a * (b + c) should be contained in a*b + a*c (or vice versa)
    # Actually interval multiplication is subdistributive
    a = IntervalBounds(torch.tensor([2.0]), torch.tensor([3.0]))
    b = IntervalBounds(torch.tensor([1.0]), torch.tensor([2.0]))
    c = IntervalBounds(torch.tensor([0.5]), torch.tensor([1.5]))

    mul_strategy = IBPMul()

    add_strategy = IBPAdd()

    # a * (b + c)
    bc = propagate(add_strategy, b, c)
    a_bc = propagate(mul_strategy, a, bc)

    # a*b + a*c
    ab = propagate(mul_strategy, a, b)
    ac = propagate(mul_strategy, a, c)
    ab_ac = propagate(add_strategy, ab, ac)

    # The results should be close (may have slight overestimation in one direction)
    # For positive values, they should be equal or very close
    assert torch.allclose(a_bc.lower, ab_ac.lower, atol=1e-6)
    assert torch.allclose(a_bc.upper, ab_ac.upper, atol=1e-6)


def test_mul_unit_interval() -> None:
    """Test multiplication by unit interval [1, 1]."""
    a = IntervalBounds(torch.tensor([2.0, -3.0]), torch.tensor([5.0, 1.0]))
    unit = IntervalBounds(torch.tensor([1.0]), torch.tensor([1.0]))

    strategy = IBPMul()
    result = propagate(strategy, a, unit)

    # Should return the same interval (with broadcasting)
    assert torch.allclose(result.lower, a.lower)
    assert torch.allclose(result.upper, a.upper)
