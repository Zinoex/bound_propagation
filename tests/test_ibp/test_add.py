from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.linear import IBPAdd
from tests.helpers import propagate


def _propagate_interval_interval(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> IntervalBounds:
    """Propagate bounds for interval + interval operation."""
    strategy = IBPAdd()
    left_bounds = IntervalBounds(lower=left_lower, upper=left_upper)
    right_bounds = IntervalBounds(lower=right_lower, upper=right_upper)
    return propagate(strategy, left_bounds, right_bounds)


def _propagate_interval_constant(
    interval_lower: torch.Tensor, interval_upper: torch.Tensor, constant: torch.Tensor | float
) -> IntervalBounds:
    """Propagate bounds for interval + constant operation."""
    strategy = IBPAdd()
    interval_bounds = IntervalBounds(lower=interval_lower, upper=interval_upper)
    return propagate(strategy, interval_bounds, constant)


def test_add_positive_intervals() -> None:
    """Test addition of two positive intervals."""
    # [2, 5] + [3, 7] = [5, 12]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([2.0]),
        left_upper=torch.tensor([5.0]),
        right_lower=torch.tensor([3.0]),
        right_upper=torch.tensor([7.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([5.0]))
    assert torch.allclose(out.upper, torch.tensor([12.0]))


def test_add_negative_intervals() -> None:
    """Test addition of two negative intervals."""
    # [-5, -2] + [-7, -3] = [-12, -5]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([-5.0]),
        left_upper=torch.tensor([-2.0]),
        right_lower=torch.tensor([-7.0]),
        right_upper=torch.tensor([-3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-12.0]))
    assert torch.allclose(out.upper, torch.tensor([-5.0]))


def test_add_mixed_sign_intervals() -> None:
    """Test addition of intervals with mixed signs."""
    # [-3, 4] + [-2, 5] = [-5, 9]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([-3.0]),
        left_upper=torch.tensor([4.0]),
        right_lower=torch.tensor([-2.0]),
        right_upper=torch.tensor([5.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-5.0]))
    assert torch.allclose(out.upper, torch.tensor([9.0]))


def test_add_zero_intervals() -> None:
    """Test addition with zero-containing intervals."""
    # [0, 0] + [1, 2] = [1, 2]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([0.0]),
        left_upper=torch.tensor([0.0]),
        right_lower=torch.tensor([1.0]),
        right_upper=torch.tensor([2.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([1.0]))
    assert torch.allclose(out.upper, torch.tensor([2.0]))


def test_add_point_intervals() -> None:
    """Test addition of point intervals (lower = upper)."""
    # [3, 3] + [5, 5] = [8, 8]
    out = _propagate_interval_interval(
        left_lower=torch.tensor([3.0]),
        left_upper=torch.tensor([3.0]),
        right_lower=torch.tensor([5.0]),
        right_upper=torch.tensor([5.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([8.0]))
    assert torch.allclose(out.upper, torch.tensor([8.0]))


def test_add_batched_intervals() -> None:
    """Test addition with batched intervals."""
    # Batch of different interval pairs
    out = _propagate_interval_interval(
        left_lower=torch.tensor([1.0, -2.0, 0.0]),
        left_upper=torch.tensor([2.0, 3.0, 1.0]),
        right_lower=torch.tensor([0.5, -1.0, -0.5]),
        right_upper=torch.tensor([1.5, 1.0, 0.5]),
    )

    expected_lower = torch.tensor([1.5, -3.0, -0.5])
    expected_upper = torch.tensor([3.5, 4.0, 1.5])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_add_multidimensional_intervals() -> None:
    """Test addition with multi-dimensional intervals."""
    out = _propagate_interval_interval(
        left_lower=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        left_upper=torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        right_lower=torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
        right_upper=torch.tensor([[1.0, 1.5], [2.0, 2.5]]),
    )

    expected_lower = torch.tensor([[1.5, 3.0], [4.5, 6.0]])
    expected_upper = torch.tensor([[3.0, 4.5], [6.0, 7.5]])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_add_with_constant_positive() -> None:
    """Test addition of interval with positive constant."""
    # [1, 3] + 5 = [6, 8]
    out = _propagate_interval_constant(
        interval_lower=torch.tensor([1.0]),
        interval_upper=torch.tensor([3.0]),
        constant=5.0,
    )

    assert torch.allclose(out.lower, torch.tensor([6.0]))
    assert torch.allclose(out.upper, torch.tensor([8.0]))


def test_add_with_constant_negative() -> None:
    """Test addition of interval with negative constant."""
    # [2, 4] + (-3) = [-1, 1]
    out = _propagate_interval_constant(
        interval_lower=torch.tensor([2.0]),
        interval_upper=torch.tensor([4.0]),
        constant=-3.0,
    )

    assert torch.allclose(out.lower, torch.tensor([-1.0]))
    assert torch.allclose(out.upper, torch.tensor([1.0]))


def test_add_with_constant_zero() -> None:
    """Test addition of interval with zero constant."""
    # [1.5, 2.5] + 0 = [1.5, 2.5]
    out = _propagate_interval_constant(
        interval_lower=torch.tensor([1.5]),
        interval_upper=torch.tensor([2.5]),
        constant=0.0,
    )

    assert torch.allclose(out.lower, torch.tensor([1.5]))
    assert torch.allclose(out.upper, torch.tensor([2.5]))


def test_add_with_tensor_constant() -> None:
    """Test addition of interval with tensor constant."""
    # [1, 2] + [3, 4, 5] should broadcast
    out = _propagate_interval_constant(
        interval_lower=torch.tensor([[1.0], [2.0]]),
        interval_upper=torch.tensor([[2.0], [3.0]]),
        constant=torch.tensor([1.0, 2.0, 3.0]),
    )

    expected_lower = torch.tensor([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    expected_upper = torch.tensor([[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_add_associativity_property() -> None:
    """Test that interval addition preserves soundness for associativity."""
    # While interval arithmetic is not associative in general,
    # we verify that (a + b) + c contains the result of a + (b + c)
    a = IntervalBounds(torch.tensor([1.0]), torch.tensor([2.0]))
    b = IntervalBounds(torch.tensor([0.5]), torch.tensor([1.5]))
    c = IntervalBounds(torch.tensor([0.2]), torch.tensor([0.8]))

    strategy = IBPAdd()

    # (a + b) + c
    ab = propagate(strategy, a, b)
    ab_c = propagate(strategy, ab, c)

    # a + (b + c)
    bc = propagate(strategy, b, c)
    a_bc = propagate(strategy, a, bc)

    # Both should give the same result for addition (unlike multiplication)
    assert torch.allclose(ab_c.lower, a_bc.lower)
    assert torch.allclose(ab_c.upper, a_bc.upper)


def test_add_commutativity_property() -> None:
    """Test that interval addition is commutative."""
    a = IntervalBounds(torch.tensor([1.0, -2.0]), torch.tensor([3.0, 1.0]))
    b = IntervalBounds(torch.tensor([0.5, -1.0]), torch.tensor([2.0, 0.5]))

    strategy = IBPAdd()

    # a + b
    ab = propagate(strategy, a, b)
    # b + a
    ba = propagate(strategy, b, a)

    assert torch.allclose(ab.lower, ba.lower)
    assert torch.allclose(ab.upper, ba.upper)
