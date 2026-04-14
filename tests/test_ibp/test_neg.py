from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.add import IBPAdd
from bound_propagation.propagation.ibp.neg import IBPNeg

from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for neg operation."""
    strategy = IBPNeg()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_neg_positive_interval() -> None:
    """Test negation of positive interval."""
    # -[2, 5] = [-5, -2]
    out = _propagate(
        lower=torch.tensor([2.0]),
        upper=torch.tensor([5.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-5.0]))
    assert torch.allclose(out.upper, torch.tensor([-2.0]))


def test_neg_negative_interval() -> None:
    """Test negation of negative interval."""
    # -[-5, -2] = [2, 5]
    out = _propagate(
        lower=torch.tensor([-5.0]),
        upper=torch.tensor([-2.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_neg_mixed_sign_interval() -> None:
    """Test negation of interval with mixed signs."""
    # -[-3, 4] = [-4, 3]
    out = _propagate(
        lower=torch.tensor([-3.0]),
        upper=torch.tensor([4.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-4.0]))
    assert torch.allclose(out.upper, torch.tensor([3.0]))


def test_neg_zero_interval() -> None:
    """Test negation of zero interval."""
    # -[0, 0] = [0, 0]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_neg_symmetric_interval() -> None:
    """Test negation of symmetric interval around zero."""
    # -[-5, 5] = [-5, 5]
    out = _propagate(
        lower=torch.tensor([-5.0]),
        upper=torch.tensor([5.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-5.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_neg_point_interval() -> None:
    """Test negation of point interval (lower = upper)."""
    # -[3, 3] = [-3, -3]
    out = _propagate(
        lower=torch.tensor([3.0]),
        upper=torch.tensor([3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([-3.0]))
    assert torch.allclose(out.upper, torch.tensor([-3.0]))


def test_neg_batched_intervals() -> None:
    """Test negation with batched intervals."""
    out = _propagate(
        lower=torch.tensor([2.0, -5.0, -3.0, 0.0]),
        upper=torch.tensor([5.0, -2.0, 4.0, 3.0]),
    )

    # -[2, 5] = [-5, -2]
    # -[-5, -2] = [2, 5]
    # -[-3, 4] = [-4, 3]
    # -[0, 3] = [-3, 0]
    expected_lower = torch.tensor([-5.0, 2.0, -4.0, -3.0])
    expected_upper = torch.tensor([-2.0, 5.0, 3.0, 0.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_neg_multidimensional() -> None:
    """Test negation with multi-dimensional intervals."""
    out = _propagate(
        lower=torch.tensor([[-3.0, 1.0], [2.0, -5.0]]),
        upper=torch.tensor([[4.0, 6.0], [7.0, -1.0]]),
    )

    expected_lower = torch.tensor([[-4.0, -6.0], [-7.0, 1.0]])
    expected_upper = torch.tensor([[3.0, -1.0], [-2.0, 5.0]])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_neg_involutory_property() -> None:
    """Test that -(-x) = x (negation is involutory)."""
    strategy = IBPNeg()
    a = IntervalBounds(torch.tensor([-3.0, 1.0, -5.0]), torch.tensor([4.0, 6.0, 2.0]))

    # -a
    neg_a = propagate(strategy, a)

    # -(-a)
    neg_neg_a = propagate(strategy, neg_a)

    # Should recover original interval
    assert torch.allclose(a.lower, neg_neg_a.lower)
    assert torch.allclose(a.upper, neg_neg_a.upper)


def test_neg_width_preservation() -> None:
    """Test that negation preserves interval width."""
    lower = torch.tensor([1.0, -5.0, -2.0])
    upper = torch.tensor([4.0, -1.0, 3.0])

    original_width = upper - lower
    out = _propagate(lower, upper)
    result_width = out.upper - out.lower

    assert torch.allclose(original_width, result_width)


def test_neg_with_addition_property() -> None:
    """Test that -(a + b) = -a + -b for intervals."""
    a = IntervalBounds(torch.tensor([1.0]), torch.tensor([3.0]))
    b = IntervalBounds(torch.tensor([2.0]), torch.tensor([5.0]))

    add_strategy = IBPAdd()
    neg_strategy = IBPNeg()

    # a + b
    a_plus_b = propagate(add_strategy, a, b)

    # -(a + b)
    neg_sum = propagate(neg_strategy, a_plus_b)

    # -a
    neg_a = propagate(neg_strategy, a)

    # -b
    neg_b = propagate(neg_strategy, b)

    # -a + -b
    sum_neg = propagate(add_strategy, neg_a, neg_b)

    # Should be equal
    assert torch.allclose(neg_sum.lower, sum_neg.lower)
    assert torch.allclose(neg_sum.upper, sum_neg.upper)
