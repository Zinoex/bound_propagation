from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.elementwise import IBPRelu
from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for relu operation."""
    strategy = IBPRelu()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_relu_positive_interval() -> None:
    """Test relu of positive interval."""
    # relu([2, 5]) = [2, 5]
    out = _propagate(
        lower=torch.tensor([2.0]),
        upper=torch.tensor([5.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_relu_negative_interval() -> None:
    """Test relu of negative interval."""
    # relu([-5, -2]) = [0, 0]
    out = _propagate(
        lower=torch.tensor([-5.0]),
        upper=torch.tensor([-2.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_relu_mixed_sign_interval() -> None:
    """Test relu of interval with mixed signs."""
    # relu([-3, 4]) = [0, 4]
    out = _propagate(
        lower=torch.tensor([-3.0]),
        upper=torch.tensor([4.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0]))


def test_relu_zero_interval() -> None:
    """Test relu of zero interval."""
    # relu([0, 0]) = [0, 0]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_relu_interval_from_zero() -> None:
    """Test relu of interval starting at zero."""
    # relu([0, 5]) = [0, 5]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([5.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_relu_interval_to_zero() -> None:
    """Test relu of interval ending at zero."""
    # relu([-5, 0]) = [0, 0]
    out = _propagate(
        lower=torch.tensor([-5.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_relu_point_interval_positive() -> None:
    """Test relu of positive point interval."""
    # relu([3, 3]) = [3, 3]
    out = _propagate(
        lower=torch.tensor([3.0]),
        upper=torch.tensor([3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([3.0]))
    assert torch.allclose(out.upper, torch.tensor([3.0]))


def test_relu_point_interval_negative() -> None:
    """Test relu of negative point interval."""
    # relu([-3, -3]) = [0, 0]
    out = _propagate(
        lower=torch.tensor([-3.0]),
        upper=torch.tensor([-3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_relu_batched_intervals() -> None:
    """Test relu with batched intervals."""
    out = _propagate(
        lower=torch.tensor([2.0, -5.0, -3.0, 0.0, -1.0]),
        upper=torch.tensor([5.0, -2.0, 4.0, 3.0, 0.0]),
    )

    # [2, 5] -> [2, 5]
    # [-5, -2] -> [0, 0]
    # [-3, 4] -> [0, 4]
    # [0, 3] -> [0, 3]
    # [-1, 0] -> [0, 0]
    expected_lower = torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0])
    expected_upper = torch.tensor([5.0, 0.0, 4.0, 3.0, 0.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_relu_multidimensional() -> None:
    """Test relu with multi-dimensional intervals."""
    out = _propagate(
        lower=torch.tensor([[-3.0, 1.0], [2.0, -5.0]]),
        upper=torch.tensor([[4.0, 6.0], [7.0, -1.0]]),
    )

    expected_lower = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    expected_upper = torch.tensor([[4.0, 6.0], [7.0, 0.0]])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_relu_non_negativity() -> None:
    """Test that relu always returns non-negative bounds."""
    # Test various intervals
    test_cases = [
        torch.tensor([[-100.0, 50.0, -10.0]]),
        torch.tensor([[1.0, -200.0, 0.0]]),
    ]

    for lower in test_cases:
        upper = lower + 10.0  # Create valid upper bound
        out = _propagate(lower, upper)

        # Both bounds should always be >= 0
        assert torch.all(out.lower >= 0.0)
        assert torch.all(out.upper >= 0.0)


def test_relu_monotonicity() -> None:
    """Test that relu is monotonically increasing."""
    # If [a, b] ⊆ [c, d], then relu([a, b]) ⊆ relu([c, d])
    inner = IntervalBounds(torch.tensor([1.0, -2.0]), torch.tensor([3.0, 1.0]))
    outer = IntervalBounds(torch.tensor([0.0, -5.0]), torch.tensor([5.0, 2.0]))

    strategy = IBPRelu()

    out_inner = propagate(strategy, inner)
    out_outer = propagate(strategy, outer)

    # relu([1, 3]) = [1, 3] should be contained in relu([0, 5]) = [0, 5]
    # relu([-2, 1]) = [0, 1] should be contained in relu([-5, 2]) = [0, 2]
    assert torch.all(out_outer.lower <= out_inner.lower)
    assert torch.all(out_outer.upper >= out_inner.upper)


def test_relu_idempotency() -> None:
    """Test that relu(relu(x)) = relu(x)."""
    strategy = IBPRelu()
    a = IntervalBounds(torch.tensor([-3.0, 1.0, -5.0]), torch.tensor([4.0, 6.0, 2.0]))

    # relu(a)
    relu_a = propagate(strategy, a)

    # relu(relu(a))
    relu_relu_a = propagate(strategy, relu_a)

    # Should be the same
    assert torch.allclose(relu_a.lower, relu_relu_a.lower)
    assert torch.allclose(relu_a.upper, relu_relu_a.upper)


def test_relu_preserves_positive_part() -> None:
    """Test that relu preserves the positive part of an interval."""
    # For x >= 0, relu(x) = x
    positive = IntervalBounds(torch.tensor([1.0, 2.0, 0.5]), torch.tensor([3.0, 5.0, 1.5]))

    strategy = IBPRelu()
    result = propagate(strategy, positive)

    assert torch.allclose(result.lower, positive.lower)
    assert torch.allclose(result.upper, positive.upper)


def test_relu_zeros_negative_part() -> None:
    """Test that relu zeros out the completely negative part."""
    # For x <= 0, relu(x) = 0
    negative = IntervalBounds(torch.tensor([-5.0, -3.0, -1.0]), torch.tensor([-2.0, -1.0, 0.0]))

    strategy = IBPRelu()
    result = propagate(strategy, negative)

    assert torch.allclose(result.lower, torch.zeros_like(result.lower))
    assert torch.allclose(result.upper, torch.zeros_like(result.upper))
