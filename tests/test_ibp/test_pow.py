from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.pow import IBPPow


def _propagate(lower: torch.Tensor, upper: torch.Tensor, power: int | float | torch.Tensor) -> IntervalBounds:
    """Propagate bounds for pow operation."""
    strategy = IBPPow()
    bounds = IntervalBounds(lower=lower, upper=upper)

    # Create a minimal mock node with attributes
    class MockNode:
        def __init__(self):
            self.attributes = {"power": power}

    node = MockNode()
    return strategy.propagate_forwards(node, input_bounds=[bounds])  # ty:ignore[invalid-argument-type]


def test_pow_positive_interval_odd_power() -> None:
    """Test pow of positive interval with odd power."""
    # [2, 3]^3 = [8, 27]
    out = _propagate(
        lower=torch.tensor([2.0]),
        upper=torch.tensor([3.0]),
        power=3,
    )

    assert torch.allclose(out.lower, torch.tensor([8.0]))
    assert torch.allclose(out.upper, torch.tensor([27.0]))


def test_pow_positive_interval_even_power() -> None:
    """Test pow of positive interval with even power."""
    # [2, 3]^2 = [4, 9]
    out = _propagate(
        lower=torch.tensor([2.0]),
        upper=torch.tensor([3.0]),
        power=2,
    )

    assert torch.allclose(out.lower, torch.tensor([4.0]))
    assert torch.allclose(out.upper, torch.tensor([9.0]))


def test_pow_negative_interval_odd_power() -> None:
    """Test pow of negative interval with odd power."""
    # [-3, -2]^3 = [-27, -8]
    out = _propagate(
        lower=torch.tensor([-3.0]),
        upper=torch.tensor([-2.0]),
        power=3,
    )

    assert torch.allclose(out.lower, torch.tensor([-27.0]))
    assert torch.allclose(out.upper, torch.tensor([-8.0]))


def test_pow_negative_interval_even_power() -> None:
    """Test pow of negative interval with even power."""
    # [-3, -2]^2 = [4, 9]
    out = _propagate(
        lower=torch.tensor([-3.0]),
        upper=torch.tensor([-2.0]),
        power=2,
    )

    assert torch.allclose(out.lower, torch.tensor([4.0]))
    assert torch.allclose(out.upper, torch.tensor([9.0]))


def test_pow_mixed_sign_interval_odd_power() -> None:
    """Test pow of mixed-sign interval with odd power."""
    # [-2, 3]^3 = [-8, 27]
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([3.0]),
        power=3,
    )

    assert torch.allclose(out.lower, torch.tensor([-8.0]))
    assert torch.allclose(out.upper, torch.tensor([27.0]))


def test_pow_mixed_sign_interval_even_power() -> None:
    """Test pow of mixed-sign interval with even power (includes zero)."""
    # [-2, 3]^2 = [0, 9]
    # Lower bound should be 0 since interval crosses zero
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([3.0]),
        power=2,
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([9.0]))


def test_pow_symmetric_interval_even_power() -> None:
    """Test pow of symmetric interval with even power."""
    # [-3, 3]^2 = [0, 9]
    out = _propagate(
        lower=torch.tensor([-3.0]),
        upper=torch.tensor([3.0]),
        power=2,
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([9.0]))


def test_pow_zero_interval() -> None:
    """Test pow of zero interval."""
    # [0, 0]^3 = [0, 0]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
        power=3,
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_pow_power_of_one() -> None:
    """Test x^1 = x."""
    out = _propagate(
        lower=torch.tensor([-2.0, 1.0]),
        upper=torch.tensor([3.0, 5.0]),
        power=1,
    )

    assert torch.allclose(out.lower, torch.tensor([-2.0, 1.0]))
    assert torch.allclose(out.upper, torch.tensor([3.0, 5.0]))


def test_pow_power_of_zero() -> None:
    """Test x^0 = 1."""
    out = _propagate(
        lower=torch.tensor([-2.0, 1.0]),
        upper=torch.tensor([3.0, 5.0]),
        power=0,
    )

    assert torch.allclose(out.lower, torch.tensor([1.0, 1.0]))
    assert torch.allclose(out.upper, torch.tensor([1.0, 1.0]))


def test_pow_batched_intervals_odd_power() -> None:
    """Test pow with batched intervals and odd power."""
    out = _propagate(
        lower=torch.tensor([2.0, -3.0, -2.0, 1.0]),
        upper=torch.tensor([3.0, -2.0, 3.0, 2.0]),
        power=3,
    )

    # [2, 3]^3 = [8, 27]
    # [-3, -2]^3 = [-27, -8]
    # [-2, 3]^3 = [-8, 27]
    # [1, 2]^3 = [1, 8]
    expected_lower = torch.tensor([8.0, -27.0, -8.0, 1.0])
    expected_upper = torch.tensor([27.0, -8.0, 27.0, 8.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_pow_batched_intervals_even_power() -> None:
    """Test pow with batched intervals and even power."""
    out = _propagate(
        lower=torch.tensor([2.0, -3.0, -2.0, 1.0]),
        upper=torch.tensor([3.0, -2.0, 3.0, 2.0]),
        power=2,
    )

    # [2, 3]^2 = [4, 9]
    # [-3, -2]^2 = [4, 9]
    # [-2, 3]^2 = [0, 9]
    # [1, 2]^2 = [1, 4]
    expected_lower = torch.tensor([4.0, 4.0, 0.0, 1.0])
    expected_upper = torch.tensor([9.0, 9.0, 9.0, 4.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_pow_higher_odd_power() -> None:
    """Test pow with higher odd power."""
    # [-2, 1]^5 = [-32, 1]
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([1.0]),
        power=5,
    )

    assert torch.allclose(out.lower, torch.tensor([-32.0]))
    assert torch.allclose(out.upper, torch.tensor([1.0]))


def test_pow_higher_even_power() -> None:
    """Test pow with higher even power."""
    # [-2, 1]^4 = [0, 16]
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([1.0]),
        power=4,
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([16.0]))


def test_pow_multidimensional() -> None:
    """Test pow with multi-dimensional intervals."""
    out = _propagate(
        lower=torch.tensor([[1.0, -2.0], [2.0, -3.0]]),
        upper=torch.tensor([[2.0, 1.0], [3.0, -1.0]]),
        power=2,
    )

    # [1, 2]^2 = [1, 4]
    # [-2, 1]^2 = [0, 4]
    # [2, 3]^2 = [4, 9]
    # [-3, -1]^2 = [1, 9]
    expected_lower = torch.tensor([[1.0, 0.0], [4.0, 1.0]])
    expected_upper = torch.tensor([[4.0, 4.0], [9.0, 9.0]])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_pow_tensor_power_odd() -> None:
    """Test pow with tensor of odd powers."""
    out = _propagate(
        lower=torch.tensor([2.0, -3.0]),
        upper=torch.tensor([3.0, -2.0]),
        power=torch.tensor([3, 3]),
    )

    expected_lower = torch.tensor([8.0, -27.0])
    expected_upper = torch.tensor([27.0, -8.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_pow_tensor_power_even() -> None:
    """Test pow with tensor of even powers."""
    out = _propagate(
        lower=torch.tensor([2.0, -3.0]),
        upper=torch.tensor([3.0, -2.0]),
        power=torch.tensor([2, 2]),
    )

    expected_lower = torch.tensor([4.0, 4.0])
    expected_upper = torch.tensor([9.0, 9.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_pow_tensor_power_mixed() -> None:
    """Test pow with tensor of mixed odd/even powers."""
    out = _propagate(
        lower=torch.tensor([2.0, -3.0, -2.0]),
        upper=torch.tensor([3.0, -2.0, 1.0]),
        power=torch.tensor([2, 3, 2]),
    )

    # [2, 3]^2 = [4, 9]
    # [-3, -2]^3 = [-27, -8]
    # [-2, 1]^2 = [0, 4]
    expected_lower = torch.tensor([4.0, -27.0, 0.0])
    expected_upper = torch.tensor([9.0, -8.0, 4.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_pow_monotonicity_odd_power() -> None:
    """Test that x^n is monotonic for odd n."""
    # For odd powers, if [a, b] ⊆ [c, d], then [a, b]^n ⊆ [c, d]^n
    out_inner = _propagate(
        lower=torch.tensor([1.0]),
        upper=torch.tensor([2.0]),
        power=3,
    )
    out_outer = _propagate(
        lower=torch.tensor([0.5]),
        upper=torch.tensor([3.0]),
        power=3,
    )

    assert out_outer.lower <= out_inner.lower
    assert out_outer.upper >= out_inner.upper


def test_pow_non_negativity_even_power() -> None:
    """Test that x^n is always non-negative for even n."""
    # Test various intervals
    test_cases = [
        (torch.tensor([-10.0, -2.0, 1.0]), torch.tensor([10.0, 5.0, 3.0])),
        (torch.tensor([-5.0, -3.0, -1.0]), torch.tensor([-1.0, 0.0, 2.0])),
    ]

    for lower, upper in test_cases:
        out = _propagate(lower, upper, power=2)
        assert torch.all(out.lower >= 0.0)
        assert torch.all(out.upper >= 0.0)
