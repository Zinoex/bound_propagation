from __future__ import annotations

import pytest
import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.sqrt import IBPSqrt

from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for sqrt operation."""
    strategy = IBPSqrt()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_sqrt_positive_interval() -> None:
    """Test sqrt of positive interval."""
    # sqrt([4, 9]) = [2, 3]
    out = _propagate(
        lower=torch.tensor([4.0]),
        upper=torch.tensor([9.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([3.0]))


def test_sqrt_zero_lower_bound() -> None:
    """Test sqrt with zero as lower bound."""
    # sqrt([0, 16]) = [0, 4]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([16.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0]))


def test_sqrt_zero_interval() -> None:
    """Test sqrt of zero interval."""
    # sqrt([0, 0]) = [0, 0]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_sqrt_point_interval() -> None:
    """Test sqrt of point interval (lower = upper)."""
    # sqrt([16, 16]) = [4, 4]
    out = _propagate(
        lower=torch.tensor([16.0]),
        upper=torch.tensor([16.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([4.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0]))


def test_sqrt_small_positive_interval() -> None:
    """Test sqrt of small positive interval."""
    # sqrt([0.25, 0.64]) = [0.5, 0.8]
    out = _propagate(
        lower=torch.tensor([0.25]),
        upper=torch.tensor([0.64]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.5]))
    assert torch.allclose(out.upper, torch.tensor([0.8]))


def test_sqrt_batched_intervals() -> None:
    """Test sqrt with batched intervals."""
    out = _propagate(
        lower=torch.tensor([0.0, 1.0, 4.0, 0.25]),
        upper=torch.tensor([4.0, 9.0, 16.0, 1.0]),
    )

    expected_lower = torch.tensor([0.0, 1.0, 2.0, 0.5])
    expected_upper = torch.tensor([2.0, 3.0, 4.0, 1.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sqrt_multidimensional() -> None:
    """Test sqrt with multi-dimensional intervals."""
    out = _propagate(
        lower=torch.tensor([[0.0, 1.0], [4.0, 9.0]]),
        upper=torch.tensor([[4.0, 16.0], [25.0, 36.0]]),
    )

    expected_lower = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    expected_upper = torch.tensor([[2.0, 4.0], [5.0, 6.0]])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_sqrt_negative_input_raises_error() -> None:
    """Test that sqrt with negative input raises ValueError."""
    with pytest.raises(ValueError, match="sqrt requires non-negative input bounds"):
        _propagate(
            lower=torch.tensor([-1.0]),
            upper=torch.tensor([4.0]),
        )


def test_sqrt_all_negative_raises_error() -> None:
    """Test that sqrt with all negative inputs raises ValueError."""
    with pytest.raises(ValueError, match="sqrt requires non-negative input bounds"):
        _propagate(
            lower=torch.tensor([-9.0]),
            upper=torch.tensor([-4.0]),
        )


def test_sqrt_monotonicity() -> None:
    """Test that sqrt is monotonically increasing."""
    # If [a, b] ⊆ [c, d], then sqrt([a, b]) ⊆ sqrt([c, d])
    inner = IntervalBounds(torch.tensor([4.0]), torch.tensor([9.0]))
    outer = IntervalBounds(torch.tensor([1.0]), torch.tensor([16.0]))

    strategy = IBPSqrt()

    out_inner = propagate(strategy, inner)
    out_outer = propagate(strategy, outer)

    # sqrt([4, 9]) = [2, 3] should be contained in sqrt([1, 16]) = [1, 4]
    assert out_outer.lower <= out_inner.lower
    assert out_outer.upper >= out_inner.upper


def test_sqrt_composition_with_square() -> None:
    """Test that sqrt(x^2) = x for positive x."""
    # For positive x, sqrt(x^2) = x
    # Manually compute x^2 then sqrt
    a_lower = torch.tensor([2.0, 3.0])
    a_upper = torch.tensor([4.0, 5.0])

    # x^2 for positive x
    a_squared_lower = a_lower**2
    a_squared_upper = a_upper**2

    # sqrt(x^2)
    sqrt_strategy = IBPSqrt()
    a_squared = IntervalBounds(a_squared_lower, a_squared_upper)
    result = propagate(sqrt_strategy, a_squared)

    # For positive x, this should recover the original interval
    assert torch.allclose(result.lower, a_lower, rtol=1e-5)
    assert torch.allclose(result.upper, a_upper, rtol=1e-5)


def test_sqrt_narrowing_property() -> None:
    """Test that sqrt narrows intervals for values > 1 and widens for values < 1."""
    # For [a, b] where a, b > 1, sqrt([a, b]) has smaller width
    large = _propagate(torch.tensor([4.0]), torch.tensor([16.0]))
    large_width_in = 16.0 - 4.0
    large_width_out = large.upper - large.lower
    assert large_width_out < large_width_in  # 4 - 2 = 2 < 12

    # For [a, b] where 0 < a, b < 1, sqrt([a, b]) has larger width
    small = _propagate(torch.tensor([0.04]), torch.tensor([0.16]))
    small_width_in = 0.16 - 0.04
    small_width_out = small.upper - small.lower
    assert small_width_out > small_width_in  # 0.4 - 0.2 = 0.2 > 0.12
