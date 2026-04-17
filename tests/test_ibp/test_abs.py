from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.elementwise import IBPAbs
from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for abs operation."""
    strategy = IBPAbs()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_abs_positive_interval() -> None:
    """Test abs of positive interval."""
    # abs([2, 5]) = [2, 5]
    out = _propagate(
        lower=torch.tensor([2.0]),
        upper=torch.tensor([5.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_abs_negative_interval() -> None:
    """Test abs of negative interval."""
    # abs([-5, -2]) = [2, 5]
    out = _propagate(
        lower=torch.tensor([-5.0]),
        upper=torch.tensor([-2.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([2.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_abs_mixed_sign_interval() -> None:
    """Test abs of interval containing zero."""
    # abs([-3, 4]) = [0, 4]
    # The lower bound should be 0 since the interval contains zero
    out = _propagate(
        lower=torch.tensor([-3.0]),
        upper=torch.tensor([4.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([4.0]))


def test_abs_mixed_sign_larger_negative() -> None:
    """Test abs of interval containing zero with larger negative magnitude."""
    # abs([-7, 3]) = [0, 7]
    out = _propagate(
        lower=torch.tensor([-7.0]),
        upper=torch.tensor([3.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([7.0]))


def test_abs_mixed_sign_larger_positive() -> None:
    """Test abs of interval containing zero with larger positive magnitude."""
    # abs([-2, 8]) = [0, 8]
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([8.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([8.0]))


def test_abs_zero_interval() -> None:
    """Test abs of zero interval."""
    # abs([0, 0]) = [0, 0]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_abs_symmetric_interval() -> None:
    """Test abs of symmetric interval around zero."""
    # abs([-5, 5]) = [0, 5]
    out = _propagate(
        lower=torch.tensor([-5.0]),
        upper=torch.tensor([5.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([5.0]))


def test_abs_batched_intervals() -> None:
    """Test abs with batched intervals of different types."""
    out = _propagate(
        lower=torch.tensor([2.0, -5.0, -3.0, 0.0, -7.0]),
        upper=torch.tensor([5.0, -2.0, 4.0, 3.0, 5.0]),
    )

    # [2, 5]: positive -> [2, 5]
    # [-5, -2]: negative -> [2, 5]
    # [-3, 4]: mixed -> [0, 4]
    # [0, 3]: non-negative -> [0, 3]
    # [-7, 5]: mixed, larger negative -> [0, 7]
    expected_lower = torch.tensor([2.0, 2.0, 0.0, 0.0, 0.0])
    expected_upper = torch.tensor([5.0, 5.0, 4.0, 3.0, 7.0])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_abs_multidimensional() -> None:
    """Test abs with multi-dimensional intervals."""
    out = _propagate(
        lower=torch.tensor([[-3.0, 1.0], [-2.0, -5.0]]),
        upper=torch.tensor([[4.0, 6.0], [7.0, -1.0]]),
    )

    expected_lower = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    expected_upper = torch.tensor([[4.0, 6.0], [7.0, 5.0]])

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_abs_non_negativity_property() -> None:
    """Test that abs always returns non-negative bounds."""
    # Test various intervals
    test_cases = [
        torch.tensor([[-10.0, 5.0, -3.0]]),
        torch.tensor([[100.0, -200.0, 0.0]]),
    ]

    for lower in test_cases:
        upper = lower + 10.0  # Create valid upper bound
        out = _propagate(lower, upper)

        # Lower bound should always be >= 0
        assert torch.all(out.lower >= 0.0)
        # Upper bound should always be >= 0
        assert torch.all(out.upper >= 0.0)


def test_abs_preservation_of_width() -> None:
    """Test that abs preserves or reduces interval width for non-zero-containing intervals."""
    # For intervals not containing zero, width is preserved
    # [2, 5]: width = 3
    out1 = _propagate(torch.tensor([2.0]), torch.tensor([5.0]))
    width1 = out1.upper - out1.lower
    assert torch.allclose(width1, torch.tensor([3.0]))

    # [-5, -2]: width = 3
    out2 = _propagate(torch.tensor([-5.0]), torch.tensor([-2.0]))
    width2 = out2.upper - out2.lower
    assert torch.allclose(width2, torch.tensor([3.0]))


def test_abs_idempotency_property() -> None:
    """Test that abs(abs(x)) = abs(x)."""
    strategy = IBPAbs()
    a = IntervalBounds(torch.tensor([-3.0, 1.0]), torch.tensor([4.0, 6.0]))

    # abs(a)
    abs_a = propagate(strategy, a)

    # abs(abs(a))
    abs_abs_a = propagate(strategy, abs_a)

    assert torch.allclose(abs_a.lower, abs_abs_a.lower)
    assert torch.allclose(abs_a.upper, abs_abs_a.upper)
