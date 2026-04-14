from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.exp import IBPExp
from bound_propagation.propagation.ibp.log import IBPLog

from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for exp operation."""
    strategy = IBPExp()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_exp_positive_interval() -> None:
    """Test exp of positive interval."""
    # exp([1, 2]) ≈ [e, e^2] ≈ [2.718, 7.389]
    out = _propagate(
        lower=torch.tensor([1.0]),
        upper=torch.tensor([2.0]),
    )

    expected_lower = torch.exp(torch.tensor([1.0]))
    expected_upper = torch.exp(torch.tensor([2.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_exp_negative_interval() -> None:
    """Test exp of negative interval."""
    # exp([-2, -1]) ≈ [e^-2, e^-1] ≈ [0.135, 0.368]
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([-1.0]),
    )

    expected_lower = torch.exp(torch.tensor([-2.0]))
    expected_upper = torch.exp(torch.tensor([-1.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_exp_mixed_sign_interval() -> None:
    """Test exp of interval with mixed signs."""
    # exp([-1, 1]) ≈ [e^-1, e] ≈ [0.368, 2.718]
    out = _propagate(
        lower=torch.tensor([-1.0]),
        upper=torch.tensor([1.0]),
    )

    expected_lower = torch.exp(torch.tensor([-1.0]))
    expected_upper = torch.exp(torch.tensor([1.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_exp_zero_interval() -> None:
    """Test exp of zero interval."""
    # exp([0, 0]) = [1, 1]
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([1.0]))
    assert torch.allclose(out.upper, torch.tensor([1.0]))


def test_exp_interval_containing_zero() -> None:
    """Test exp of interval containing zero."""
    # exp([-1, 1]) should have exp(-1) < 1 < exp(1)
    out = _propagate(
        lower=torch.tensor([-1.0]),
        upper=torch.tensor([1.0]),
    )

    # Should contain 1
    assert out.lower < 1.0
    assert out.upper > 1.0


def test_exp_point_interval() -> None:
    """Test exp of point interval (lower = upper)."""
    # exp([2, 2]) = [e^2, e^2]
    out = _propagate(
        lower=torch.tensor([2.0]),
        upper=torch.tensor([2.0]),
    )

    expected = torch.exp(torch.tensor([2.0]))
    assert torch.allclose(out.lower, expected)
    assert torch.allclose(out.upper, expected)


def test_exp_batched_intervals() -> None:
    """Test exp with batched intervals."""
    lower = torch.tensor([0.0, -1.0, 1.0, -2.0])
    upper = torch.tensor([1.0, 0.0, 2.0, -1.0])

    out = _propagate(lower, upper)

    expected_lower = torch.exp(lower)
    expected_upper = torch.exp(upper)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_exp_multidimensional() -> None:
    """Test exp with multi-dimensional intervals."""
    lower = torch.tensor([[-1.0, 0.0], [1.0, -2.0]])
    upper = torch.tensor([[1.0, 2.0], [3.0, 0.0]])

    out = _propagate(lower, upper)

    expected_lower = torch.exp(lower)
    expected_upper = torch.exp(upper)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_exp_large_positive_values() -> None:
    """Test exp with large positive values."""
    # exp([5, 10]) should be large but finite
    out = _propagate(
        lower=torch.tensor([5.0]),
        upper=torch.tensor([10.0]),
    )

    expected_lower = torch.exp(torch.tensor([5.0]))
    expected_upper = torch.exp(torch.tensor([10.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)
    assert torch.isfinite(out.lower).all()
    assert torch.isfinite(out.upper).all()


def test_exp_large_negative_values() -> None:
    """Test exp with large negative values (approaches 0)."""
    # exp([-10, -5]) should be very small but positive
    out = _propagate(
        lower=torch.tensor([-10.0]),
        upper=torch.tensor([-5.0]),
    )

    expected_lower = torch.exp(torch.tensor([-10.0]))
    expected_upper = torch.exp(torch.tensor([-5.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)
    assert torch.all(out.lower > 0)
    assert torch.all(out.upper > 0)


def test_exp_monotonicity() -> None:
    """Test that exp is monotonically increasing."""
    # If [a, b] ⊆ [c, d], then exp([a, b]) ⊆ exp([c, d])
    inner = IntervalBounds(torch.tensor([0.5]), torch.tensor([1.5]))
    outer = IntervalBounds(torch.tensor([0.0]), torch.tensor([2.0]))

    strategy = IBPExp()

    out_inner = propagate(strategy, inner)
    out_outer = propagate(strategy, outer)

    assert out_outer.lower <= out_inner.lower
    assert out_outer.upper >= out_inner.upper


def test_exp_always_positive() -> None:
    """Test that exp always returns positive bounds."""
    # Test various intervals including very negative values
    test_cases = [
        (torch.tensor([-100.0]), torch.tensor([-50.0])),
        (torch.tensor([-10.0]), torch.tensor([10.0])),
        (torch.tensor([0.0]), torch.tensor([5.0])),
    ]

    for lower, upper in test_cases:
        out = _propagate(lower, upper)
        assert torch.all(out.lower > 0)
        assert torch.all(out.upper > 0)


def test_exp_composition_with_log() -> None:
    """Test that exp(log(x)) = x for positive x."""

    # Positive interval
    a = IntervalBounds(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 5.0]))

    log_strategy = IBPLog()
    exp_strategy = IBPExp()

    # log(a)
    log_a = propagate(log_strategy, a)

    # exp(log(a))
    result = propagate(exp_strategy, log_a)

    # Should recover the original interval
    assert torch.allclose(result.lower, a.lower, rtol=1e-5)
    assert torch.allclose(result.upper, a.upper, rtol=1e-5)


def test_exp_identity_at_zero() -> None:
    """Test that exp(0) = 1."""
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([1.0]))
    assert torch.allclose(out.upper, torch.tensor([1.0]))
