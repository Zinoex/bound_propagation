from __future__ import annotations

import pytest
import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.exp import IBPExp
from bound_propagation.propagation.ibp.log import IBPLog

from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for log operation."""
    strategy = IBPLog()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_log_positive_interval() -> None:
    """Test log of positive interval."""
    # log([e, e^2]) = [1, 2]
    e = torch.e
    out = _propagate(
        lower=torch.tensor([e]),
        upper=torch.tensor([e**2]),
    )

    assert torch.allclose(out.lower, torch.tensor([1.0]))
    assert torch.allclose(out.upper, torch.tensor([2.0]))


def test_log_positive_interval_less_than_one() -> None:
    """Test log of positive interval less than 1."""
    # log([e^-2, e^-1]) = [-2, -1]
    e = torch.e
    out = _propagate(
        lower=torch.tensor([e**-2]),
        upper=torch.tensor([e**-1]),
    )

    assert torch.allclose(out.lower, torch.tensor([-2.0]))
    assert torch.allclose(out.upper, torch.tensor([-1.0]))


def test_log_interval_around_one() -> None:
    """Test log of interval around 1."""
    # log([0.5, 2]) = [log(0.5), log(2)]
    out = _propagate(
        lower=torch.tensor([0.5]),
        upper=torch.tensor([2.0]),
    )

    expected_lower = torch.log(torch.tensor([0.5]))
    expected_upper = torch.log(torch.tensor([2.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_log_at_one() -> None:
    """Test log(1) = 0."""
    out = _propagate(
        lower=torch.tensor([1.0]),
        upper=torch.tensor([1.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_log_point_interval() -> None:
    """Test log of point interval (lower = upper)."""
    # log([2, 2]) = [log(2), log(2)]
    out = _propagate(
        lower=torch.tensor([2.0]),
        upper=torch.tensor([2.0]),
    )

    expected = torch.log(torch.tensor([2.0]))
    assert torch.allclose(out.lower, expected)
    assert torch.allclose(out.upper, expected)


def test_log_batched_intervals() -> None:
    """Test log with batched intervals."""
    e = torch.e
    lower = torch.tensor([1.0, e**-1, e, 0.5])
    upper = torch.tensor([e, 1.0, e**2, 2.0])

    out = _propagate(lower, upper)

    expected_lower = torch.log(lower)
    expected_upper = torch.log(upper)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_log_multidimensional() -> None:
    """Test log with multi-dimensional intervals."""
    e = torch.e
    lower = torch.tensor([[1.0, e], [0.5, e**-1]])
    upper = torch.tensor([[e, e**2], [2.0, 1.0]])

    out = _propagate(lower, upper)

    expected_lower = torch.log(lower)
    expected_upper = torch.log(upper)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_log_zero_lower_raises_error() -> None:
    """Test that log with zero lower bound raises ValueError."""
    with pytest.raises(ValueError, match="log requires positive input bounds"):
        _propagate(
            lower=torch.tensor([0.0]),
            upper=torch.tensor([2.0]),
        )


def test_log_negative_input_raises_error() -> None:
    """Test that log with negative input raises ValueError."""
    with pytest.raises(ValueError, match="log requires positive input bounds"):
        _propagate(
            lower=torch.tensor([-1.0]),
            upper=torch.tensor([2.0]),
        )


def test_log_all_negative_raises_error() -> None:
    """Test that log with all negative inputs raises ValueError."""
    with pytest.raises(ValueError, match="log requires positive input bounds"):
        _propagate(
            lower=torch.tensor([-5.0]),
            upper=torch.tensor([-1.0]),
        )


def test_log_monotonicity() -> None:
    """Test that log is monotonically increasing."""
    # If [a, b] ⊆ [c, d] and all positive, then log([a, b]) ⊆ log([c, d])
    inner = IntervalBounds(torch.tensor([2.0]), torch.tensor([4.0]))
    outer = IntervalBounds(torch.tensor([1.0]), torch.tensor([8.0]))

    strategy = IBPLog()

    out_inner = propagate(strategy, inner)
    out_outer = propagate(strategy, outer)

    assert out_outer.lower <= out_inner.lower
    assert out_outer.upper >= out_inner.upper


def test_log_composition_with_exp() -> None:
    """Test that log(exp(x)) = x."""

    a = IntervalBounds(torch.tensor([-2.0, 0.0]), torch.tensor([1.0, 3.0]))

    exp_strategy = IBPExp()
    log_strategy = IBPLog()

    # exp(a)
    exp_a = propagate(exp_strategy, a)

    # log(exp(a))
    result = propagate(log_strategy, exp_a)

    # Should recover the original interval
    assert torch.allclose(result.lower, a.lower, rtol=1e-5)
    assert torch.allclose(result.upper, a.upper, rtol=1e-5)


def test_log_small_positive_values() -> None:
    """Test log with very small positive values."""
    # log([0.001, 0.01]) should be very negative but finite
    out = _propagate(
        lower=torch.tensor([0.001]),
        upper=torch.tensor([0.01]),
    )

    expected_lower = torch.log(torch.tensor([0.001]))
    expected_upper = torch.log(torch.tensor([0.01]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)
    assert torch.isfinite(out.lower).all()
    assert torch.isfinite(out.upper).all()
    assert torch.all(out.lower < 0)
    assert torch.all(out.upper < 0)


def test_log_large_positive_values() -> None:
    """Test log with large positive values."""
    # log([100, 1000]) should be positive and finite
    out = _propagate(
        lower=torch.tensor([100.0]),
        upper=torch.tensor([1000.0]),
    )

    expected_lower = torch.log(torch.tensor([100.0]))
    expected_upper = torch.log(torch.tensor([1000.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)
    assert torch.isfinite(out.lower).all()
    assert torch.isfinite(out.upper).all()
    assert torch.all(out.lower > 0)
    assert torch.all(out.upper > 0)
