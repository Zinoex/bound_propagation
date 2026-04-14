from __future__ import annotations

import torch

from bound_propagation.bounds import IntervalBounds
from bound_propagation.propagation.ibp.neg import IBPNeg
from bound_propagation.propagation.ibp.sigmoid import IBPSigmoid
from bound_propagation.propagation.ibp.tanh import IBPTanh
from tests.helpers import propagate


def _propagate(lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
    """Propagate bounds for tanh operation."""
    strategy = IBPTanh()
    bounds = IntervalBounds(lower=lower, upper=upper)
    return propagate(strategy, bounds)


def test_tanh_positive_interval() -> None:
    """Test tanh of positive interval."""
    # tanh([1, 2]) ≈ [0.762, 0.964]
    out = _propagate(
        lower=torch.tensor([1.0]),
        upper=torch.tensor([2.0]),
    )

    expected_lower = torch.tanh(torch.tensor([1.0]))
    expected_upper = torch.tanh(torch.tensor([2.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_tanh_negative_interval() -> None:
    """Test tanh of negative interval."""
    # tanh([-2, -1]) ≈ [-0.964, -0.762]
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([-1.0]),
    )

    expected_lower = torch.tanh(torch.tensor([-2.0]))
    expected_upper = torch.tanh(torch.tensor([-1.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_tanh_mixed_sign_interval() -> None:
    """Test tanh of interval with mixed signs."""
    # tanh([-1, 1]) ≈ [-0.762, 0.762]
    out = _propagate(
        lower=torch.tensor([-1.0]),
        upper=torch.tensor([1.0]),
    )

    expected_lower = torch.tanh(torch.tensor([-1.0]))
    expected_upper = torch.tanh(torch.tensor([1.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_tanh_zero() -> None:
    """Test tanh(0) = 0."""
    out = _propagate(
        lower=torch.tensor([0.0]),
        upper=torch.tensor([0.0]),
    )

    assert torch.allclose(out.lower, torch.tensor([0.0]))
    assert torch.allclose(out.upper, torch.tensor([0.0]))


def test_tanh_interval_containing_zero() -> None:
    """Test tanh of interval containing zero."""
    # tanh([-2, 2]) should contain 0
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([2.0]),
    )

    # Should contain 0
    assert out.lower < 0.0
    assert out.upper > 0.0


def test_tanh_point_interval() -> None:
    """Test tanh of point interval (lower = upper)."""
    out = _propagate(
        lower=torch.tensor([1.5]),
        upper=torch.tensor([1.5]),
    )

    expected = torch.tanh(torch.tensor([1.5]))
    assert torch.allclose(out.lower, expected)
    assert torch.allclose(out.upper, expected)


def test_tanh_batched_intervals() -> None:
    """Test tanh with batched intervals."""
    lower = torch.tensor([0.0, -1.0, 1.0, -2.0])
    upper = torch.tensor([1.0, 0.0, 2.0, -1.0])

    out = _propagate(lower, upper)

    expected_lower = torch.tanh(lower)
    expected_upper = torch.tanh(upper)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_tanh_multidimensional() -> None:
    """Test tanh with multi-dimensional intervals."""
    lower = torch.tensor([[-1.0, 0.0], [1.0, -2.0]])
    upper = torch.tensor([[1.0, 2.0], [3.0, 0.0]])

    out = _propagate(lower, upper)

    expected_lower = torch.tanh(lower)
    expected_upper = torch.tanh(upper)

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)


def test_tanh_large_positive_values() -> None:
    """Test tanh with large positive values (approaches 1)."""
    # tanh([5, 10]) should be very close to [1, 1]
    out = _propagate(
        lower=torch.tensor([5.0]),
        upper=torch.tensor([10.0]),
    )

    expected_lower = torch.tanh(torch.tensor([5.0]))
    expected_upper = torch.tanh(torch.tensor([10.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)
    # Should be very close to 1
    assert out.lower > 0.9999
    assert out.upper > 0.9999


def test_tanh_large_negative_values() -> None:
    """Test tanh with large negative values (approaches -1)."""
    # tanh([-10, -5]) should be very close to [-1, -1]
    out = _propagate(
        lower=torch.tensor([-10.0]),
        upper=torch.tensor([-5.0]),
    )

    expected_lower = torch.tanh(torch.tensor([-10.0]))
    expected_upper = torch.tanh(torch.tensor([-5.0]))

    assert torch.allclose(out.lower, expected_lower)
    assert torch.allclose(out.upper, expected_upper)
    # Should be very close to -1
    assert out.lower < -0.9999
    assert out.upper < -0.9999


def test_tanh_bounded_by_interval() -> None:
    """Test that tanh always returns bounds in [-1, 1]."""
    # Test various intervals
    test_cases = [
        (torch.tensor([-100.0]), torch.tensor([100.0])),
        (torch.tensor([-10.0]), torch.tensor([10.0])),
        (torch.tensor([0.0]), torch.tensor([5.0])),
    ]

    for lower, upper in test_cases:
        out = _propagate(lower, upper)
        # Tanh approaches but can equal -1 and 1 in the limits
        assert torch.all(out.lower >= -1.0)
        assert torch.all(out.upper <= 1.0)


def test_tanh_monotonicity() -> None:
    """Test that tanh is monotonically increasing."""
    # If [a, b] ⊆ [c, d], then tanh([a, b]) ⊆ tanh([c, d])
    inner = IntervalBounds(torch.tensor([0.5]), torch.tensor([1.5]))
    outer = IntervalBounds(torch.tensor([0.0]), torch.tensor([2.0]))

    strategy = IBPTanh()

    out_inner = propagate(strategy, inner)
    out_outer = propagate(strategy, outer)

    assert out_outer.lower <= out_inner.lower
    assert out_outer.upper >= out_inner.upper


def test_tanh_odd_function_property() -> None:
    """Test that tanh is an odd function: tanh(-x) = -tanh(x)."""
    a = IntervalBounds(torch.tensor([1.0, 0.5]), torch.tensor([2.0, 1.5]))

    strategy = IBPTanh()

    neg_strategy = IBPNeg()

    # tanh(a)
    tanh_a = propagate(strategy, a)

    # -a
    neg_a = propagate(neg_strategy, a)

    # tanh(-a)
    tanh_neg_a = propagate(strategy, neg_a)

    # -tanh(a)
    neg_tanh_a = propagate(neg_strategy, tanh_a)

    # tanh(-x) should equal -tanh(x)
    assert torch.allclose(tanh_neg_a.lower, neg_tanh_a.lower)
    assert torch.allclose(tanh_neg_a.upper, neg_tanh_a.upper)


def test_tanh_symmetric_interval() -> None:
    """Test tanh of symmetric interval around zero."""
    # tanh([-a, a]) should give symmetric result
    out = _propagate(
        lower=torch.tensor([-2.0]),
        upper=torch.tensor([2.0]),
    )

    # Should be symmetric: lower = -upper
    assert torch.allclose(out.lower, -out.upper)


def test_tanh_relation_to_sigmoid() -> None:
    """Test that tanh(x) = 2*sigmoid(2*x) - 1."""

    a = IntervalBounds(torch.tensor([0.5]), torch.tensor([1.5]))

    tanh_strategy = IBPTanh()
    sigmoid_strategy = IBPSigmoid()

    # tanh(a)
    tanh_a = propagate(tanh_strategy, a)

    # 2*a
    two_a = IntervalBounds(2 * a.lower, 2 * a.upper)

    # sigmoid(2*a)
    sig_2a = propagate(sigmoid_strategy, two_a)

    # 2*sigmoid(2*a) - 1
    result = IntervalBounds(2 * sig_2a.lower - 1, 2 * sig_2a.upper - 1)

    # Should be approximately equal
    assert torch.allclose(tanh_a.lower, result.lower, atol=1e-6)
    assert torch.allclose(tanh_a.upper, result.upper, atol=1e-6)
